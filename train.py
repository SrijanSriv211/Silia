import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from model import Silia, Config
from encoder import Encoder
from optimizer import MuonDist, Muon

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from colorama import Style, Fore, init
from rich.progress import track
from itertools import chain

import random, pickle, torch, regex, json, time, math, sys

def print0(*text, println=True, overwrite=False, save_to_file=True, log_path="bin"):
	if println:
		print(*text)

	if not save_to_file:
		return

	# save cleaned text to the file
	if not os.path.isdir(log_path):
		os.mkdir(log_path)

	with open(os.path.join(log_path, "out.txt"), "w" if overwrite else "a", encoding="utf-8") as f:
		f.write(" ".join(tuple(ansi_escape.sub('', part) for part in text)) + "\n")

# separate the integer part (for hours, minutes, and seconds) from the fractional part (for milliseconds)
def calc_total_time(seconds):
	sec_int, millis = divmod(seconds, 1)
	millis = int(millis * 1000) # convert the fractional part to milliseconds

	min, sec = divmod(int(sec_int), 60)
	hour, min = divmod(min, 60)
	hours, minutes, seconds = int(hour), int(min), int(sec)

	t = [
		f"{hours} hour" + ("s" if hours > 1 else "") if hours > 0 else None,
		f"{minutes} minute" + ("s" if minutes > 1 else "") if minutes > 0 else None,
		f"{seconds} second" + ("s" if seconds > 1 else "") if seconds > 0 else None,
		f"{millis} ms" if millis > 0 else None
	]
	t = list(filter(None, t))

	return ", ".join(t) if t else "0 seconds"

# load config
CONFIG = json.loads(open(sys.argv[1], "r", encoding="utf-8").read()) if len(sys.argv) > 1 else {
	"dataset": {
		"data_division": 0.8,
		"load_from_file": True,
		"path": "data/webtext.bin"
	},
	"checkpoints": {
		"path": "bin/c1",
		"interval": 2000,
		"create_checkpoints": True
	},
	"model_hyperparams": {
		"vocab_size": 16384,
		"block_size": 1024,
		"n_layer": 4,
		"n_head": 16,
		"n_embd": 128
	},
	"optimizer_hyperparams": {
		"eps": 1e-10,
		"beta1": 0.9,
		"beta2": 0.95,
		"weight_decay": 1e-1,
		"use_muon": True,
		"momentum": 0.95
	},
	"encoder_path": "bin/cl8k.bin",
	"init_from": "scratch",
	"seed": 18,

	"gradient_accumulation_steps": 1,
	"batch_size": 4,

	"max_iters": 50000,
	"eval_interval": 2000,
	"log_interval": 200,
	"eval_iters": 200,

	"decay_lr": True,
	"lr_decay_iters": 50000,
	"learning_rate": 3e-3,
	"cooldown_frac": 0.4,
	"warmup_iters": 2000,
	"min_lr": 3e-4
}

# save the text in a text file
init(autoreset=True)
ansi_escape = regex.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
	init_process_group(backend='nccl')
	ddp_rank = int(os.environ['RANK'])
	ddp_local_rank = int(os.environ['LOCAL_RANK'])
	ddp_world_size = int(os.environ['WORLD_SIZE'])
	device = f'cuda:{ddp_local_rank}'
	torch.cuda.set_device(device)
	master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
	seed_offset = ddp_rank # each process gets a different seed

# if not ddp, we are running on a single gpu, and one process
else:
	master_process = True
	seed_offset = 0
	ddp_world_size = 1

# init seed
if CONFIG["seed"] != "auto":
	if torch.cuda.is_available():
		torch.cuda.manual_seed(CONFIG["seed"] + seed_offset)
	torch.manual_seed(CONFIG["seed"] + seed_offset)
	random.seed(CONFIG["seed"] + seed_offset)

if master_process:
	os.makedirs(CONFIG["checkpoints"]["path"], exist_ok=True)
log_path = CONFIG["checkpoints"]["path"]

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# set device
device_type = "cuda" if torch.cuda.is_available() else "cpu"
init_from = CONFIG["init_from"][11:] if CONFIG["init_from"].startswith("pretrained,") else "scratch"

# print the device
if master_process:
	print0(f"config: {Fore.WHITE}{Style.DIM}`{json.dumps(CONFIG)}`", overwrite=(init_from == "scratch"), log_path=log_path)
	print0("Training on", f"{Fore.YELLOW}{Style.BRIGHT}{device_type}", log_path=log_path)

# load stats
checkpoint = None if init_from == "scratch" else torch.load(init_from)
stats = checkpoint["stats"] if checkpoint is not None and "stats" in checkpoint.keys() else {
	"step": 0,
	"loss": {
		"train": [],
		"test": [],
		"val": []
	},
	"lr": []
}

# create an instance of Silia
hyperparams = CONFIG["model_hyperparams"] if checkpoint is None else checkpoint["hyperparams"]
conf = Config(**hyperparams)
model = Silia(conf)

# load the state dict
if checkpoint is not None:
	model.load_state_dict(checkpoint["model"])
model.to(device_type)

# optimizers!
optimizer_hyperparams = CONFIG["optimizer_hyperparams"] if checkpoint is None else checkpoint["optimizer_hyperparams"]

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
adam_params = embed_params

if not optimizer_hyperparams["use_muon"]:
	adam_params = embed_params + hidden_matrix_params

# init the optimizer(s)
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = torch.optim.AdamW(
	adam_params, lr=CONFIG["learning_rate"], betas=(optimizer_hyperparams["beta1"], optimizer_hyperparams["beta2"]),
	eps=optimizer_hyperparams["eps"], weight_decay=optimizer_hyperparams["weight_decay"], fused=True
)
optimizers = [optimizer1]

if optimizer_hyperparams["use_muon"]:
	muon = MuonDist if ddp else Muon
	optimizer2 = muon(
		hidden_matrix_params,
		lr=CONFIG["learning_rate"],
		momentum=optimizer_hyperparams["momentum"],
		weight_decay=optimizer_hyperparams["weight_decay"]
	)
	optimizers.append(optimizer2)

# load optimizer(s) state dict if loading from checkpoint
if checkpoint is not None:
	for o, s in zip(optimizers, checkpoint["optimizers"]):
		o.load_state_dict(s)

class dataloader:
	def __init__(self, path, block_size, batch_size, data_division=0.8, isfile=True):
		self.path = path
		self.data_division = data_division
		self.block_size, self.batch_size = block_size, batch_size

		self.files = [path] if isfile else [os.path.join(path, i) for i in os.listdir(path) if os.path.isfile(os.path.join(path, i))]

	def load_dataset(self):
		self.train, self.val = [], []

		for file in self.files:
			with open(file, "rb") as f:
				dataset = pickle.load(f)["dataset"]

			random.shuffle(dataset)
			flat_dataset = chain.from_iterable(dataset)
			# for data in dataset:
			# 	n_train_toks = int(len(data) * self.data_division)
			# 	self.train.extend(data[:n_train_toks])
			# 	self.val.extend(data[n_train_toks:])
			del dataset

			flat_dataset = list(flat_dataset)
			n_train_toks = int(len(flat_dataset) * self.data_division)

			self.train.extend(flat_dataset[:n_train_toks])
			self.val.extend(flat_dataset[n_train_toks:])

		n_train_toks, n_val_toks = len(self.train), len(self.val)
		self.train = torch.tensor(self.train, dtype=torch.int64)
		self.val = torch.tensor(self.val, dtype=torch.int64)
		return n_train_toks, n_val_toks

	def next_batch(self, split):
		data = self.train if split == "train" else self.val
		ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
		x = torch.stack([data[i:i + self.block_size] for i in ix])
		y = torch.stack([data[i+1:i+1 + self.block_size] for i in ix])
		return x.to(device_type), y.to(device_type)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, next_batch):
	out = {}
	model.eval()
	for split in ["train", "val"]:
		losses = torch.zeros(CONFIG["eval_iters"])
		for k in track(range(CONFIG["eval_iters"]), description=f"{Fore.WHITE}{Style.BRIGHT}calc {Fore.WHITE}{Style.DIM}{split} loss{Style.RESET_ALL}"):
			X, Y = next_batch(split)
			_, loss = model(X, Y)
			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out

def get_state(model, optimizers):
	state_dict = model.state_dict()
	unwanted_prefix = '_orig_mod.'

	for k, v in list(state_dict.items()):
		if k.startswith(unwanted_prefix):
			state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

	return {
		"device": device_type,
		"model": state_dict,
		"hyperparams": hyperparams
	}

# load encoder
enc = Encoder()
enc.load(CONFIG["encoder_path"])

# load dataset
dataset = dataloader(
	CONFIG["dataset"]["path"],
	hyperparams["block_size"], CONFIG["batch_size"],
	CONFIG["dataset"]["data_division"], CONFIG["dataset"]["load_from_file"]
)
n_train_toks, n_val_toks = dataset.load_dataset()

if master_process:
	print0(f"{Fore.WHITE}{Style.BRIGHT}{((n_train_toks + n_val_toks)/1e6)}M", "total tokens", log_path=log_path)
	print0(
		f"{Fore.WHITE}{Style.BRIGHT}{(n_train_toks/1e6)}M", "train tokens,",
		f"{Fore.WHITE}{Style.BRIGHT}{(n_val_toks/1e6)}M", "val tokens",
		log_path=log_path
	)

	# report number of parameters
	print0(
		f"{Fore.WHITE}{Style.BRIGHT}{sum(p.numel() for p in model.parameters())/1e6}M", "parameters,",
		f"{Fore.WHITE}{Style.BRIGHT}{sum(p.numel() for p in model.blocks.parameters())/1e6}M", "non-embedding parameters",
		log_path=log_path
	)

	# training loop
	# start training the model
	print0("started training", log_path=log_path)

# wrap model into DDP container
if ddp:
	model = DDP(model, device_ids=[ddp_local_rank])

start_time = eval_t0 = test_t0 = time.time()
n_steps = CONFIG["max_iters"] - stats["step"] + 1
steps_per_epoch = int((n_train_toks + n_val_toks) / (hyperparams["block_size"] * CONFIG["batch_size"]))
raw_model = model.module if ddp else model # unwrap DDP container if needed

for _ in range(n_steps):
	# determine and set the learning rate for this iteration
	## learning rate decay scheduler (cosine with warmup)
	if not CONFIG["decay_lr"]:
		lr = CONFIG["learning_rate"]

	## 1) linear warmup for warmup_iters steps
	elif stats["step"] < CONFIG["warmup_iters"]:
		lr = CONFIG["learning_rate"] * (stats["step"] + 1) / (CONFIG["warmup_iters"] + 1)

	## 2) constant learning rate for some time
	elif stats["step"] / CONFIG["lr_decay_iters"] <= 1 - CONFIG["cooldown_frac"]:
		lr = CONFIG["learning_rate"]

	## 3) if stats["step"] > lr_decay_iters, lr = min learning rate
	elif stats["step"] > CONFIG["lr_decay_iters"]:
		lr = CONFIG["min_lr"]

	## 4) in between, use cosine decay down to min learning rate
	else:
		const_lr_iters = int((1 - CONFIG["cooldown_frac"]) * CONFIG["lr_decay_iters"])
		decay_ratio = (stats["step"] - const_lr_iters) / (CONFIG["lr_decay_iters"] - const_lr_iters)

		assert 0 <= decay_ratio <= 1
		coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
		lr = CONFIG["min_lr"] + coeff * (CONFIG["learning_rate"] - CONFIG["min_lr"])

	## set optimizers' learning rate
	for o in optimizers:
		for group in o.param_groups:
			group["lr"] = lr
	stats["lr"].append(lr)

	# training section
	for micro_step in range(CONFIG["gradient_accumulation_steps"]):
		X, Y = dataset.next_batch("train")
		# in DDP training we only need to sync gradients at the last micro step.
		# the official way to do this is with model.no_sync() context manager, but
		# I really dislike that this bloats the code and forces us to repeat code
		# looking at the source of that context manager, it just toggles this variable
		if ddp:
			model.require_backward_grad_sync = (micro_step == CONFIG["gradient_accumulation_steps"] - 1)
		_, loss = model(X, Y)
		# scale the loss to account for gradient accumulation
		loss = loss / CONFIG["gradient_accumulation_steps"]
		loss.backward() # backward pass
	torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

	if optimizer_hyperparams["use_muon"]:
		for group in optimizers[1].param_groups:
			frac = min(stats["step"] / 300, 1) # momentum warmup for muon
			group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

	## step the optimizers
	for o in optimizers:
		o.step()

	## flush the gradients as soon as we can, no need for this memory anymore
	optimizers[0].zero_grad(set_to_none=True)
	model.zero_grad(set_to_none=True)

	# validation section
	## save checkpoint
	if CONFIG["checkpoints"]["create_checkpoints"] and stats["step"] > 0 and stats["step"] % CONFIG["checkpoints"]["interval"] == 0 and master_process:
		print0(f"saved checkpoint at step {Fore.WHITE}{Style.BRIGHT}{stats["step"]}", log_path=log_path)
		torch.save(get_state(raw_model, optimizers), f"{CONFIG["checkpoints"]["path"]}/step{stats["step"]}.bin")
		with open(f"{CONFIG["checkpoints"]["path"]}/step{stats["step"]}.json", "w", encoding="utf-8") as f:
			json.dump(stats, f, indent=4)

	## log train-val loss
	if stats["step"] > 0 and stats["step"] % CONFIG["eval_interval"] == 0 and master_process:
		losses = estimate_loss(model, dataset.next_batch)
		eval_t1 = time.time()
		eval_dt = eval_t1 - eval_t0
		eval_t0 = eval_t1

		print0(
			f"{Fore.WHITE}{Style.BRIGHT}step",
			f"{Fore.WHITE}{Style.DIM}[{stats["step"]}/{CONFIG["max_iters"]}]"
			f"{Fore.RESET}{Style.RESET_ALL}:",
			f"train loss {Fore.WHITE}{Style.BRIGHT}{losses["train"]:.4f}"
			f"{Fore.RESET}{Style.RESET_ALL},",
			f"val loss {Fore.WHITE}{Style.BRIGHT}{losses["val"]:.4f}"
			f"{Fore.RESET}{Style.RESET_ALL},",
			f"lr {Fore.WHITE}{Style.BRIGHT}{lr:.7f}"
			f"{Fore.RESET}{Style.RESET_ALL},",
			f"time took {Fore.WHITE}{Style.DIM}{calc_total_time(eval_dt)}",
			log_path=log_path
		)
		stats["loss"]["train"].append(losses["train"].item())
		stats["loss"]["val"].append(losses["val"].item())

		### sample generation
		out = raw_model.generate([random.randint(0, len(enc.vocab) + len(enc.special_tokens))], hyperparams["block_size"], device=device_type)[0].tolist()
		print0(f"{Fore.WHITE}{Style.DIM}```\n{enc.decode(out)}\n```", log_path=log_path)

	## log test loss
	### get loss as float. note: this is a CPU-GPU sync point
	### scale up to undo the division above, approximating the true total loss (exact would have been a sum)
	lossf = loss.item() * CONFIG["gradient_accumulation_steps"]
	stats["loss"]["test"].append(lossf)

	if stats["step"] % CONFIG["log_interval"] == 0 and master_process:
		test_t1 = time.time()
		test_dt = test_t1 - test_t0
		test_t0 = test_t1

		toks_per_sec = (CONFIG["batch_size"] * CONFIG["gradient_accumulation_steps"] * hyperparams["block_size"] * CONFIG["log_interval"]) / test_dt
		print0(
			f"{Fore.WHITE}{Style.BRIGHT}iter",
			f"{Fore.WHITE}{Style.DIM}[{stats["step"]}/{CONFIG["max_iters"]}]"
			f"{Fore.RESET}{Style.RESET_ALL}:",
			f"loss {Fore.WHITE}{Style.BRIGHT}{lossf:.4f}"
			f"{Fore.RESET}{Style.RESET_ALL},",
			f"dt {Fore.WHITE}{Style.DIM}{calc_total_time(test_dt)}"
			f"{Fore.RESET}{Style.RESET_ALL},",
			f"tok/s {Fore.WHITE}{Style.DIM}{toks_per_sec:.2f}",
			log_path=log_path
		)
	stats["step"] += 1

print0("total time:", calc_total_time(time.time() - start_time), log_path=log_path)
torch.save(get_state(model, optimizers), f"{CONFIG["checkpoints"]["path"]}/final.bin")
with open(f"{CONFIG["checkpoints"]["path"]}/final.json", "w", encoding="utf-8") as f:
	json.dump(stats, f, indent=4)

if ddp:
	destroy_process_group()
