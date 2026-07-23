import os, sys, json, math, urllib.request
import torch
from model import Silia, Config
from encoder import Encoder

try:
	from datasets import load_dataset
except ImportError:
	print("Missing 'datasets' library. Install with: uv pip install datasets")
	sys.exit(1)


MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "bin/c1/final/model.bin"
TOKENIZER_PATH = sys.argv[2] if len(sys.argv) > 2 else "data/o512.bin"
CONFIG_PATH = sys.argv[3] if len(sys.argv) > 3 else "data/s2.json"
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
	print(f"Device: {DEVICE}")
	print(f"Model:  {MODEL_PATH}")
	print(f"Tok:	{TOKENIZER_PATH}")
	print(f"Config: {CONFIG_PATH}")
	if LIMIT:
		print(f"Limit:  {LIMIT} per benchmark")
	print()

	enc = Encoder()
	enc.load(TOKENIZER_PATH)

	# train.py always prepends this sink token at position 0 of every
	# training sequence (dataloader.next_batch: x = cat([sink_col, y[:,:-1]])),
	# and model.generate() does the same. Verify this key exists in your
	# tokenizer's special_tokens dict - this will print them and raise if not.
	if "<|actor|>" not in enc.special_tokens:
		print("Available special tokens:", enc.special_tokens)
		raise KeyError("'<|actor|>' not found in tokenizer special_tokens - update the key above")
	SINK_TOK = enc.special_tokens["<|actor|>"]

	with open(CONFIG_PATH) as f:
		train_config = json.load(f)
	hp = train_config["model_hyperparams"]
	conf = Config(**hp)

	model = Silia(conf)
	ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
	model.load_state_dict(ckpt["model"])
	model.to(DEVICE)
	model.eval()
	print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
	print()

	@torch.no_grad()
	def score_ending(ctx_text, ending_text):
		# Encode the joined text once so the boundary space between ctx and
		# ending is tokenized exactly as it appears in real running text,
		# instead of encoding the two pieces separately and losing the
		# joining space (this tokenizer encodes each regex chunk
		# independently, so slicing by len(ctx_ids) is safe here).
		ctx_ids = enc.encode(ctx_text, allowed_special="none")
		full_ids = enc.encode(ctx_text + " " + ending_text, allowed_special="none")
		ending_ids = full_ids[len(ctx_ids):]
		tokens = ctx_ids + ending_ids

		if len(tokens) > conf.block_size:
			tokens = tokens[-(conf.block_size):]
			ctx_len = max(0, conf.block_size - len(ending_ids))
		else:
			ctx_len = len(ctx_ids)

		# Match train.py's exact input convention: x = [SINK] + tokens[:-1],
		# y = tokens (unshifted). x[i] always predicts y[i].
		x = torch.tensor([SINK_TOK] + tokens[:-1], dtype=torch.long, device=DEVICE).unsqueeze(0)
		y = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)
		y[:, :ctx_len] = -1  # ignore context positions, score only the ending

		_, loss = model(x, y)
		return loss.item(), len(ending_ids)

	# ----- HellaSwag -----
	print("--- HellaSwag ---")
	try:
		ds = load_dataset("Rowan/hellaswag", split="validation")
	except Exception as e:
		print(f"  FAIL: {e}")
	else:
		if LIMIT:
			ds = ds.select(range(LIMIT))
		n = len(ds)
		acc, acc_norm, n_acc, n_acc_norm = 0, 0, 0, 0
		for i, ex in enumerate(ds):
			ctx = ex["ctx"]
			label = int(ex["label"])
			endings = ex["endings"]
			losses, tot_losses = [], []
			for e in endings:
				pl, ntok = score_ending(ctx, e)
				losses.append(pl)
				tot_losses.append(pl * ntok)
			pred = tot_losses.index(min(tot_losses))
			pred_n = losses.index(min(losses))
			if pred == label:
				acc += 1
			if pred_n == label:
				acc_norm += 1
			n_acc += 1
			n_acc_norm += 1
			if (i + 1) % 500 == 0:
				print(f"  [{i+1}/{n}] acc={acc/n_acc:.4f}  acc_norm={acc_norm/n_acc_norm:.4f}")
		print(f"  acc={acc/n_acc:.4f} ({acc}/{n_acc})")
		print(f"  acc_norm={acc_norm/n_acc_norm:.4f} ({acc_norm}/{n_acc_norm})")
		print()

	# ----- PIQA -----
	print("--- PIQA ---")
	try:
		url = "https://yonatanbisk.com/piqa/data/valid.jsonl"
		url_labels = "https://yonatanbisk.com/piqa/data/valid-labels.lst"
		data_json = urllib.request.urlopen(url).read().decode()
		data_labels = urllib.request.urlopen(url_labels).read().strip().split()
		lines = data_json.strip().split("\n")
		if LIMIT:
			lines = lines[:LIMIT]
			data_labels = data_labels[:LIMIT]
		n = len(lines)
		acc = 0
		for i, (line, label_str) in enumerate(zip(lines, data_labels)):
			ex = json.loads(line)
			label = int(label_str)
			l1, _ = score_ending(ex["goal"], ex["sol1"])
			l2, _ = score_ending(ex["goal"], ex["sol2"])
			pred = 0 if l1 < l2 else 1
			if pred == label:
				acc += 1
			if (i + 1) % 500 == 0:
				print(f"  [{i+1}/{n}] acc={acc/(i+1):.4f}")
		print(f"  acc={acc/n:.4f} ({acc}/{n})")
		print()
	except Exception as e:
		print(f"  FAIL: {e}")

	# ----- LAMBADA -----
	print("--- LAMBADA ---")
	try:
		ds = load_dataset("EleutherAI/lambada_openai", split="test")
	except Exception as e:
		print(f"  FAIL: {e}")
	else:
		if LIMIT:
			ds = ds.select(range(LIMIT))
		n = len(ds)
		total_nll = 0.0	 # sum of per-WORD total NLL (each word weighted equally,
							 # matching the standard LAMBADA convention - NOT
							 # divided by that word's own token count)
		n_examples = 0
		word_acc = 0
		for i, ex in enumerate(ds):
			text = ex["text"]
			text = text.strip()
			idx = text.rfind(" ")
			if idx == -1:
				continue
			prefix = text[:idx]
			word = text[idx + 1:]

			# Encode the full text once so the boundary space is tokenized
			# correctly (see score_ending's comment above for why).
			prefix_ids = enc.encode(prefix, allowed_special="none")
			full_ids = enc.encode(text, allowed_special="none")
			word_ids = full_ids[len(prefix_ids):]
			tokens = full_ids

			if len(tokens) > conf.block_size:
				tokens = tokens[-(conf.block_size):]
				plen = max(0, conf.block_size - len(word_ids))
			else:
				plen = len(prefix_ids)

			# Match train.py's exact input convention: x = [SINK] + tokens[:-1],
			# y = tokens (unshifted). x[i] always predicts y[i].
			x = torch.tensor([SINK_TOK] + tokens[:-1], dtype=torch.long, device=DEVICE).unsqueeze(0)
			y = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)
			y[:, :plen] = -1  # ignore prefix positions, score only the word

			with torch.no_grad():
				logits, loss = model(x, y)

			# loss.item() is the MEAN nll over this word's tokens (cross_entropy
			# averages over unmasked/ignore_index=-1-excluded positions).
			# Multiplying by len(word_ids) recovers the word's TOTAL nll, which
			# is what gets averaged per-example below (not per-token) - this is
			# the fix. The old version divided the running total_loss by a
			# running total TOKEN count, which under-weights multi-token words
			# relative to the standard per-word LAMBADA perplexity convention.
			word_nll = loss.item() * len(word_ids)
			total_nll += word_nll
			n_examples += 1

			logits = logits[0]
			wstart, wend = plen, len(tokens)
			ok = True
			for j in range(wstart, wend):
				if logits[j].argmax().item() != y[0, j].item():
					ok = False
					break
			if ok:
				word_acc += 1

			if (i + 1) % 500 == 0:
				ppl = math.exp(total_nll / n_examples) if n_examples else float("inf")
				print(f"  [{i+1}/{n}] ppl={ppl:.4f}  word_acc={word_acc/n_examples:.4f}")
		ppl = math.exp(total_nll / n_examples) if n_examples else float("inf")
		print(f"  ppl={ppl:.4f}")
		print(f"  word_acc={word_acc/n_examples:.4f} ({word_acc}/{n_examples})")
		print()


if __name__ == "__main__":
	main()
