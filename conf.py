import json, os

lrs = [
	("7e-4", "7e-5"),
	("8e-4", "8e-5"),
	("9e-4", "9e-5"),
	("1e-3", "1e-4"),
	("2e-3", "2e-4"),
	("3e-3", "3e-4"),
	("4e-3", "4e-4"),
]

cl = [128, 256, 512, 1024]

def m(lr, idx, context_len):
	return {
		"dataset": {
			"data_division": 0.8,
			"load_from_file": True,
			"path": "data/synth-100M.bin"
		},
		"checkpoints": {
			"path": f"bin/20m/{context_len}b/s{idx}",
			"interval": 2000,
			"create_checkpoints": True
		},
		"model_hyperparams": {
			"vocab_size": 16384,
			"block_size": context_len,
			"n_layer": 6,
			"n_head": 8,
			"n_embd": 64
		},
		"optimizer_hyperparams": {
			"eps": 1e-10,
			"beta1": 0.9,
			"beta2": 0.95,
			"weight_decay": 1e-1,
			"use_muon": True,
			"momentum": 0.95
		},
		"encoder_path": "bin/cl16k.bin",
		"init_from": "scratch",
		"seed": 18,

		"gradient_accumulation_steps": 1,
		"batch_size": 8,

		"max_iters": 20000,
		"eval_interval": 2000,
		"log_interval": 200,
		"eval_iters": 200,

		"decay_lr": True,
		"lr_decay_iters": 20000,
		"learning_rate": lr[0],
		"cooldown_frac": 0.6,
		"warmup_iters": 1000,
		"min_lr": lr[1]
	}

for c in cl:
	for i, l in enumerate(lrs):
		C = m(l, i+1, c)

		path = C["checkpoints"]["path"].replace("bin", "config").replace("/", "\\") + ".json"
		os.makedirs(path[:-7], exist_ok=True)
		print(path)
		t = json.dumps(C, indent=4).replace(f'"learning_rate": "{l[0]}"', f'"learning_rate": {l[0]}').replace(f'"min_lr": "{l[1]}"', f'"min_lr": {l[1]}')

		with open(path, "w", encoding="utf-8") as f:
			f.write(t + "\n")
