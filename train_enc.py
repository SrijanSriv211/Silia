from encoder import Encoder
import pandas as pd, json, sys, os

special_tokens = ["<|end-text|>", "<|actor|>"]
vocab_size = int(sys.argv[1]) - len(special_tokens)
chunk_range = None if sys.argv[2] == "None" else int(sys.argv[2])
dataset_path = sys.argv[3]
outpath = sys.argv[4]

dir = os.path.split(outpath)[0]
os.makedirs(dir, exist_ok=True)

if dataset_path.endswith(".txt"):
	with open(dataset_path, "r", encoding="utf-8") as f:
		text = f.read() + "\n"

elif dataset_path.endswith(".json"):
	with open(dataset_path, "r", encoding="utf-8") as f:
		o = json.load(f)
		text = "\n".join(o) + "\n"
		del o

elif dataset_path.endswith(".parquet"):
	df = pd.read_parquet(dataset_path)
	text = df["text"].tolist()
	del df
	text = "\n".join(text) + "\n"

else:
	raise Exception("Dataset extensions must be `.txt`, `.json` or `.parquet`")

#* set `vocab_size` in `config.json`
enc = Encoder()
enc.train(text, vocab_size, chunk_range)
enc.register_special_tokens(*special_tokens)
enc.save(outpath)
print("Special Tokens:\n", enc.special_tokens)
