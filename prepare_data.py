from encoder import Encoder
from rich.progress import track
from colorama import Style, Fore, init
import pandas as pd, pickle, sys

init(autoreset=True)
dataset_path = sys.argv[1]
enc_path = sys.argv[2]
outpath = sys.argv[3]
# -1 becase we later add `<|end-text|>` special token
seq_len = None if sys.argv[4] == "None" else int(sys.argv[4]) - 1

enc = Encoder()
enc.load(enc_path)
data = []

df = pd.read_parquet(dataset_path)
data = df["text"].tolist()
del df

if seq_len is not None:
	data = [
		s[i:i+seq_len]
		for s in data
		for i in range(0, len(s), seq_len)
	]

lsum = lambda x: sum([len(i) for i in x])

n_chars = lsum(data)
for i, x in enumerate(track(data, f"{Fore.WHITE}{Style.BRIGHT}encoding {Fore.WHITE}{Style.DIM}dataset{Style.RESET_ALL}")):
	data[i] = enc.encode(f"{x}<|end-text|>", allowed_special="all")

n_toks = lsum(data)
print(f"{(n_chars/1e6)}M total chars,", f"{(n_toks/1e6)}M total tokens")

with open(outpath, "wb") as f:
	pickle.dump({"dataset": data}, f)
