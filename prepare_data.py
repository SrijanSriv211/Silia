from encoder import Encoder
from rich.progress import track
from colorama import Style, Fore, init
import pandas as pd, pickle, json, os

init(autoreset=True)

dataset_path = [
	"data/fineweb-edu-100M/train-00000-of-00002.parquet",
	"data/fineweb-edu-100M/train-00001-of-00002.parquet"
]
enc_path = "bin/o12k.bin"

enc = Encoder()
enc.load(enc_path)
data = []

df = pd.read_parquet(dataset_path)
data = df["text"].tolist()

lsum = lambda x: sum([len(i) for i in x])

n_chars = lsum(data)
for i, x in enumerate(track(data, f"{Fore.WHITE}{Style.BRIGHT}encoding {Fore.WHITE}{Style.DIM}fineweb-100M{Style.RESET_ALL}")):
	data[i] = enc.encode(f"{x}<|eot|>\n", allowed_special="all")

n_toks = lsum(data)
print(f"{(n_chars/1e6)}M total chars,", f"{(n_toks/1e6)}M total tokens")

with open("data/fineweb-100M.bin", "wb") as f:
	pickle.dump({"dataset": data}, f)
