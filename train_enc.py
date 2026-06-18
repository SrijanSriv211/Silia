from encoder import Encoder
import pandas as pd, os

special_tokens = ["<|sink|>", "<|eop|>", "<|eot|>"]
dataset_path = "data/fineweb-edu-100M/train-00000-of-00002.parquet"
outpath = "bin/o12k.bin"
vocab_size = 12288 - len(special_tokens)

dir = os.path.split(outpath)[0]
if not os.path.isdir(dir):
	os.mkdir(dir)

enc = Encoder()
df = pd.read_parquet(dataset_path)
text = df["text"].tolist()
text = "\n".join(text) + "\n"

#* set `vocab_size` in `config.json` 4096
enc.train(text, vocab_size)
enc.register_special_tokens(*special_tokens)
enc.save(outpath)

print("Special Tokens:\n", enc.special_tokens)
