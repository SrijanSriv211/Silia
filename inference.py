from model import Config, Silia
from encoder import Encoder

from colorama import Style, Fore, init
import argparse, random, torch

def generate(i, e, l=256, t=0.8, f=40, T=None, device="cpu"):
	# create an instance of Silia
	conf = Config(**i["hyperparams"])
	model = Silia(conf)

	# remove `_orig_mod.` prefix from state_dict (if it's there)
	state_dict = i["model"]
	unwanted_prefix = '_orig_mod.'

	for k, v in list(state_dict.items()):
		if k.startswith(unwanted_prefix):
			state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

	# load the saved model state_dict
	model.load_state_dict(state_dict)
	model.to(device)
	model.eval() # set the model to evaluation mode

	# compile the model
	torch.compile(model)

	# load the encoder
	enc = Encoder()
	enc.load(e)

	# encode text and generate output
	out = []
	enctxt = enc.encode(T, allowed_special="all") if T is not None else [random.randint(0, len(enc.vocab) + len(enc.special_tokens))]
	for tok in model.generate(enctxt, max_new_tokens=l, device=device, temperature=t, top_k=f):
		print(f"{Fore.WHITE}{Style.DIM}{enc.decode([tok])}", end="", flush=True)
		out.append(tok)
	print(enc.decode(out))
	return enc.decode(out)

if __name__ == "__main__":
	init(autoreset = True)
	parser = argparse.ArgumentParser(description="A powerful text encryption and decryption program.")
	parser.add_argument("--model", "-i", help="model path", required=True)
	parser.add_argument("--encoder", "-e", help="encoder path", required=True)
	parser.add_argument("--length", "-l", help="output length", type=int, default=1024)
	parser.add_argument("--temperature", "-t", help="output temperature", type=float, default=0.8)
	parser.add_argument("--top_k", "-f", help="output top_k", type=int, default=50)
	parser.add_argument("--text_prompt", "-T", help="Text input from the command line.", default=None)
	args = parser.parse_args()

	device = "cuda" if torch.cuda.is_available() else "cpu"
	if args.text_prompt:
		text = args.text_prompt
		print(f"{Fore.WHITE}{Style.BRIGHT}> {text}")
		generate(
            torch.load(args.model, map_location=device),
            args.encoder, args.length, args.temperature, args.top_k, text, device
        )

	else:
		while True:
			text = input("> ")

			if text == "/q":
				break

			elif text.strip() == "":
				text = None

			generate(
                torch.load(args.model, map_location=device),
                args.encoder, args.length, args.temperature, args.top_k, text, device
            )
