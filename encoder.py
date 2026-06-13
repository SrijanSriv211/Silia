from colorama import init, Fore, Style
import pickle, regex, time, numpy as np
from collections import defaultdict

init(autoreset=True)

GPT4_SPLIT_PATTERN = "|".join(
	[
		r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
		r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
		r"""\p{N}{1,3}""",
		r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
		r"""\s*[\r\n]+""",
		r"""\s+(?!\S)""",
		r"""\s+""",
	]
)

def calc_total_time(seconds):
	sec_int, millis = divmod(seconds, 1)
	millis = int(millis * 1000)
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


def get_stats_batched(ids_list, weights):
	"""
	Batch stats across all chunks at once using a flat pass.
	Much faster than looping get_stats per chunk because we avoid
	repeated dict lookups in a Python for-loop over chunks.
	"""
	counts = defaultdict(int)
	for chunk, w in zip(ids_list, weights):
		for pair in zip(chunk, chunk[1:]):
			counts[pair] += w
	return counts


def merge(ids, pair, idx):
	newids = []
	i = 0
	p0, p1 = pair
	while i < len(ids):
		if i < len(ids) - 1 and ids[i] == p0 and ids[i+1] == p1:
			newids.append(idx)
			i += 2
		else:
			newids.append(ids[i])
			i += 1
	return newids


def merge_batch(ids_list, pair, idx):
	"""Merge pair in all chunks in one tight Python loop — no subprocess overhead."""
	p0, p1 = pair
	result = []
	for ids in ids_list:
		newids = []
		i = 0
		while i < len(ids):
			if i < len(ids) - 1 and ids[i] == p0 and ids[i+1] == p1:
				newids.append(idx)
				i += 2
			else:
				newids.append(ids[i])
				i += 1
		result.append(newids)
	return result


class Encoder:
	def __init__(self, pattern=None):
		self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
		self.compiled_pattern = regex.compile(self.pattern)
		self.special_tokens = {}
		self.inverse_special_tokens = {}
		self.vocab = {idx: bytes([idx]) for idx in range(256)}

	def train(self, text, vocab_size=256, text_range=None):
		assert vocab_size >= 256
		start_time = time.time()

		print(
			"encoding text with", f"{Fore.WHITE}{Style.BRIGHT}{len(text)/1e6:.2f}M", "total characters and",
			f"{Fore.WHITE}{Style.BRIGHT}{len(set(text))}", "unique characters"
		)

		if text_range is not None:
			text = text[:text_range]
			print(
				"ranged text has", f"{Fore.WHITE}{Style.BRIGHT}{len(text)/1e6:.2f}M", "characters and",
				f"{Fore.WHITE}{Style.BRIGHT}{len(set(text))}", "unique characters"
			)

		text_chunks = regex.findall(self.compiled_pattern, text)
		del text

		print(f"encoding text chunks... {Fore.WHITE}{Style.DIM}(takes a ~minute)")

		ids = [list(ch.encode("utf-8")) for ch in text_chunks]
		del text_chunks

		# Deduplicate chunks, track weights
		tmp = {}
		for byte_str in ids:
			b = bytes(byte_str)
			tmp[b] = tmp.get(b, 0) + 1

		ids = [list(k) for k in tmp.keys()]
		idsw = list(tmp.values())

		print("training on vocab size", f"{Fore.WHITE}{Style.BRIGHT}{vocab_size}")

		n_merges = vocab_size - 256
		last_print_time = time.time()

		for i in range(n_merges):
			# Batched stats: one pass over all chunks
			stats = get_stats_batched(ids, idsw)

			pair = max(stats, key=stats.get)
			idx = 256 + i
			self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

			# Batched merge: one tight loop over all chunks, no subprocess
			ids = merge_batch(ids, pair, idx)

			current_print_time = time.time()
			print(
				f"{Fore.WHITE}{Style.BRIGHT}merge",
				f"{Fore.WHITE}{Style.DIM}[{i+1}/{n_merges}]:",
				f"{pair} -> {idx}",
				f"{Fore.WHITE}{Style.DIM}({self.vocab[idx]})",
				f"had {Fore.WHITE}{Style.BRIGHT}{stats[pair]}{Style.RESET_ALL} occurrences,",
				f"{Fore.WHITE}{Style.DIM}time: {calc_total_time(current_print_time - last_print_time)}"
			)
			last_print_time = current_print_time

		print("vocab size:", f"{Fore.WHITE}{Style.BRIGHT}{len(self.vocab)}")
		print("time taken:", f"{Fore.WHITE}{Style.BRIGHT}{calc_total_time(time.time() - start_time)}")

	def register_special_tokens(self, *special_tokens):
		self.special_tokens = dict([(x, i + len(self.vocab)) for i, x in enumerate(special_tokens)])
		self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

	def decode(self, ids):
		part_bytes = []
		for idx in ids:
			if idx in self.vocab:
				part_bytes.append(self.vocab[idx])
			elif idx in self.inverse_special_tokens:
				part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
			else:
				raise ValueError(f"invalid token id: {idx}")
		return b"".join(part_bytes).decode("utf-8", errors="replace")

	def _encode_chunk(self, text_bytes, inverse_vocab):
		ids = list(text_bytes)
		if len(ids) < 2:
			return ids
		changed = True
		while changed:
			changed = False
			newids = []
			i = 0
			while i < len(ids):
				if i < len(ids) - 1:
					merged_bytes = self.vocab.get(ids[i], b'') + self.vocab.get(ids[i+1], b'')
					merged_id = inverse_vocab.get(merged_bytes)
					if merged_id is not None:
						newids.append(merged_id)
						i += 2
						changed = True
						continue
				newids.append(ids[i])
				i += 1
			ids = newids
		return ids

	def encode_ordinary(self, text, inverse_vocab=None):
		if inverse_vocab is None:
			inverse_vocab = {v: k for k, v in self.vocab.items()}
		text_chunks = regex.findall(self.compiled_pattern, text)
		ids = []
		for chunk in text_chunks:
			ids.extend(self._encode_chunk(chunk.encode("utf-8"), inverse_vocab))
		return ids

	def encode(self, str, isfile=False, allowed_special="none_raise"):
		text = str
		if isfile:
			with open(str, "r", encoding="utf-8") as f:
				text = f.read()

		special = None
		if allowed_special == "all":
			special = self.special_tokens
		elif allowed_special == "none":
			special = {}
		elif allowed_special == "none_raise":
			special = {}
			assert all(token not in text for token in self.special_tokens)
		elif isinstance(allowed_special, set):
			special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
		else:
			raise ValueError(f"allowed_special={allowed_special} not understood")

		inverse_vocab = {v: k for k, v in self.vocab.items()}

		if not special:
			return self.encode_ordinary(text, inverse_vocab)

		special_pattern = "(" + "|".join(regex.escape(k) for k in special) + ")"
		special_chunks = regex.split(special_pattern, text)
		del text

		ids = []
		for part in special_chunks:
			if part in special:
				ids.append(special[part])
			else:
				ids.extend(self.encode_ordinary(part, inverse_vocab))
		return ids

	def save(self, checkpoint):
		with open(checkpoint, "wb") as f:
			pickle.dump({
				"pattern": self.pattern,
				"special": self.special_tokens,
				"vocab": self.vocab
			}, f)

	def load(self, checkpoint: str):
		with open(checkpoint, "rb") as f:
			model = pickle.load(f)
		self.pattern = model["pattern"]
		self.special_tokens = model["special"]
		self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
		self.vocab = model["vocab"]
