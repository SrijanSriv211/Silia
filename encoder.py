from multiprocessing import Pool, cpu_count
from colorama import init, Fore, Style
import pickle, regex, json, time, os

init(autoreset=True)

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
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
	# separate the integer part (for hours, minutes, and seconds) from the fractional part (for milliseconds)
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

# https://github.com/karpathy/minbpe/pull/82/files#diff-2f6d110dc37c6714f3b44335b029a950adfb0c58e2c3013e030a9bbdd76ed02d
def get_stats(ids, counts=None, weight=1):
	"""
	Given a list of integers, return a dictionary of counts of consecutive pairs, multiplied by weight
	Example: [1, 2, 3, 1, 2] -> {(1, 2): 2*weight, (2, 3): 1*weight, (3, 1): 1*weight}
	Optionally allows to update an existing dictionary of counts
	"""
	counts = {} if counts is None else counts
	for pair in zip(ids, ids[1:]): # iterate consecutive elements
		counts[pair] = counts.get(pair, 0) + weight
	return counts

def merge(ids, pair, idx):
	"""
	In the list of integers (ids), replace all consecutive occurrences
	of pair with the new integer token idx
	Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
	"""
	newids = []
	i = 0
	while i < len(ids):
		# if not at the very last position AND the pair matches, replace it
		if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
			newids.append(idx)
			i += 2

		else:
			newids.append(ids[i])
			i += 1

	return newids

# optimized: merge a batch of (chunk, weight) pairs given a single (pair -> idx)
def _merge_batch(args):
	"""Merge pair->idx in a list of chunks. Runs in a worker process."""
	chunks, weights, pair, idx = args
	new_chunks = [merge(c, pair, idx) for c in chunks]
	return new_chunks

# optimized encode chunk: O(n) instead of O(n^2) using a single left-to-right pass
def _encode_chunk_fast(text_bytes, inverse_vocab):
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
				key = bytes([ids[i]]) + bytes([ids[i+1]]) if ids[i] < 256 and ids[i+1] < 256 else None
				# use prebuilt inverse_vocab which maps bytes -> token id
				merged = inverse_vocab.get(key) if key else None
				if merged is None:
					# try looking up via vocab bytes concatenation (handles merged tokens)
					pass
				if merged is not None:
					newids.append(merged)
					i += 2
					changed = True
					continue
			newids.append(ids[i])
			i += 1
		ids = newids
	return ids

# worker for parallel encode_ordinary (must be top-level for pickling)
_GLOBAL_ENCODER_STATE = {}

def _worker_encode_chunk(chunk_bytes):
	inverse_vocab = _GLOBAL_ENCODER_STATE['inverse_vocab']
	vocab = _GLOBAL_ENCODER_STATE['vocab']
	ids = list(chunk_bytes)
	if len(ids) < 2:
		return ids
	changed = True
	while changed:
		changed = False
		newids = []
		i = 0
		while i < len(ids):
			if i < len(ids) - 1:
				merged_bytes = vocab.get(ids[i], b'') + vocab.get(ids[i+1], b'')
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

def _init_worker(state):
	_GLOBAL_ENCODER_STATE.update(state)

class Encoder:
	def __init__(self, pattern=None):
		"""
		- pattern: optional string to override the default (GPT-4 split pattern)
		- special_tokens: str -> int dictionary of special tokens
			example: {'<|endoftext|>': 100257}
		"""
		self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
		self.compiled_pattern = regex.compile(self.pattern)
		self.special_tokens = {}
		self.inverse_special_tokens = {}
		self.vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes

	def train(self, text, vocab_size=256, text_range=None, n_workers=None):
		"""
		- path: [name, is_dir]
		- vocab_size: max number of merges to be made - 256 bytes
		- text_range: how much many chars from the text should be used to train (default: None means entire text)
		"""
		assert vocab_size >= 256
		start_time = time.time()
		if n_workers is None:
			n_workers = cpu_count()

		print(
			"encoding text with", f"{Fore.WHITE}{Style.BRIGHT}{len(text)/1e6}M", "total characters and",
			f"{Fore.WHITE}{Style.BRIGHT}{len(set(text))}", "unique characters"
		)

		if text_range is not None:
			print(
				"ranged text has", f"{Fore.WHITE}{Style.BRIGHT}{len(text[:text_range])/1e6}M", "characters and",
				f"{Fore.WHITE}{Style.BRIGHT}{len(set(text[:text_range]))}", "unique characters"
			)

		# split the text up into text chunks
		text_chunks = regex.findall(self.compiled_pattern, text if text_range is None else text[:text_range])
		del text

		print(f"encoding text chunks... {Fore.WHITE}{Style.DIM}(takes a ~minute)")

		# input text preprocessing
		ids = [list(ch.encode("utf-8")) for ch in text_chunks]
		del text_chunks

		# keep just one instance of identical chunks, keep their count in idsw
		# https://github.com/karpathy/minbpe/pull/82/files#diff-6b5737d60acbc8d11dba46334d76c559796c1aca8d51e13ed069236f947b9e1f
		tmp = {}
		for byte_str in ids:
			byte_str = bytes(byte_str)
			tmp[byte_str] = tmp.get(byte_str, 0) + 1

		ids = [list(k) for k in tmp.keys()]
		idsw = list(tmp.values())

		print("training on vocab size", f"{Fore.WHITE}{Style.BRIGHT}{vocab_size}")
		print(f"using {Fore.WHITE}{Style.BRIGHT}{n_workers}{Style.RESET_ALL} workers")

		n_merges = vocab_size - 256
		last_print_time = time.time()

		# split chunks into batches for parallel merge
		batch_size = max(1, len(ids) // n_workers)

		# iteratively merge the most common pairs to create new tokens
		for i in range(n_merges):
			# count the number of times every consecutive pair appears
			stats = {}

			# passing in stats will update it in place, adding up counts
			for j, chunk_ids in enumerate(ids):
				get_stats(chunk_ids, stats, idsw[j])

			# find the pair with the highest count
			pair = max(stats, key=stats.get)

			# mint a new token: assign it the next available id
			idx = 256 + i
			self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

			# replace all occurrences of pair in ids with idx
			# parallel merge
			if n_workers > 1 and len(ids) > n_workers:
				batches = []
				wbatches = []

				for b in range(n_workers):
					start = b * batch_size
					end = start + batch_size if b < n_workers - 1 else len(ids)
					batches.append(ids[start:end])
					wbatches.append(idsw[start:end])

				args = [(batches[b], wbatches[b], pair, idx) for b in range(n_workers)]

				with Pool(n_workers) as pool:
					results = pool.map(_merge_batch, args)

				ids = []

				for r in results:
					ids.extend(r)
			else:
				ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

			# verbose
			current_print_time = time.time()
			print(
				f"{Fore.WHITE}{Style.BRIGHT}merge",
				f"{Fore.WHITE}{Style.DIM}[{i+1}/{n_merges}]"
				":",
				f"{pair} -> {idx}",
				f"{Fore.WHITE}{Style.DIM}({self.vocab[idx]})",
				f"had {Fore.WHITE}{Style.BRIGHT}{stats[pair]}{Style.RESET_ALL} occurrences"
				f"{Style.RESET_ALL},",
				f"{Fore.WHITE}{Style.DIM}time taken: {calc_total_time(current_print_time-last_print_time)}"
			)
			last_print_time = current_print_time

		# print the total time taken to do all the merges
		print("vocab size:", f"{Fore.WHITE}{Style.BRIGHT}{len(self.vocab)}")
		print("time taken:", f"{Fore.WHITE}{Style.BRIGHT}{calc_total_time(time.time()-start_time)}")

	# special_tokens is a dictionary of str -> int
	# example: {"<|endoftext|>": 100257}
	def register_special_tokens(self, *special_tokens):
		self.special_tokens = dict([(x, i + len(self.vocab)) for i, x in enumerate(special_tokens)])
		self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

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

	def _build_inverse_vocab(self):
		return {v: k for k, v in self.vocab.items()}

	def _encode_chunk(self, text_bytes, inverse_vocab):
		"""O(n) left-to-right greedy merge pass, repeated until stable."""
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

	def encode_ordinary(self, text, inverse_vocab=None, n_workers=1):
		"""Encoding that ignores any special tokens. Optionally parallel."""
		if inverse_vocab is None:
			inverse_vocab = self._build_inverse_vocab()

		text_chunks = regex.findall(self.compiled_pattern, text)
		chunk_bytes_list = [chunk.encode("utf-8") for chunk in text_chunks]

		if n_workers > 1 and len(chunk_bytes_list) > n_workers:
			state = {'inverse_vocab': inverse_vocab, 'vocab': self.vocab}
			with Pool(n_workers, initializer=_init_worker, initargs=(state,)) as pool:
				results = pool.map(_worker_encode_chunk, chunk_bytes_list)

			ids = []
			for r in results:
				ids.extend(r)

		else:
			ids = []
			for chunk_bytes in chunk_bytes_list:
				ids.extend(self._encode_chunk(chunk_bytes, inverse_vocab))

		return ids

	def encode(self, str, isfile=False, allowed_special="none_raise", n_workers=1):
		"""
		Unlike encode_ordinary, this function handles special tokens.
		allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
		n_workers: number of parallel workers for encoding chunks
		"""
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

		inverse_vocab = self._build_inverse_vocab()

		if not special:
			return self.encode_ordinary(text, inverse_vocab, n_workers=n_workers)

		special_pattern = "(" + "|".join(regex.escape(k) for k in special) + ")"
		special_chunks = regex.split(special_pattern, text)
		del text

		ids = []
		for part in special_chunks:
			if part in special:
				ids.append(special[part])

			else:
				ids.extend(self.encode_ordinary(part, inverse_vocab, n_workers=n_workers))

		return ids

	def save(self, checkpoint):
		"""
		Saves two files: checkpoint.bin
		- model file is the critical one, intended for load()
		"""
		# write the model: to be used in load() later
		with open(checkpoint, "wb") as f:
			pickle.dump({
				"pattern": self.pattern,
				"special": self.special_tokens,
				"vocab": self.vocab
			}, f)

	def load(self, checkpoint: str):
		# read the model file
		with open(checkpoint, "rb") as f:
			model = pickle.load(f)

		self.pattern = model["pattern"]
		self.special_tokens = model["special"]
		self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
		self.vocab = model["vocab"]
