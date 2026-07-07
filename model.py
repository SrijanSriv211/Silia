from torch.nn import functional as F
from dataclasses import dataclass
import torch.nn as nn, torch

@dataclass
class Config:
	vocab_size: int = 8192
	block_size: int = 1024
	n_layer: int = 2
	n_head: int = 4
	n_embd: int = 64

def norm(x):
	return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
	assert x.ndim == 4  # multihead attention
	d = x.shape[3] // 2
	x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
	y1 = x1 * cos + x2 * sin # rotate pairs of dims
	y2 = x1 * (-sin) + x2 * cos
	return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
	def __init__(self, config: Config, chunk=1):
		super().__init__()
		self.n_head = config.n_head
		self.n_embd = config.n_embd
		n_qkv = config.n_embd * self.n_head

		self.qkv = nn.Linear(config.n_embd, 3*n_qkv, bias=False)
		self.out = nn.Linear(n_qkv, config.n_embd*chunk, bias=False)

		self.reset_cache()

	def reset_cache(self):
		self.k_cache = None
		self.v_cache = None
		self.cache_pos = 0

	def setup_cache(self, batch_size, max_seq_len, device, dtype):
		self.k_cache = torch.empty(batch_size, self.n_head, max_seq_len, self.n_embd, device=device, dtype=dtype)
		self.v_cache = torch.empty_like(self.k_cache)
		self.cache_pos = 0

	def forward(self, x, cos_sin, use_cache=False):
		B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

		# calculate query, key, values for all heads in batch and move head forward to be the batch dim
		q, k, v  = self.qkv(x).view(B, T, self.n_head, -1).chunk(3, dim=-1)

		# apply rotary embeddings to queries and keys to get relative positional encoding
		cos, sin = cos_sin
		q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
		q, k = norm(q), norm(k) # QK norm

		# make head be batch dim, i.e. (B, T, nh, hs) -> (B, nh, T, hs)
		q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

		if use_cache:
			pos = self.cache_pos
			self.k_cache[:, :, pos:pos+T] = k
			self.v_cache[:, :, pos:pos+T] = v

			self.cache_pos += T
			k = self.k_cache[:, :, :self.cache_pos]
			v = self.v_cache[:, :, :self.cache_pos]

		# causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
		y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=(use_cache == False))
		y = y.transpose(1, 2).contiguous().view(B, T, -1) # re-assemble all head outputs side by side

		# output projection
		return self.out(y)

class Block(nn.Module):
	def __init__(self, config: Config):
		super().__init__()
		self.attn1 = CausalSelfAttention(config, 2)
		self.attn2 = CausalSelfAttention(config)

	def forward(self, x, cos_sin, use_cache=False):
		u, v = self.attn1(norm(x), cos_sin, use_cache).chunk(2, dim=-1)
		y = u * F.silu(v)
		return x + self.attn2(y, cos_sin, use_cache)

class Silia(nn.Module):
	def __init__(self, config: Config):
		super().__init__()
		assert config.vocab_size is not None
		assert config.block_size is not None
		self.config = config

		# factorized token embeddings
		self.embed = nn.Embedding(config.vocab_size, config.n_embd)
		self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
		self.unembed = nn.Linear(config.n_embd, config.vocab_size, bias=False)
		self.embed.weight = self.unembed.weight

		# to support meta device initialization, we init the rotary embeddings here, but it's fake
		# as for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
		# so let's just over-compute them, but assert fail if we ever reach that amount.
		# in the future we can dynamically grow the cache, for now it's fine.
		self.rotary_block_size = config.block_size * 10 # 10X over-compute should be enough, TODO make nicer?
		cos, sin = self._precompute_rotary_embeddings(self.rotary_block_size, config.n_embd)
		self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
		self.register_buffer("sin", sin, persistent=False)

	def _precompute_rotary_embeddings(self, block_size, d_head, base=10000):
		# stride the channels
		channel_range = torch.arange(0, d_head, 2)
		inv_freq = 1.0 / (base ** (channel_range / d_head))
		# stride the time steps
		t = torch.arange(block_size)
		# calculate the rotation frequencies at each (time, channel) pair
		freqs = torch.outer(t, inv_freq)
		cos, sin = freqs.cos(), freqs.sin()
		return cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting

	def setup_cache(self, batch_size, device, dtype):
		for block in self.blocks:
			block.attn1.setup_cache(batch_size, self.rotary_block_size, device, dtype)
			block.attn2.setup_cache(batch_size, self.rotary_block_size, device, dtype)

	def reset_cache(self):
		for block in self.blocks:
			block.attn1.reset_cache()
			block.attn2.reset_cache()

	def forward(self, idx, targets=None, use_cache=False):
		B, T = idx.size()

		# grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim))
		assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
		if use_cache:
			pos = self.blocks[0].attn1.cache_pos
			cos_sin = self.cos[:, pos:pos+T], self.sin[:, pos:pos+T]

		else:
			cos_sin = self.cos[:, :T], self.sin[:, :T]

		# token embeddings of shape (b, t, n_embd)
		x = self.embed(idx)
		x = norm(x)

		for block in self.blocks:
			x = block(x, cos_sin, use_cache)

		# forward the lm_head (compute logits)
		x = norm(x)
		logits = self.unembed(x)

		# if we are given some desired targets also calculate the loss
		loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction="mean")
		return logits, loss

	@torch.no_grad()
	def generate(self, idx, max_new_tokens, device, temperature=1.0, top_k=50):
		self.reset_cache()
		self.setup_cache(batch_size=1, device=device, dtype=self.embed.weight.dtype)

		# forward the model to get the logits for the index in the sequence
		idx = torch.tensor(idx, dtype=torch.int64, device=device).unsqueeze(0)
		logits, _ = self(idx, use_cache=True)

		for _ in range(max_new_tokens):
			logits = logits[:, -1, :]

			# pluck the logits at the final step and scale by desired temperature
			if temperature > 0:
				logits = logits / temperature

				# optionally crop the logits to only the top k options
				if top_k is not None:
					v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
					logits[logits < v[:, [-1]]] = -float("Inf")

				# apply softmax to convert logits to (normalized) probabilities,
				# sample from the distribution and,
				probs = F.softmax(logits, dim=-1)
				idx_next = torch.multinomial(probs, num_samples=1)

			else:
				idx_next = torch.argmax(logits, dim=-1, keepdim=True)

			idx = torch.cat([idx, idx_next], dim=1)
			logits, _ = self(idx_next, use_cache=True)

			# stream tokens
			yield idx_next.item()
