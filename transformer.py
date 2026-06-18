from torch.nn import functional as F
from dataclasses import dataclass
import torch.nn as nn, torch

@dataclass
class GPTConfig:
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
	def __init__(self, config: Config):
		super().__init__()
		self.n_head = config.n_head
		self.n_embd = config.n_embd
		d_model = self.n_embd * self.n_head

		self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
		self.out = nn.Linear(d_model, d_model, bias=False)

	def forward(self, x, cos_sin):
		B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

		# calculate query, key, values for all heads in batch and move head forward to be the batch dim
		q, k, v = self.qkv(norm(x)).view(B, T, self.n_head, self.n_embd).chunk(3, dim=-1)

		# apply rotary embeddings to queries and keys to get relative positional encoding
		cos, sin = cos_sin
		q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
		q, k = norm(q), norm(k) # QK norm

		# make head be batch dim, i.e. (B, T, nh, hs) -> (B, nh, T, hs)
		q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

		# causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
		y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
		y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

		# output projection
		return self.out(y)

class SwiGLU(nn.Module):
	def __init__(self, config: Config):
		super().__init__()
		d_model = self.n_embd * self.n_head

		self.uv = nn.Linear(d_model, 4*d_model, bias=False)
		self.out = nn.Linear(4*d_model, d_model, bias=False)

	def forward(self, x, cos_sin):
		u, v = self.uv(norm(x)).chunk(2, dim=-1)
		y = u * F.silu(v)
		return self.out(y)

class Block(nn.Module):
	def __init__(self, config: Config):
		super().__init__()
		self.attn = CausalSelfAttention(config)
		self.swiglu = SwiGLU(config)

	def forward(self, x, cos_sin):
		y = x + self.attn(x, cos_sin)
		return y + self.swiglu(y)

class GPT(nn.Module):
	def __init__(self, config: Config):
		super().__init__()
		assert config.vocab_size is not None
		assert config.block_size is not None
		self.config = config
		d_model = self.n_embd * self.n_head

		# factorized token embeddings
		self.embed = nn.Embedding(config.vocab_size, d_model)
		self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
		self.unembed = nn.Linear(d_model, config.vocab_size, bias=False)
		self.embed.weight = self.unembed.weight

		# to support meta device initialization, we init the rotary embeddings here, but it's fake
		# as for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
		# so let's just over-compute them, but assert fail if we ever reach that amount.
		# in the future we can dynamically grow the cache, for now it's fine.
		self.rotary_block_size = config.block_size * 10 # 10X over-compute should be enough, TODO make nicer?
		cos, sin = self._precompute_rotary_embeddings(self.rotary_block_size, d_model)
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

	def forward(self, idx, targets=None):
		B, T = idx.size()

		# grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim))
		assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
		cos_sin = self.cos[:, :+T], self.sin[:, :+T]

		# token embeddings of shape (b, t, n_embd)
		x = self.embed(idx)
		x = norm(x)

		for block in self.blocks:
			x = block(x, cos_sin)

		# forward the lm_head (compute logits)
		x = norm(x)
		logits = self.unembed(x)

		# if we are given some desired targets also calculate the loss
		loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction="mean")
		return logits, loss

	@torch.no_grad()
	def generate(self, idx, max_new_tokens, device, temperature=1.0, top_k=None):
		idx = torch.tensor(idx, dtype=torch.int64, device=device).unsqueeze(0)

		for _ in range(max_new_tokens):
			# our very first step, pass the initial sequence context to the model
			# if the sequence context is growing too long we must crop it at block_size
			idx_cond = idx[:, -self.rotary_block_size:] if idx.size(1) > self.rotary_block_size else idx

			# forward the model to get the logits for the index in the sequence
			logits, _ = self(idx_cond)
			logits = logits[:, -1, :]

			# https://github.com/karpathy/nanoGPT/pull/546/
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
		return idx
