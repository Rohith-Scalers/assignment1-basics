import torch 
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def logging_wrapper(func):
        def wrapper(self, x):
            print(f"Input shape: {x.shape}")
            output = func(self, x)
            print(f"Output shape: {output.shape}")
            return output
        return wrapper
    @logging_wrapper
    def forward(self, x):
        # x: (Batch, Seq_Len, Embed)
        norm_x = x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return norm_x * self.weight

class Rope:
    def __init__(self, dim, max_seq_len=2048):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("sin_cache", torch.zeros(max_seq_len, dim // 2))
        self.register_buffer("cos_cache", torch.zeros(max_seq_len, dim // 2))
        self._build_cache()

    def _build_cache(self):
        seq = torch.arange(self.max_seq_len)
        freqs = torch.einsum("i,j->ij", seq, self.inv_freq)
        self.sin_cache.copy_(torch.sin(freqs))
        self.cos_cache.copy_(torch.cos(freqs))

    def forward(self, x):
        # x: (Batch, Seq_Len, Embed)
        seq_len = x.size(1)
        sin = self.sin_cache[:seq_len]
        cos = self.cos_cache[:seq_len]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rotated


class SiluMLP(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.w1    = nn.Linear(embed_dim, (8 * embed_dim) // 3)
        self.w2    = nn.Linear(embed_dim, (8 * embed_dim) // 3)
        self.w3    = nn.Linear((8 * embed_dim) // 3, embed_dim)

    def forward(self, x):

        # x: (Batch, Seq_Len, Embed)
        
        # 1. Expand
        x1 = self.w1(x) # (Batch, Seq_Len, (8 * Embed) // 3)
        x2 = self.w2(x) # (Batch, Seq_Len, (8 * Embed) // 3)
        
        # 2. Activate
        x1 = F.silu(x1) # (Batch, Seq_Len, (8 * Embed) // 3)
        
        # 3. Gating
        x = x1 * x2     # (Batch, Seq_Len, (8 * Embed) // 3)
        
        # 4. Project back
        return self.w3(x) # (Batch, Seq_Len, Embed)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GQA(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads # How many Qs per KV
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match the number of Query heads"""
        if n_rep == 1:
            return x
        bs, n_kv_heads, slen, head_dim = x.shape
        # This expands the head dimension by n_rep
        return (
            x[:, :, None, :, :]
            .expand(bs, n_kv_heads, n_rep, slen, head_dim)
            .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
        )

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        
        # 1. Linear Projections
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 2. Reshape for Multi-Head (B, H, S, D)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 3. THE GQA MAGIC: Repeat KV heads to match Q
        # xk/xv go from (B, n_kv_heads, S, D) -> (B, n_heads, S, D)
        keys = self.repeat_kv(xk, self.n_rep)
        values = self.repeat_kv(xv, self.n_rep)

        # 4. Scaled Dot-Product Attention
        # Now shapes match: Q(B, 8, S, D) and K(B, 8, S, D)
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal Mask (Optional but standard for GPT)
        mask = torch.tril(torch.ones(seqlen, seqlen, device=x.device)).view(1, 1, seqlen, seqlen)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        
        probs = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(probs, values) # (B, n_heads, S, D)

        # 5. Restore original shape
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
        
        


    