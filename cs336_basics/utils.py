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

class GQA(nn.Module):
    def __init__(self, seq_len,embed_dim,num_heads=8,heads_kv=4):
        super().__init__()
        self.num_heads = num_heads
        self.hkv_dim = hkv_dim
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim,(num_heads // heads_kv) * embed_dim)
        self.wv = nn.Linear(embed_dim,(num_heads // heads_kv) * embed_dim)
    def forward(self, x):
        # x: (Batch, Seq_Len, Embed)
        q = self.wq(x) # (Batch, Seq_Len, Embed)
        k = self.wk(x) # (Batch, Seq_Len, (Num_Heads // Heads_KV) * Embed)
        v = self.wv(x) # (Batch, Seq_Len, (Num_Heads // Heads_KV) * Embed)
        return q, k, v
    def attention(self, q, k, v):
        # q: (Batch, Seq_Len, Embed)
        # k: (Batch, Seq_Len, (Num_Heads // Heads_KV) * Embed)
        # v: (Batch, Seq_Len, (Num_Heads // Heads_KV) * Embed)
        batch_size, seq_len, _ = q.size()
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2) # (Batch, Num_Heads, Seq_Len, Head_Dim)
        k = k.view(batch_size, seq_len, self.num_heads // self.heads_kv, -1).transpose(1, 2) # (Batch, Num_Heads // Heads_KV, Seq_Len, Head_Dim)
        v = v.view(batch_size, seq_len, self.num_heads // self.heads_kv, -1).transpose(1, 2) # (Batch, Num_Heads // Heads_KV, Seq_Len, Head_Dim)
        attn_scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(q.size(-1)) # (Batch, Num_Heads // Heads_KV, Seq_Len, Seq_Len)
        attn_weights = torch.softmax(attn_scores, dim=-1) # (Batch, Num_Heads // Heads_KV, Seq_Len, Seq_Len)
        attn_output = torch.einsum("bhqk,bhvd->bhqd", attn_weights, v) # (Batch, Num_Heads // Heads_KV, Seq_Len, Head_Dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # (Batch, Seq_Len, Embed)
        return attn_output
        
        


    