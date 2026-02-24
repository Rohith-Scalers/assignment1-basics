import os
from typing import BinaryIO
import torch

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def get_stats(ids):
    pairs = {}
    for i in range(1, len(ids)):
        pair = (ids[i-1], ids[i])
        pairs[pair] = pairs.get(pair, 0) + 1
    return pairs
def merge(ids, merger, new_idx):
    new_ids = []
    i = 0
    
    while i < len(ids):
        # Check if we can form a pair
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == tuple(merger):
            new_ids.append(new_idx)
            i += 2  # Skip the next token (since merged)
        else:
            new_ids.append(ids[i])
            i += 1
    
    return new_ids
def train(text, vocab_size):
    ids = list(text.encode("utf-8"))
    # The vocab maps ID -> full bytes representation
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {} 
    
    num_merges = vocab_size - 256
    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats: break
        
        pair = max(stats, key=stats.get)
        new_id = 256 + i
        
        # KEY PART: Update vocab with the concatenated bytes
        p0, p1 = pair
        vocab[new_id] = vocab[p0] + vocab[p1] # This is already bytes!
        
        merges[pair] = new_id
        ids = merge(ids, pair, new_id)
        
    return merges, vocab
def encode(text, vocab,merges):
    byte_ids = list(text.encode("utf-8"))
    # Apply merges in order they were learned
    for new_id in sorted(vocab.keys()):
        if new_id < 256:
            continue
        pair = next(k for k, v in merges.items() if v == new_id)
        byte_ids = merge(byte_ids, pair, new_id)
    
    return byte_ids
def decode(byte_ids, vocab):
    decoded_bytes = bytearray()
    for byte_id in byte_ids:
        decoded_bytes.extend(vocab[byte_id])
    return decoded_bytes.decode("utf-8", errors="ignore")

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim)) 
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = torch.mean(x,dim=-1, keepdim=True)
        variance = torch.var(x,dim=-1, keepdim=True)
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps) 
        return self.gamma * x_normalized + self.beta

import torch.nn as nn
import torch.nn.functional as F

class GELUMLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c_fc    = nn.Linear(d_model, 4 * d_model)
        self.c_proj  = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        # x: (Batch, Seq_Len, Embed)
        
        # 1. Expand
        x = self.c_fc(x) # (Batch, Seq_Len, 4 * Embed)
        
        # 2. Activate
        x = F.gelu(x) 
        
        # 3. Contract
        x = self.c_proj(x) # (Batch, Seq_Len, Embed)
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        # One big linear layer to get Q, K, V all at once
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # Final projection layer
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size() # Batch, Time (Seq_Len), Channels (Embed)

        # 1. Get Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # 2. Reshape for Multi-Head (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # --- YOUR TURN: THE ATTENTION MATH ---
        
        # A. Calculate raw scores (B, nh, T, T)
        # att = ... 
                                                # A. Calculate raw scores
        att = torch.matmul(q, k.transpose(-2, -1)) # (B, nh, T, T)
        
        # B. Scale by 1/sqrt(head_dim)
        head_dim = q.size(-1)
        att = att / math.sqrt(head_dim) 
        
        # C. Apply Causal Mask
        # register_buffer is usually used for masks, but this works:
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        
        # D. Softmax and Apply to V
        att = torch.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, hs)

        # 3. Reassemble: (B, nh, T, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y)
    
class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln_1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = GELUMLP(d_model)

    def forward(self, x):
        # 1. Path 1: Residual + Attention(LayerNorm(x))
        # x = x + self.attn(self.ln_1(x))
        
        # 2. Path 2: Residual + MLP(LayerNorm(x))
        # x = ...
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

## Usage
with open(..., "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
