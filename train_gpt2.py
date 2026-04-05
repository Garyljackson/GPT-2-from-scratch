import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import tiktoken
from datasets import load_dataset

# ============================================================
# STEP 1: Load the TinyStories dataset
# ============================================================
# TinyStories is a dataset of simple children's stories.
# We use a small subset so this runs on any laptop.

print("Loading TinyStories dataset...")
ds = load_dataset("roneneldan/TinyStories")

# Use only 1000 stories for training, 100 for validation
train_subset = ds['train'].select(range(1000))
valid_subset = ds['validation'].select(range(100))

ds_small = {
    'train': train_subset,
    'validation': valid_subset
}

print(f"Training stories: {len(ds_small['train'])}")
print(f"Validation stories: {len(ds_small['validation'])}")

# Quick peek at the first story
print(f"\nFirst story preview: {ds_small['train'][0]['text'][:100]}...")

# ============================================================
# STEP 2: Set up the tokenizer
# ============================================================
# The tokenizer converts text to numbers (token IDs).
# We use the same tokenizer as GPT-2: tiktoken with "gpt2" encoding.

encoder = tiktoken.get_encoding("gpt2")

# Quick demo of how tokenization works
demo_text = "Hello, I'm a language model,"
demo_tokens = encoder.encode(demo_text)
print(f"\nTokenization demo:")
print(f"  Text:   '{demo_text}'")
print(f"  Tokens: {demo_tokens}")
print(f"  Back:   '{encoder.decode(demo_tokens)}'")

# ============================================================
# STEP 3: Create the PyTorch Dataset
# ============================================================
# This class does three things:
#   1. Tokenizes all stories into one long list of token IDs
#   2. Splits that list into fixed-length chunks (context windows)
#   3. Creates input/target pairs where the target is shifted by one position
#
# The shift is the key insight: given "The cat sat on", the model
# learns to predict "cat sat on the" — each token predicting the next.

class TinyStoriesDataset(Dataset):
    def __init__(self, split, encoder, context_length=128):
        self.encoder = encoder
        self.context_length = context_length

        cache_path = f"tinystories_{split}_tokens_small.pt"

        if os.path.exists(cache_path):
            print(f"Loading cached tokens from {cache_path}")
            self.tokens = torch.load(cache_path, weights_only=True)
        else:
            print(f"Tokenizing {split} split...")
            self.tokens = []
            for i in range(len(ds_small[split])):
                text = ds_small[split][i]['text']
                tokens = self.encoder.encode(text)
                self.tokens.extend(tokens)
                # Add end-of-text token between stories
                self.tokens.append(encoder.eot_token)

            self.tokens = torch.tensor(self.tokens, dtype=torch.long)
            torch.save(self.tokens, cache_path)
            print(f"Saved tokenized data to {cache_path}")

        print(f"  Total tokens: {len(self.tokens):,}")
        print(f"  Training chunks: {len(self.tokens) // self.context_length:,}")

    def __len__(self):
        return len(self.tokens) // self.context_length

    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start + self.context_length
        x = self.tokens[start:end]         # Input:  positions 0..127
        y = self.tokens[start+1:end+1]     # Target: positions 1..128
        return x, y


# Create the training dataset and data loader
train_dataset = TinyStoriesDataset('train', encoder, context_length=128)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Verify the input/target shift pattern
x, y = train_dataset[0]
print(f"\nVerifying data pipeline:")
print(f"  Input tokens [0:5]:  {x[:5].tolist()}")
print(f"  Target tokens [0:5]: {y[:5].tolist()}")
print(f"  Input text:  '{encoder.decode(x[:20].tolist())}'")
print(f"  Target text: '{encoder.decode(y[:20].tolist())}'")

# ============================================================
# STEP 4: Define the model configuration
# ============================================================
# These hyperparameters define the size of GPT-2 (124M version).
# The only change from the real GPT-2 is block_size: we use 128
# instead of 1024 to match our shorter context length.

@dataclass
class GPTConfig:
    block_size: int = 128    # Maximum sequence length
    vocab_size: int = 50257  # Number of tokens in GPT-2's vocabulary
    n_layer: int = 12        # Number of transformer blocks (depth)
    n_head: int = 12         # Number of attention heads (width)
    n_embd: int = 768        # Embedding dimension

# ============================================================
# STEP 5: The MLP (Feed-Forward Network)
# ============================================================
# Each transformer block contains an MLP that processes each token
# independently. It expands the representation to 4x the embedding
# dimension, applies a non-linearity (GELU), then projects back down.
#
# Think of it as: the attention layer figures out WHICH tokens matter,
# and the MLP figures out WHAT to do with that information.

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)  # 768 -> 3072
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)  # 3072 -> 768

    def forward(self, x):
        x = self.c_fc(x)       # Expand to higher dimension
        x = self.gelu(x)       # Non-linear activation
        x = self.c_proj(x)     # Compress back down
        return x

# ============================================================
# STEP 6: Causal Self-Attention
# ============================================================
# This is where tokens "look at" each other to build context.
# "Causal" means each token can only attend to tokens that came
# BEFORE it (not future tokens). This is enforced by the mask.
#
# Multi-head attention splits the 768-dim embedding into 12 heads
# of 64 dimensions each. Each head learns a different "type" of
# relationship (e.g., one head might track syntax, another semantics).

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            f"Embedding dim {config.n_embd} must be divisible by n_head {config.n_head}"

        # Combined Q, K, V projection (more efficient than three separate layers)
        # Input: 768 -> Output: 2304 (which gets split into three 768-dim tensors)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection: combines all heads back together
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Causal mask: a lower-triangular matrix of ones
        # Position i can only attend to positions 0..i (not future positions)
        # register_buffer means this is saved with the model but NOT trained
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # Batch, sequence length (Time), embedding dim (Channels)

        # Compute Q, K, V all at once, then split
        qkv = self.c_attn(x)                          # [B, T, 2304]
        q, k, v = qkv.split(self.n_embd, dim=2)       # 3x [B, T, 768]

        # Reshape into multiple heads: [B, T, 768] -> [B, 12, T, 64]
        # Each head gets 64 dimensions to work with
        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)  # [B, 12, T, 64]
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # [B, 12, T, 64]
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)  # [B, 12, T, 64]

        # Compute attention scores: how much should each token attend to each other?
        # q @ k^T gives a [T, T] matrix of attention scores per head
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Shape: [B, 12, T, 64] @ [B, 12, 64, T] = [B, 12, T, T]

        # Apply causal mask: future positions get -inf, which softmax turns into 0
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # Normalize to probabilities (each row sums to 1)
        att = F.softmax(att, dim=-1)

        # Weighted sum of values based on attention probabilities
        y = att @ v  # [B, 12, T, T] @ [B, 12, T, 64] = [B, 12, T, 64]

        # Reassemble all heads: [B, 12, T, 64] -> [B, T, 768]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection to let heads interact
        y = self.c_proj(y)
        return y
    
# ============================================================
# STEP 7: The Transformer Block
# ============================================================
# A block is: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
#
# GPT-2 uses "Pre-Norm" (normalize BEFORE the sub-layer), which is
# different from the original transformer paper (which normalizes after).
# Pre-Norm makes deep networks easier to train.

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)    # Norm before attention
        self.attn  = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)    # Norm before MLP
        self.mlp   = MLP(config)

    def forward(self, x):
        # Attention with residual connection
        # The "+ x" is the residual: it lets gradients flow directly through
        x = x + self.attn(self.ln_1(x))

        # MLP with residual connection
        x = x + self.mlp(self.ln_2(x))

        return x

# ============================================================
# STEP 8: The Full GPT Model
# ============================================================
# The complete architecture:
#   1. Token embeddings (vocab -> vectors)
#   2. Position embeddings (position -> vectors)
#   3. 12 transformer blocks
#   4. Final layer norm
#   5. Language model head (vectors -> vocab probabilities)

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # Token embedding: each of the 50,257 tokens gets a 768-dim vector
            wte = nn.Embedding(config.vocab_size, config.n_embd),

            # Position embedding: each of the 128 positions gets a 768-dim vector
            wpe = nn.Embedding(config.block_size, config.n_embd),

            # The stack of 12 transformer blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),

            # Final layer normalization (added by GPT-2, not in original transformer)
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        # Language model head: projects from 768 dims back to 50,257 vocab size
        # We create it as a proper Linear layer (needed for weight loading)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the output projection shares weights with the input embedding.
        # This means the model uses the SAME matrix to:
        #   - convert token IDs to vectors (input)
        #   - convert vectors back to token probabilities (output)
        # This reduces parameters and acts as a regularizer.
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block size {self.config.block_size}"

        # Create position indices: [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Look up embeddings
        tok_emb = self.transformer.wte(idx)   # [B, T, 768] — token content
        pos_emb = self.transformer.wpe(pos)   # [T, 768] — position info (broadcasts)

        # Combine token and position information
        x = tok_emb + pos_emb

        # Pass through all 12 transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm
        x = self.transformer.ln_f(x)

        # Project to vocabulary size to get next-token predictions
        logits = self.lm_head(x)  # [B, T, 50257]

        # Compute loss if we have targets (during training)
        loss = None
        if targets is not None:
            # Flatten for cross_entropy: it expects (N, C) and (N,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # [B*T, 50257]
                targets.view(-1)                    # [B*T]
            )

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Load pretrained GPT-2 weights from HuggingFace into our architecture."""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        print(f"Loading weights from pretrained gpt: {model_type}")

        # Configuration for each GPT-2 variant
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),   # 124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024  # Full GPT-2 uses 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        # Load HuggingFace model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith('.attn.masked_bias')
                      and not k.endswith('.attn.bias')]

        # OpenAI used Conv1D; we use Linear. These weights need transposing.
        transposed = [
            'attn.c_attn.weight',
            'attn.c_proj.weight',
            'mlp.c_fc.weight',
            'mlp.c_proj.weight'
        ]

        assert len(sd_keys_hf) == len(sd_keys), \
            f"Mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# ============================================================
# STEP 9: Verify our architecture matches OpenAI's GPT-2
# ============================================================
print("\n--- Architecture Verification ---")
print("Loading official GPT-2 weights into our model...")
model_verify = GPT.from_pretrained('gpt2')
print("SUCCESS! Our architecture matches OpenAI's GPT-2 exactly.")
del model_verify  # Free memory — we'll create a fresh model for training

# ============================================================
# STEP 10: Text generation function
# ============================================================
def generate(model, prompt, max_length=30, num_sequences=5, device='cpu'):
    """Generate text from a prompt using top-k sampling."""
    model.eval()

    tokens = encoder.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_sequences, 1)  # [num_sequences, prompt_len]
    x = tokens.to(device)

    torch.manual_seed(42)
    if device == 'cuda':
        torch.cuda.manual_seed(42)

    with torch.no_grad():
        while x.size(1) < max_length:
            logits, _ = model(x)
            # Take logits for the LAST token position only
            logits = logits[:, -1, :]  # [num_sequences, vocab_size]
            probs = F.softmax(logits, dim=-1)
            # Top-k sampling: only consider the 50 most likely next tokens
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # Sample from these top 50
            ix = torch.multinomial(topk_probs, 1)
            # Map back to actual token IDs
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    print(f"\nGenerated text (prompt: '{prompt}'):")
    for i in range(num_sequences):
        decoded = encoder.decode(x[i, :max_length].tolist())
        print(f"  > {decoded}")

# ============================================================
# STEP 11: See what an untrained model produces (should be gibberish)
# ============================================================
print("\n--- Untrained Model Output ---")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

config = GPTConfig()
model = GPT(config)
model = model.to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

generate(model, "Hello, I'm a language model,", device=device)