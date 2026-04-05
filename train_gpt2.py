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