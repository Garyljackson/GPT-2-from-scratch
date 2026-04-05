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