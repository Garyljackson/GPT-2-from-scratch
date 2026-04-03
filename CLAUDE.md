# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPT-2 implementation from scratch, following the [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) tutorial. Educational project focused on understanding LLM internals through hands-on implementation with PyTorch.

## Environment

- **Container**: PyTorch 2.11 + CUDA 12.8 (devcontainer with NVIDIA GPU support)
- **Package manager**: `uv` (installed system-wide; use `uv pip install --system` for new packages)
- **Python dependencies**: `pip install -r requirements.txt` (already installed in container)
- **Key libraries**: PyTorch, tiktoken, matplotlib, TensorFlow (for weight loading/comparison), tqdm, numpy, pandas

## Commands

```bash
# Install dependencies
uv pip install --system --no-cache -r requirements.txt

# Run a Python script
python <script.py>

# Run Jupyter
jupyter lab

# Run a single test file
python -m pytest <test_file.py>

# Run a specific test
python -m pytest <test_file.py>::<test_name> -v
```

## Conventions

- Model weights and checkpoints (`.bin`, `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.h5`) are gitignored — never commit them.
- Jupyter notebooks may be used for experimentation; source files for reusable code.
