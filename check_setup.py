import torch
import tiktoken
import transformers

print(f"PyTorch version: {torch.__version__}") 
print(f"Transformers version: {transformers.__version__}") 

if torch.cuda.is_available(): 
	print("Success! GPU detected: NVIDIA CUDA") 
elif torch.backends.mps.is_available(): 
	print("Success! GPU detected: Apple Metal (MPS)") 
else: 
	print("No GPU detected. Running on CPU (this is fine, just slower).")