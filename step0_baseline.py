from transformers import pipeline, set_seed

# Load the official GPT-2 (124M parameter version)
# This will download ~500MB of model weights the first time
generator = pipeline('text-generation', model='gpt2')

set_seed(42)

results = generator(
    "Hello, I'm a language model,",
    max_length=30,              # Generate 30 tokens total (including the prompt)
    num_return_sequences=5      # Generate 5 different completions
)

for r in results:
    print(r['generated_text'])
    print("---")