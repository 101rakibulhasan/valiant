# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="neoai-inc/Llama-3-neoAI-8B-Chat-v0.1")
pipe(messages)