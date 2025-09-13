import requests
from openai import OpenAI
from tokenomics.embedding_benchmark import TextGenerator

base_url = "http://localhost:8000"
client = OpenAI(base_url=f"{base_url}/v1", api_key="dummy")

# Initialize text generator
text_gen = TextGenerator()

# Start profiler
requests.post(f"{base_url}/start_profile")

HOW_MANY_WORDS = 800
BATCH_SIZE = 1

response = client.embeddings.create(
    input=[text_gen.generate_text(HOW_MANY_WORDS)] * BATCH_SIZE,
    model="Qwen/Qwen3-Embedding-8B"
)

print(response.data[0].embedding)

# Stop profiler
requests.post(f"{base_url}/stop_profile")