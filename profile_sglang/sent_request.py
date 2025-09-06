import requests
from openai import OpenAI

base_url = "http://localhost:8000"
client = OpenAI(base_url=f"{base_url}/v1", api_key="dummy")

# Start profiler
requests.post(f"{base_url}/start_profile")

response = client.embeddings.create(
    input=[f"Your text string goes here {i}"*1000 for i in range(1)],
    model="Qwen/Qwen3-Embedding-8B"
)

print(response.data[0].embedding)

# Stop profiler
requests.post(f"{base_url}/stop_profile")