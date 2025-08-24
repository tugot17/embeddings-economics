import requests
from text_generator import TextGenerator

text_generator = TextGenerator()
prompts = text_generator.generate_batch(batch_size=16, length_distribution="long")

requests.post("http://localhost:8000/start_profile")

response = requests.post("http://localhost:8000/v1/embeddings", 
                       json={"model": "Qwen/Qwen3-Embedding-4B", "input": prompts})

requests.post("http://localhost:8000/stop_profile")

embeddings = response.json()["data"]
print(f"Got {len(embeddings)} embeddings")