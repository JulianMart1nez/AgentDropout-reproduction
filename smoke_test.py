"""
Quick connectivity test: confirms the OpenRouter API key and base URL work,
using a tiny 5-token request (near-zero cost) before running anything larger.
Run with: python smoke_test.py
"""
import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")
print("BASE_URL:", base_url)
print("API_KEY starts with:", (api_key or "")[:12], "...")

client = OpenAI(base_url=base_url, api_key=api_key)
resp = client.chat.completions.create(
    model="meta-llama/llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "Reply with exactly the word: pong"}],
    max_tokens=5,
)
print("MODEL USED:", resp.model)
print("REPLY:", resp.choices[0].message.content)
print("USAGE:", resp.usage)
