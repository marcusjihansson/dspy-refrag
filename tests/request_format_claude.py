import os

import requests

api_key = os.getenv("OPENROUTER_MONEY_KEY")

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

data = {
    "model": "anthropic/claude-3-haiku",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a short fact about black holes."},
    ],
    "temperature": 0.7,
    "max_tokens": 300,
}

response = requests.post(url, headers=headers, json=data)

if response.ok:
    print(response.json()["choices"][0]["message"]["content"])
else:
    print("Error:", response.status_code)
    print(response.text)
