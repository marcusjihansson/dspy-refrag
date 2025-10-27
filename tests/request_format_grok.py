import os

import requests

# Load your API key (replace with your own key or set as env var)
api_key = os.getenv("OPENROUTER_MONEY_KEY")

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

data = {
    "model": "x-ai/grok-4-fast",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a short fact about black holes."},
    ],
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    print(response.json()["choices"][0]["message"]["content"])
else:
    print("Error:", response.status_code)
    print(response.text)
