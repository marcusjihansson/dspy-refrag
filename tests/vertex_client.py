import os

from google import genai
from google.genai.types import GenerateContentConfig

# Initialize the client for Vertex AI
client = genai.Client(
    vertexai=True, project=os.getenv("PROJECT_ID"), location=os.getenv("LOCATION")
)

# Use Gemini
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="What is the capital of France?",
    config=GenerateContentConfig(temperature=0.7, max_output_tokens=1024),
)
print(response.text)

# Use Gemini
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is the capital of France?",
    config=GenerateContentConfig(temperature=0.7, max_output_tokens=1024),
)
print(response.text)

