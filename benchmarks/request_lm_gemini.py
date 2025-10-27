import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from google import genai
from google.genai.types import GenerateContentConfig

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # If dotenv is not installed, assume env vars are set manually


@dataclass
class RequestLMResponse:
    text: str
    usage: Dict[str, Any]
    raw: Optional[Dict[str, Any]] = None


class GeminiLM:
    """
    Client for Gemini via Vertex AI.
    Designed solely for benchmarking: returns both text and usage reliably.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff: float = 1.5,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
    ) -> None:
        # Use provided parameters, fall back to environment variables
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        self.project_id = project_id or os.getenv("PROJECT_ID")
        self.location = location or os.getenv("LOCATION")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.project_id:
            raise ValueError("PROJECT_ID must be set via parameter or environment variable.")
        if not self.location:
            raise ValueError("LOCATION must be set via parameter or environment variable.")

        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        # Initialize the client
        # For Vertex AI, use project and location; API key is not used
        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
        )

    def __call__(self, prompt: str) -> RequestLMResponse:
        config = GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )

                text = response.text if response.text else ""

                # Vertex AI may not provide detailed usage, so use empty dict
                usage = {}

                # Convert response to dict for raw if possible
                raw = response.__dict__ if hasattr(response, '__dict__') else None

                return RequestLMResponse(text=text, usage=usage, raw=raw)

            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    sleep_s = self.backoff ** (attempt - 1)
                    time.sleep(sleep_s)
                else:
                    raise RuntimeError(f"GeminiLM failed after {self.max_retries} attempts: {e}") from e

        raise RuntimeError(
            f"GeminiLM failed after {self.max_retries} attempts: {last_err}"
        )


if __name__ == "__main__":
    # Simple test query
    try:
        lm = GeminiLM(
            model=os.getenv("GEMINI_MODEL"),
            project_id=os.getenv("PROJECT_ID"),
            location=os.getenv("LOCATION"),
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        print("Using model:", lm.model)
        response = lm("Hello, world!")
        print("Response:", response.text)
        print("Usage:", response.usage)
    except Exception as e:
        print("Test failed:", e)