import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

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


class OllamaLM:
    """
    HTTP client for Ollama chat completions.
    Designed solely for benchmarking: returns both text and usage reliably.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff: float = 1.5,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        # Use provided parameters, fall back to environment variables
        self.model = model or os.getenv("OLLAMA_MODEL")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY")
        if not self.model:
            raise ValueError("OLLAMA_MODEL must be set via parameter or environment variable.")
        if not self.base_url:
            raise ValueError("OLLAMA_BASE_URL must be set via parameter or environment variable.")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.extra_headers = extra_headers or {}
        # Clean up trailing slash
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _should_retry(self, status_code: int) -> bool:
        """Determine if request should be retried based on status code."""
        return status_code == 429 or (500 <= status_code < 600)

    def __call__(self, prompt: str) -> RequestLMResponse:
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(
                    url,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )

                # Check if we should retry
                if self._should_retry(r.status_code):
                    if attempt < self.max_retries:
                        sleep_s = self.backoff ** (attempt - 1)
                        time.sleep(sleep_s)
                        continue
                    else:
                        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

                # Raise for other HTTP errors (non-retryable)
                r.raise_for_status()

                data = r.json()

                # Validate and extract response
                if not isinstance(data, dict):
                    raise ValueError(f"Expected dict response, got {type(data)}")

                message = data.get("message")
                if not message or not isinstance(message, dict):
                    raise ValueError("Response missing 'message' object")

                text = message.get("content", "")

                # Ollama doesn't provide usage, so empty dict
                usage = {}

                return RequestLMResponse(text=text, usage=usage, raw=data)

            except requests.exceptions.RequestException as e:
                last_err = e
                # Only retry on connection/timeout errors
                if attempt < self.max_retries and isinstance(
                    e,
                    (requests.exceptions.ConnectionError, requests.exceptions.Timeout),
                ):
                    sleep_s = self.backoff ** (attempt - 1)
                    time.sleep(sleep_s)
                    continue
                else:
                    raise RuntimeError(f"Request failed: {e}") from e
            except (ValueError, KeyError) as e:
                # Don't retry parsing errors
                raise RuntimeError(f"Failed to parse response: {e}") from e
            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                sleep_s = self.backoff ** (attempt - 1)
                time.sleep(sleep_s)

        raise RuntimeError(
            f"OllamaLM failed after {self.max_retries} attempts: {last_err}"
        )


if __name__ == "__main__":
    # Simple test query
    try:
        lm = OllamaLM(
            model=os.getenv("OLLAMA_MODEL"),
            base_url=os.getenv("OLLAMA_BASE_URL"),
            api_key=os.getenv("OLLAMA_API_KEY"),
        )
        print("Using model:", lm.model)
        response = lm("Hello, world!")
        print("Response:", response.text)
        print("Usage:", response.usage)
    except Exception as e:
        print("Test failed:", e)