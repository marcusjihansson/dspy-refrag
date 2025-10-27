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


class GrokLM:
    """
    Client for Grok via OpenRouter (OpenAI-compatible API).
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
        response_format: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Use provided parameters, fall back to environment variables
        self.model = model or os.getenv("GROK_MODEL", "x-ai/grok-4-fast")
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.api_key = (
            api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_MONEY_KEY")
        )
        if not self.model:
            raise ValueError("GROK_MODEL must be set via parameter or environment variable.")
        if not self.base_url:
            raise ValueError("OPENROUTER_BASE_URL must be set via parameter or environment variable.")
        if not self.api_key:
            raise ValueError(
                "API key not provided. Set OPENROUTER_API_KEY, OPENROUTER_MONEY_KEY, or OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.extra_headers = extra_headers or {}
        self.response_format = response_format
        # Clean up trailing slash
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # OpenRouter optional headers
        if "HTTP_REFERER" in os.environ:
            headers["HTTP-Referer"] = os.environ["HTTP_REFERER"]
        if "X_TITLE" in os.environ:
            headers["X-Title"] = os.environ["X_TITLE"]
        headers.update(self.extra_headers)
        return headers

    def _should_retry(self, status_code: int) -> bool:
        """Determine if request should be retried based on status code."""
        return status_code == 429 or (500 <= status_code < 600)

    def __call__(self, prompt: str) -> RequestLMResponse:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        if self.response_format is not None:
            payload["response_format"] = self.response_format

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

                choices = data.get("choices")
                if not choices or not isinstance(choices, list) or len(choices) == 0:
                    raise ValueError("Response missing 'choices' array")

                message = choices[0].get("message")
                if not message or not isinstance(message, dict):
                    raise ValueError("Response missing 'message' object")

                text = message.get("content", "")
                usage = data.get("usage") or {}

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
            f"GrokLM failed after {self.max_retries} attempts: {last_err}"
        )


if __name__ == "__main__":
    # Simple test query
    try:
        lm = GrokLM(
            model=os.getenv("GROK_MODEL"),
            base_url=os.getenv("OPENROUTER_BASE_URL"),
            api_key=os.getenv("OPENROUTER_MONEY_KEY"),
        )
        print("Using model:", lm.model)
        response = lm("Hello, world!")
        print("Response:", response.text)
        print("Usage:", response.usage)
    except Exception as e:
        print("Test failed:", e)