import json
import logging
import time
from types import SimpleNamespace

from groq import APIConnectionError, APIStatusError, BadRequestError, RateLimitError

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3

# Served instead of a 500 so the frontend always receives a valid answer payload.
FALLBACK_ANSWER = {
    "answer": "I couldn't generate a grounded answer for this question. Please try rephrasing it.",
    "citations": [],
}


def call_groq_with_fallback(client, *, model, messages, temperature, max_tokens):
    """Return (raw_text, usage) and never raise on model-side failures.

    Strict JSON mode hard-fails with a 400 when the model emits nothing
    (e.g. when it abstains), so retries drop JSON mode and raise the
    temperature slightly to avoid the same empty generation.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": min(1.0, temperature + 0.2 * (attempt - 1)),
            "max_tokens": max_tokens,
        }
        if attempt == 1:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            logger.warning("Attempt %d: Groq rejected generation: %s", attempt, exc)
            continue
        except (RateLimitError, APIConnectionError) as exc:
            logger.warning("Attempt %d: transient Groq error: %s", attempt, exc)
            time.sleep(attempt)
            continue
        except APIStatusError:
            logger.exception("Attempt %d: non-retryable Groq status", attempt)
            break

        content = response.choices[0].message.content if response.choices else None
        if content and content.strip():
            return content, response.usage
        logger.warning("Attempt %d: empty model output", attempt)

    return json.dumps(FALLBACK_ANSWER), SimpleNamespace(prompt_tokens=0, completion_tokens=0)
