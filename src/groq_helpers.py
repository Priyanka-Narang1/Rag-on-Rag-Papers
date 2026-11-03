import json
import logging
import re

from groq import APIConnectionError, APIStatusError, BadRequestError, RateLimitError

logger = logging.getLogger(__name__)

MODEL = "llama-3.1-8b-instant"
MAX_ATTEMPTS = 3

# Returned instead of a 500 so the frontend always gets a valid payload.
FALLBACK_RESPONSE = {
    "answer": "I couldn't generate a grounded answer for this question. Please try rephrasing it.",
    "citations": [],
    "key_takeaways": [],
    "abstain": True,
}


def _extract_json(text):
    """Parse JSON even when the model wraps it in prose or code fences."""
    if not text or not text.strip():
        return None
    candidate = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _call_groq(client, messages, json_mode):
    kwargs = {"model": MODEL, "messages": messages, "temperature": 0.1, "max_tokens": 1024}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    if not response.choices:
        return None
    return response.choices[0].message.content


def generate_structured_answer(client, messages):
    """Strict JSON mode first, then free-text parsing.

    Strict mode hard-fails (400) on empty output, which happens when the
    model abstains in plain text.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        json_mode = attempt == 1
        try:
            parsed = _extract_json(_call_groq(client, messages, json_mode))
            if parsed is not None:
                return parsed
            logger.warning("Attempt %d: model output was not parseable JSON", attempt)
        except BadRequestError as exc:
            logger.warning("Attempt %d: Groq rejected generation: %s", attempt, exc)
        except (RateLimitError, APIConnectionError) as exc:
            logger.warning("Attempt %d: transient Groq error: %s", attempt, exc)
        except APIStatusError:
            logger.exception("Attempt %d: non-retryable Groq status", attempt)
            break
    return dict(FALLBACK_RESPONSE)
