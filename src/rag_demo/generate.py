"""Stage 7 of the RAG pipeline: generation (prompt -> Claude's answer).

This is the "G" in RAG, and the only stage that calls an API. The system
prompt and user message built by augment.py are sent to Claude, which writes
an answer grounded in the retrieved chunks, citing them as [chunk 12, p. 34].

For comparison, the same question can be sent with no context at all
("no-RAG"). Claude then answers from general knowledge: often plausible, but
with no citations and no guarantee it matches what this document says.

The API key is read from .env (never printed). Errors are turned into short,
plain explanations: a bad key, rate limits, network problems, a bad model name.
"""

import time
from dataclasses import dataclass, field

import anthropic

from rag_demo import config
from rag_demo.augment import CITATION_PATTERN, Prompt


class GenerationError(RuntimeError):
    """Raised when Claude could not be reached or returned no answer."""


@dataclass
class Citation:
    chunk_id: int
    pages: str  # as written by Claude, e.g. "377" or "376-377"
    in_context: bool  # False means Claude cited a chunk it was never shown


@dataclass
class Answer:
    text: str
    model: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    seconds: float
    citations: list[Citation] = field(default_factory=list)


def _client() -> anthropic.Anthropic:
    try:
        return anthropic.Anthropic(api_key=config.require_anthropic_api_key())
    except config.MissingAPIKeyError as err:
        raise GenerationError(str(err)) from err


def _send(system: str | None, user: str, model: str, max_tokens: int) -> Answer:
    """Send one message to Claude and return its answer with token usage."""
    request = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user}],
        # A short, grounded answer needs no extended reasoning. Turning thinking
        # off keeps the whole max_tokens budget for the answer itself, and keeps
        # this stage a plain prompt -> answer call.
        "thinking": {"type": "disabled"},
    }
    if system:
        request["system"] = system

    started = time.perf_counter()
    try:
        response = _client().messages.create(**request)
    except anthropic.AuthenticationError as err:
        raise GenerationError("The API rejected the key (401). Check ANTHROPIC_API_KEY in .env.") from err
    except anthropic.PermissionDeniedError as err:
        raise GenerationError(f"The API key is not allowed to use {model} (403).") from err
    except anthropic.NotFoundError as err:
        raise GenerationError(f"Model '{model}' was not found (404). Check CLAUDE_MODEL or --model.") from err
    except anthropic.RateLimitError as err:
        wait = err.response.headers.get("retry-after", "a few")
        raise GenerationError(f"Rate limited (429), even after retrying. Wait {wait} seconds and try again.") from err
    except anthropic.BadRequestError as err:
        raise GenerationError(f"The API rejected the request (400): {err.message}") from err
    except anthropic.APIStatusError as err:
        raise GenerationError(f"The API returned an error ({err.status_code}): {err.message}") from err
    except anthropic.APIConnectionError as err:  # includes timeouts
        raise GenerationError(
            "Could not reach api.anthropic.com. Check your internet connection (or VPN/proxy)."
        ) from err
    seconds = time.perf_counter() - started

    if response.stop_reason == "refusal":
        raise GenerationError("Claude declined to answer this question.")
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise GenerationError(f"Claude returned no text (stop reason: {response.stop_reason}).")

    return Answer(
        text=text,
        model=response.model,
        stop_reason=response.stop_reason,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        seconds=seconds,
    )


def find_citations(text: str, prompt: Prompt) -> list[Citation]:
    """Find the [chunk N, p. X] citations in an answer, in order, without repeats.

    Each citation is checked against the chunks actually supplied as context,
    so a citation to a chunk Claude was never shown is flagged.
    """
    supplied = {r.chunk.id for r in prompt.results}
    seen: set[int] = set()
    citations = []
    for chunk_id, pages in CITATION_PATTERN.findall(text):
        chunk_id = int(chunk_id)
        if chunk_id not in seen:
            seen.add(chunk_id)
            citations.append(Citation(chunk_id, pages, in_context=chunk_id in supplied))
    return citations


def generate_answer(
    prompt: Prompt, model: str = config.CLAUDE_MODEL, max_tokens: int = config.MAX_TOKENS
) -> Answer:
    """RAG: answer using the retrieved chunks, with citations."""
    answer = _send(prompt.system, prompt.user, model, max_tokens)
    answer.citations = find_citations(answer.text, prompt)
    return answer


def generate_without_context(
    question: str, model: str = config.CLAUDE_MODEL, max_tokens: int = config.MAX_TOKENS
) -> Answer:
    """No-RAG: the same question with no document context and no system prompt."""
    return _send(None, question, model, max_tokens)
