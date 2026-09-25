"""Stage 6 of the RAG pipeline: augmentation (question + chunks -> prompt).

This is the "A" in RAG. Claude has never seen this PDF, so the retrieved chunks
are pasted into the prompt as context, and the system prompt tells Claude how
to use them:

- answer only from the supplied context, not from general knowledge;
- cite the chunk ID and page for each claim, so every statement can be checked;
- say plainly when the answer is not in the document, rather than guessing.

The chunks go first, wrapped in numbered XML tags, and the question comes last.
Putting long material before the question, with clear tags around each piece,
helps the model keep track of which text came from where.

Nothing here calls an API: the output is just two strings (the system prompt
and the user message), so the whole prompt can be inspected before it is sent.
"""

import re
from dataclasses import dataclass

from rag_demo.retrieve import Result

SYSTEM_PROMPT = """\
You answer questions about a document ({pdf_name}) using only the excerpts \
supplied in <context>. Each excerpt is a <chunk> with an id and the page(s) \
it comes from.

Rules:
- Use only information in the excerpts. Do not add facts from general \
knowledge, even if you know them.
- Cite every claim with the chunk id and page it came from, in the form \
[chunk 12, p. 34]. For an excerpt spanning pages, use [chunk 12, pp. 34-35]. \
If a sentence draws on several excerpts, cite each one.
- If the excerpts do not contain the answer, say plainly that the document \
does not say. Then, if some excerpts are related, briefly say what they do \
cover. Do not guess.
- Be concise: answer the question directly, then add supporting detail."""

# How the answer cites a chunk: "[chunk 12, p. 34]". Used to find citations later.
CITATION_PATTERN = re.compile(r"\[chunk (\d+),\s*pp?\.\s*([\d\-–]+)\]")

# A rough rule of thumb for English text: about 4 characters per token.
CHARS_PER_TOKEN = 4


def page_attribute(pages: list[int]) -> str:
    return str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"


@dataclass
class Prompt:
    system: str
    user: str
    results: list[Result]  # the chunks included as context, best first

    @property
    def estimated_tokens(self) -> dict[str, int]:
        """Estimated token counts. The API reports exact figures after a call."""
        system = len(self.system) // CHARS_PER_TOKEN
        user = len(self.user) // CHARS_PER_TOKEN
        return {"system": system, "user": user, "total": system + user}


def build_context(results: list[Result]) -> str:
    """Wrap each retrieved chunk in a numbered <chunk> tag, best match first."""
    lines = ["<context>"]
    for r in results:
        c = r.chunk
        lines.append(f'  <chunk id="{c.id}" page="{page_attribute(c.pages)}">{c.text}</chunk>')
    lines.append("</context>")
    return "\n".join(lines)


def build_prompt(question: str, results: list[Result], pdf_name: str) -> Prompt:
    """Assemble the system prompt and the user message (context first, question last)."""
    user = f"{build_context(results)}\n\n<question>{question}</question>"
    return Prompt(system=SYSTEM_PROMPT.format(pdf_name=pdf_name), user=user, results=results)
