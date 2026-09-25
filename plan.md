# RAG Demo: Project Plan

## Purpose

A small Python project that demonstrates how a Retrieval-Augmented Generation (RAG) pipeline works, using a single PDF and the Anthropic API. Each stage (chunking, embedding, indexing, retrieval, augmentation) lives in its own module and can be run and inspected on its own, so the pipeline is visible rather than hidden inside a framework.

## Instructions for Claude Code

- Build **one phase at a time**, in order. Stop after each phase, run its "Done when" check, show the output, and wait for approval before starting the next phase.
- Do **not** use LangChain, LlamaIndex, or similar frameworks. Every stage should be plain, readable Python.
- Favour clarity over cleverness: this is a teaching project. Add short docstrings explaining *what each stage does in the RAG pipeline*, not only what the code does.
- Never print, log, or commit API keys.

## Key facts and decisions

- **Anthropic does not offer an embedding model.** The Anthropic key is used for generation only. Embeddings come from a local model, so no second API key is needed. (Voyage AI, Anthropic's recommended embeddings provider, was considered as an optional swap but dropped to keep setup lightweight: it needs its own key, and its Python package pulls in LangChain.)
- **Indexing uses NumPy only.** The NumPy index shows exactly what a vector index is: a matrix searched with one dot product (about 20 ms for ~1,000 chunks). A vector database such as Chroma was planned as a second backend but dropped to keep setup lightweight (see Phase 9); the `VectorIndex` interface leaves room to add one later.

## Tech stack

| Stage | Choice | Reason |
|---|---|---|
| PDF extraction | `pypdf` | Simple; keeps page numbers for citations |
| Chunking | Hand-written, two strategies | Compare fixed-size and paragraph-aware splits |
| Embedding | `sentence-transformers` (`all-MiniLM-L6-v2`) | Free, offline, small (384-dim vectors), no API key |
| Indexing | NumPy matrix + JSON metadata | Transparent, fast enough at this scale, no extra dependencies |
| Retrieval | Cosine similarity, top-k | Easy to print and explain scores |
| Augmentation + generation | `anthropic` SDK, `claude-sonnet-5` (or `claude-haiku-4-5-20251001` for lower cost) | Uses the existing API key |
| Interface | CLI built with `typer` + `rich` | Each stage becomes a demo command |
| Config / secrets | `python-dotenv` | Keys stay in `.env` |
| Tests | `pytest` | Chunker and index checks |

## Project structure

```
rag-demo/
├── data/                   # the source PDF goes here
├── index/                  # generated: vectors.npy, chunks.json, manifest.json, pages/ (extraction cache)
├── src/rag_demo/
│   ├── config.py           # chunk size, overlap, top-k, model names
│   ├── extract.py          # PDF -> list of (page_number, text)
│   ├── chunk.py            # text -> chunks with metadata
│   ├── embed.py            # Embedder interface: LocalEmbedder
│   ├── index.py            # VectorIndex interface: NumpyIndex
│   ├── retrieve.py         # query -> top-k chunks with scores
│   ├── augment.py          # chunks + question -> final prompt
│   ├── generate.py         # prompt -> Claude answer
│   └── cli.py              # one command per stage
├── tests/
├── eval/questions.json     # small hand-written Q&A set
├── .env.example            # ANTHROPIC_API_KEY=, optional EMBED_MODEL=
├── .gitignore              # .env, index/, data/*.pdf
├── pyproject.toml
├── plan.md
└── README.md
```

## Default configuration

| Setting | Default |
|---|---|
| `CHUNK_STRATEGY` | `fixed` |
| `CHUNK_SIZE` | 800 characters |
| `CHUNK_OVERLAP` | 150 characters |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` (or a path to a downloaded copy) |
| `TOP_K` | 5 |
| `CLAUDE_MODEL` | `claude-sonnet-5` |
| `MAX_TOKENS` | 1024 |

All settings live in `config.py` and can be overridden by CLI flags.

---

## Phase 1: Setup

- Create the project with `pyproject.toml`, a virtual environment, and a `rag-demo` console entry point.
- Load `ANTHROPIC_API_KEY` from `.env` via `python-dotenv`.
- Add `.env.example` and a `.gitignore` covering `.env`, `index/`, and `data/*.pdf`.
- Print a clear error if the Anthropic key is missing (without echoing any key value).

**Done when:** `rag-demo --help` lists the commands, and `rag-demo check` confirms the key loaded (showing only "found" or "missing").

## Phase 2: Extraction

- Read the PDF page by page with `pypdf`.
- Clean whitespace (collapse repeated spaces, fix hyphenated line breaks where simple).
- Keep the page number with each block of text.
- Warn when a page returns no text (a sign of a scanned, image-only page; OCR is out of scope).

**Done when:** `rag-demo extract` prints the page count, characters per page, any empty-page warnings, and a short preview of the first page.

## Phase 3: Chunking

- **Fixed-size with overlap:** slide a window of `CHUNK_SIZE` characters, stepping by `CHUNK_SIZE - CHUNK_OVERLAP`. Prefer to break at the nearest whitespace so words aren't split.
- **Paragraph-aware:** split on blank lines, then merge small paragraphs up to `CHUNK_SIZE`; split any paragraph that is larger than `CHUNK_SIZE`.
- Each chunk stores: `id`, `text`, `pages` (list), `start_char`, `end_char`, `length`, `strategy`.

**Done when:** `rag-demo chunk --strategy fixed` (and `--strategy paragraph`) shows the chunk count, a length histogram, and three sample chunks with the overlap region highlighted in colour.

## Phase 4: Embedding

- Define an `Embedder` interface with `embed_documents(texts)` and `embed_query(text)`.
- `LocalEmbedder` uses `sentence-transformers` with `all-MiniLM-L6-v2`.
- The model loads from the local copy when one exists, and only contacts huggingface.co on first use. `EMBED_MODEL` can point to a downloaded model folder for networks that block Hugging Face.
- Embed in batches and normalise every vector to unit length.

**Done when:** `rag-demo embed` prints the matrix shape (for a 142-chunk PDF: `142 x 384`), time taken, and the first eight numbers of one vector, so the audience sees what an embedding looks like.

## Phase 5: Indexing (NumPy)

- Define a `VectorIndex` interface: `add(vectors, chunks)`, `search(query_vector, k)`, `save(path)`, `load(path)`.
- `NumpyIndex` stacks vectors into one matrix saved as `vectors.npy`, with chunk metadata in `chunks.json`.
- Write `manifest.json` recording: embedding model name, vector dimension, chunk strategy, chunk size, overlap, PDF file name, PDF SHA-256 hash, and build timestamp.
- On load, refuse to use an index built with a different embedding model and explain why (mismatched embedding models are a classic silent RAG bug).

**Done when:** `rag-demo ingest` runs Phases 2 to 5 end to end, and `rag-demo index-info` prints the manifest.

## Phase 6: Retrieval

- Embed the query with the same embedder used to build the index.
- `NumpyIndex.search` computes `scores = vectors @ query_vector` (cosine similarity, since vectors are unit length) and returns the top-k with scores.
- Return results as a list of `(chunk, score)` pairs.

**Done when:** `rag-demo search "your question"` shows a ranked table with rank, score, page(s), and the first 150 characters of each chunk.

## Phase 7: Augmentation

- Build the final prompt from the question and retrieved chunks.
- System prompt instructs Claude to:
  - answer only from the supplied context,
  - cite chunk IDs and page numbers for each claim,
  - say plainly when the answer is not in the document, rather than guessing.
- Wrap each chunk in numbered tags, placed before the question:

```xml
<context>
  <chunk id="3" page="12">...chunk text...</chunk>
  <chunk id="7" page="15">...chunk text...</chunk>
</context>

<question>...</question>
```

**Done when:** `rag-demo ask "..." --show-prompt` prints the full assembled system prompt and user message, with token-count estimate, before anything is sent to the API.

## Phase 8: Generation

- Send the prompt with the `anthropic` SDK using `CLAUDE_MODEL`.
- Print the answer, the cited chunks and pages, and input/output token usage from the response.
- Add a `--no-rag` flag that sends the same question with no context, for a side-by-side comparison.
- Handle API errors (auth, rate limit, network) with clear messages.

**Done when:** `rag-demo ask "..."` returns a cited answer, and `rag-demo ask "..." --compare` shows the RAG and no-RAG answers side by side.

## Phase 9: Second index backend (Chroma) (skipped)

Skipped to keep setup lightweight. At this scale (~1,000 chunks) the NumPy index searches every chunk in about 20 ms, so a vector database brings no practical gain, while `chromadb` adds a large set of dependencies (onnxruntime, gRPC, OpenTelemetry), version-specific configuration, and telemetry to disable.

What a vector database adds only matters at larger scale: approximate nearest-neighbour search (e.g. HNSW) over millions of vectors, incremental updates instead of full rebuilds, metadata filtering, and running as a shared service. In a demo, this point can be made by walking through `index.py`. A backend can still be added later by implementing the `VectorIndex` interface; the manifest already records `index_backend`.

## Phase 10: Tests and evaluation

- **Unit tests (`pytest`):**
  - Chunker: overlap length is correct; concatenating chunks (minus overlaps) reproduces the source text; no empty chunks.
  - NumPy index: known vectors return known nearest neighbours; save/load round-trips.
  - Manifest: loading with a mismatched embedding model raises a clear error.
- **Retrieval evaluation:**
  - `eval/questions.json` holds 5 to 10 questions written from the PDF, each with the page number that holds the answer.
  - `rag-demo eval` reports hit rate at k = 1, 3, 5.
  - `rag-demo eval --chunk-size 400` (and other settings) rebuilds a temporary index and reruns, to show how chunking choices change retrieval quality.

**Done when:** `pytest` passes and `rag-demo eval` prints a results table.

## Phase 11: README

- One-paragraph explanation of RAG in plain language.
- A section per stage: what it does, why it matters, which module holds it, and the command that demonstrates it.
- A "Live demo script": the exact sequence of commands to run in front of an audience, from `extract` through `ask --compare`.
- Setup steps, including how to use a downloaded copy of the embedding model on networks that block Hugging Face.

**Done when:** a new user can follow the README from a fresh clone to a working `ask` command.

---

## Suggested live demo order

1. `rag-demo extract`: show raw text and page numbers.
2. `rag-demo chunk --strategy fixed`: show chunks and overlap.
3. `rag-demo embed`: show what a vector looks like.
4. `rag-demo ingest` then `rag-demo index-info`: show the stored index.
5. `rag-demo search "..."`: show ranked chunks and scores.
6. `rag-demo ask "..." --show-prompt`: show the augmented prompt.
7. `rag-demo ask "..." --compare`: RAG answer versus no-RAG answer.

## Out of scope (first version)

- OCR for scanned PDFs
- Multiple documents
- Web UI (a Streamlit page is a possible follow-up)
- Hybrid search (BM25 plus vectors) and reranking
- A vector database backend (Chroma); see Phase 9
