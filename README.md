# RAG Demo

A small Python project that shows, stage by stage, how Retrieval-Augmented Generation works over a single PDF, using a local embedding model and Claude. There is no framework: each stage is a short, readable module with its own command, so you can run it and see what it produced.

## What is RAG?

A language model like Claude knows a lot, but it has never read *your* document, and when it answers from general knowledge there is no way to check where a claim came from. Retrieval-Augmented Generation fixes this in three steps. First, the document is cut into small passages and each one is turned into a list of numbers (an *embedding*) that captures its meaning. When a question arrives, it is embedded the same way, and the passages whose numbers are closest (the most similar in meaning) are **retrieved**. Those passages are pasted into the prompt, **augmenting** the question with the evidence, and Claude **generates** an answer that uses only that evidence and cites the passage and page for each claim. If the document doesn't contain the answer, Claude says so instead of guessing.

```
PDF ─► extract ─► chunk ─► embed ─► index          (done once, by `ingest`)
                                      │
question ─► embed ─► retrieve top-k ◄─┘
                         │
                         ▼
                      augment (prompt = chunks + question) ─► Claude ─► cited answer
```

## Setup

You need Python 3.10 or newer, an [Anthropic API key](https://console.anthropic.com/), and a PDF. The embedding model runs locally and needs no key.

**1. Clone the repository and create a virtual environment.**

```bash
git clone <this repository's URL> rag-demo
cd rag-demo
python -m venv .venv
```

Activate it. On Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

**2. Install the project.** This installs the `rag-demo` command and its dependencies, including PyTorch (a large download; the CPU version is enough).

```bash
pip install -e ".[dev]"
```

On Windows, if this fails with *"Could not install packages due to an OSError"* and a hint about long paths, the project folder's path is too long for some of PyTorch's files (Windows limits paths to 260 characters by default). Clone to a shorter path such as `C:\code\rag-demo`, or [enable long path support](https://pip.pypa.io/warnings/enable-long-paths) in Windows.

**3. Add your API key.** Copy the example file and put your key after `ANTHROPIC_API_KEY=`. The `.env` file is ignored by git, and the key is never printed.

```bash
cp .env.example .env
```

(On Windows without a Unix shell: `copy .env.example .env`.) Then check it loaded:

```bash
rag-demo check
```

**4. Add a PDF.** Put exactly one PDF in `data/` (or pass `--pdf path/to/file.pdf` to any command). PDFs in `data/` are ignored by git. The questions in `eval/questions.json` were written for *Salt, Fat, Acid, Heat*; for another PDF, write your own (see [Evaluation](#evaluation)).

**5. Build the index.** This extracts, chunks and embeds the PDF, then saves the index to `index/`. The first run downloads the embedding model (~90 MB) from huggingface.co; after that it runs offline. For a ~500-page book, allow a minute or two.

```bash
rag-demo ingest
```

**6. Ask a question.**

```bash
rag-demo ask "How long should I brine a turkey?"
```

### If huggingface.co is blocked on your network

Some corporate networks and VPNs block the site the embedding model downloads from; `ingest` then stops with *"Could not load the embedding model"*. Either:

- connect once from a network that allows it (for example, turn off the VPN) and run `rag-demo ingest`. The model is saved on your machine and later runs never contact the site; or
- download the model folder on another machine, copy it over, and add its path to `.env`. To download it: `pip install huggingface_hub`, then `hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir all-MiniLM-L6-v2`. (A plain `git clone` of the model also works, but only with Git LFS installed; without it you get placeholder files.)

  ```
  EMBED_MODEL=C:\path\to\all-MiniLM-L6-v2
  ```

## The pipeline, stage by stage

Each stage has its own module in `src/rag_demo/` and a command that runs it on its own and shows the result.

### 1. Extraction: PDF → text

**What it does:** reads the PDF page by page with `pypdf` and cleans up the text: tabs and repeated spaces collapse, lines wrapped by the page layout are joined back into paragraphs, and words hyphenated across a line break are rejoined. Each page keeps its page number.

**Why it matters:** everything downstream works on this text, and the page numbers are what make citations possible. Pages that produce no text (scanned pages or illustrations) are reported, because they can never be found by search. OCR is out of scope.

**Module:** `extract.py` · **Command:** `rag-demo extract` (options: `--page N` to preview a page, `--all` to list every page). The extracted text is cached in `index/pages/` so later stages start instantly.

### 2. Chunking: text → passages

**What it does:** cuts the text into chunks of at most `CHUNK_SIZE` characters, never splitting a word. Two strategies:

- `fixed`: a window that slides along the text; neighbouring chunks overlap by about `CHUNK_OVERLAP` characters, so a sentence cut at one chunk's edge appears whole in the next.
- `paragraph`: follows the author's paragraphs, merging short ones and splitting long ones into equal pieces. No overlap.

Every chunk records its character offsets and the page(s) it came from.

**Why it matters:** a chunk is the unit that gets retrieved and shown to Claude. Small chunks match precisely but can cut an idea in half; large ones keep context but blur several topics into one embedding.

**Module:** `chunk.py` · **Command:** `rag-demo chunk --strategy fixed` (or `paragraph`; also `--chunk-size`, `--overlap`). Shows the chunk count, a length histogram, and three sample chunks with the overlapping text highlighted.

### 3. Embedding: passages → vectors

**What it does:** turns each chunk into a list of 384 numbers with the local model `all-MiniLM-L6-v2` (via `sentence-transformers`), in batches, and scales every vector to length 1.

**Why it matters:** texts with similar meaning get vectors that point in similar directions, so search works by meaning rather than by matching words. Because every vector has length 1, the similarity of two vectors is just their dot product. Anthropic doesn't offer an embedding model, so this stage runs locally and needs no API key.

**Module:** `embed.py` · **Command:** `rag-demo embed`. Shows the matrix shape (e.g. `1,083 x 384`), the time taken, and the first numbers of one vector.

### 4. Indexing: vectors → a searchable store

**What it does:** saves the vectors as one matrix (`index/vectors.npy`), the chunks as JSON (`index/chunks.json`), and a manifest recording how the index was built: embedding model, chunk settings, PDF name and SHA-256 hash, and build time.

**Why it matters:** embedding is the slow part, so it is done once. The manifest guards against a classic silent RAG bug: querying with a different embedding model from the one that built the index. The results would look confident but mean nothing, so the index refuses to load and explains why instead.

**Module:** `index.py` · **Commands:** `rag-demo ingest` runs stages 1 to 4 and saves the index; `rag-demo index-info` prints the manifest and checks it still matches the current model and PDF.

### 5. Retrieval: question → the most relevant passages

**What it does:** embeds the question with the same model, computes its similarity to every chunk with one matrix multiplication, and returns the top k (default 5) with their scores.

**Why it matters:** this is the "R" in RAG. The answer can never be better than the chunks retrieved here. Scores are cosine similarities: a good match on this book scores around 0.6 to 0.75, an unrelated chunk around 0.2.

**Module:** `retrieve.py` · **Command:** `rag-demo search "your question"` (`-k` sets how many). Shows a ranked table of score, page(s), chunk ID, and the start of each chunk. No API call is made.

### 6. Augmentation: passages + question → prompt

**What it does:** builds the prompt. The system prompt tells Claude to answer only from the supplied excerpts, to cite every claim as `[chunk 12, p. 34]`, and to say plainly when the document doesn't contain the answer. The user message wraps each chunk in a numbered tag, with the question last:

```xml
<context>
  <chunk id="831" page="376-377">...chunk text...</chunk>
  <chunk id="833" page="377">...chunk text...</chunk>
</context>

<question>How long should I brine a turkey?</question>
```

**Why it matters:** this is the "A" in RAG. The instructions are what turn a general-purpose model into one that sticks to the document and shows its sources.

**Module:** `augment.py` · **Command:** `rag-demo ask "your question" --show-prompt`. Prints the exact system prompt and user message with an estimated token count, and sends nothing.

### 7. Generation: prompt → cited answer

**What it does:** sends the prompt to Claude (`claude-sonnet-5` by default) with the `anthropic` SDK, prints the answer, lists the cited chunks with their pages and retrieval ranks, and reports the input and output tokens. A citation to a chunk Claude was never shown is flagged. API errors (bad key, rate limit, network, unknown model) produce one-line explanations.

**Why it matters:** this is the "G" in RAG, and the only stage that calls an API. Comparing with and without retrieval shows what RAG adds: answers grounded in this document, with sources you can check.

**Module:** `generate.py` · **Commands:**

- `rag-demo ask "your question"`: the answer, with citations.
- `rag-demo ask "your question" --compare`: the RAG answer and a no-context answer side by side.
- `rag-demo ask "your question" --no-rag`: the no-context answer only.

Each question costs well under a cent with the default model.

## Live demo script

The exact sequence for running this in front of an audience. Run `rag-demo ingest` once beforehand so the model is downloaded and the index exists, and widen the terminal to at least 120 columns.

```bash
# 1. Extraction: the raw text, page by page. Point out the page numbers and the empty (image) pages.
rag-demo extract --page 41

# 2. Chunking: how the text is cut up. Point out the highlighted overlap between neighbouring chunks.
rag-demo chunk --strategy fixed
rag-demo chunk --strategy paragraph

# 3. Embedding: what a chunk looks like as numbers.
rag-demo embed --show 541

# 4. Indexing: build and inspect the stored index.
rag-demo ingest
rag-demo index-info

# 5. Retrieval: ranked chunks and similarity scores. No AI model involved yet.
rag-demo search "How long should I brine a turkey?"

# 6. Augmentation: the exact prompt Claude will receive. Nothing is sent.
rag-demo ask "How long should I brine a turkey?" --show-prompt

# 7. Generation: a cited answer, then RAG versus no RAG.
rag-demo ask "How long should I brine a turkey?"
rag-demo ask "Why should butter be kept cold when making pie dough?" --compare

# Bonus: a question the book doesn't answer. Claude should say so rather than guess.
rag-demo ask "What does the book recommend for cooking sushi rice?"
```

`ingest` takes about a minute; to keep the demo moving, run it beforehand and just show `index-info`. Each command that loads the embedding model (`embed`, `ingest`, `search`, `ask`) takes a few seconds to start.

## Evaluation

`eval/questions.json` holds questions written from the PDF, each with the page(s) where the answer is. `rag-demo eval` retrieves the top chunks for each question and reports **hit rate at k**: the share of questions where a chunk from an answer page appears in the top 1, 3 and 5 results. No API calls are made.

```bash
rag-demo eval                                 # the saved index
rag-demo eval --chunk-size 400 --overlap 75   # a temporary index with other settings
rag-demo eval --strategy paragraph
```

Passing any chunk setting builds a temporary index in memory (about a minute), leaving the saved index untouched, so you can see how chunking choices change retrieval. With only a handful of questions, one question moves the hit rate by about 10 percentage points, so look at which questions change rank rather than reading much into the totals.

To use your own PDF, replace the questions. Page numbers are PDF page numbers as shown by `rag-demo extract`, which can differ from the numbers printed on the pages:

```json
{
  "questions": [
    {"question": "Why does the author advise against iodised salt?", "pages": [31], "answer": "It tastes metallic."}
  ]
}
```

## Tests

```bash
pytest
```

The unit tests cover the text cleaning rules, the chunker (overlap length, chunks reassembling into the source text, no empty or oversized chunks, no split words, page tracking), the NumPy index (known nearest neighbours, save and load), and the manifest's refusal of a mismatched embedding model. They use small made-up inputs, so they need neither the PDF nor the embedding model, and run in under a second.

## Configuration

Defaults live in `src/rag_demo/config.py`; most can be overridden per command with the flags shown above.

| Setting | Default | Override |
|---|---|---|
| `CHUNK_STRATEGY` | `fixed` | `--strategy` |
| `CHUNK_SIZE` | 800 characters | `--chunk-size` |
| `CHUNK_OVERLAP` | 150 characters | `--overlap` |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | `EMBED_MODEL=` in `.env` (a model name or a folder path) |
| `TOP_K` | 5 | `-k` / `--top-k` |
| `CLAUDE_MODEL` | `claude-sonnet-5` | `--model` |
| `MAX_TOKENS` | 1024 | `--max-tokens` |

Chunk settings passed to `ingest` are recorded in the manifest. Changing `EMBED_MODEL` requires running `rag-demo ingest` again: the saved index will refuse to load otherwise.

## Project layout

```
rag-demo/
├── data/                   # your PDF goes here (ignored by git)
├── index/                  # generated by ingest: vectors.npy, chunks.json, manifest.json, pages/ cache
├── src/rag_demo/
│   ├── config.py           # settings and the API key
│   ├── extract.py          # 1. PDF -> cleaned text per page
│   ├── chunk.py            # 2. text -> chunks with page numbers
│   ├── embed.py            # 3. chunks -> unit-length vectors
│   ├── index.py            # 4. vectors + chunks -> saved, checked index
│   ├── retrieve.py         # 5. question -> top-k chunks
│   ├── augment.py          # 6. chunks + question -> prompt
│   ├── generate.py         # 7. prompt -> Claude's cited answer
│   ├── evaluate.py         # retrieval hit rate
│   └── cli.py              # one command per stage
├── tests/                  # pytest unit tests
├── eval/questions.json     # evaluation questions for the sample PDF
├── .env.example            # copy to .env and add your key
└── plan.md                 # the build plan
```

## Troubleshooting

| Message | Fix |
|---|---|
| `ANTHROPIC_API_KEY is missing` | Create `.env` from `.env.example` and add your key. |
| `The API rejected the key (401)` | The key in `.env` is wrong or revoked. |
| `Could not load the embedding model` | huggingface.co is unreachable; see [If huggingface.co is blocked](#if-huggingfaceco-is-blocked-on-your-network). |
| `No index found` | Run `rag-demo ingest`. |
| `This index was built with '...'` | The embedding model changed since the index was built. Run `rag-demo ingest` again. |
| `No PDF found` / `Several PDFs` | Put exactly one PDF in `data/`, or pass `--pdf`. |
| Accents and dashes show as `�` | Your terminal isn't using UTF-8. Windows Terminal handles it; on the old console, run `chcp 65001` first. |

## Design decisions

- **No framework.** No LangChain or LlamaIndex: every stage is plain Python you can read in a few minutes.
- **Local embeddings only.** Anthropic doesn't offer an embedding model. A hosted provider (Voyage AI) would need a second API key, so the free local model is used instead.
- **NumPy instead of a vector database.** At this size, searching every chunk takes about 20 ms. A vector database such as Chroma adds approximate search, incremental updates and filtering, which only pay off at a much larger scale, at the cost of many dependencies. `index.py` defines a `VectorIndex` interface, so one could be added later.
- **Out of scope:** OCR for scanned PDFs, multiple documents, a web interface, and hybrid keyword-plus-vector search.
