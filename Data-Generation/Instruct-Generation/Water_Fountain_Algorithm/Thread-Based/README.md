# IO-Based QA Generator: Contextual Question Answering from Legislative Texts

This repository provides a modular pipeline for generating high-quality legal questions and answers on German tax law using multiple interchangeable processor strategies and input/output pipelines. It supports context-aware generation from legislative paragraphs using batch job execution, semantic retrieval, and multi-endpoint LLM orchestration. Compared to the other pipeline, it features more I/O-intensive workflows and thus adapts both process-based and thread-based parallelism depending on the processor configuration. The maximum started parallel jobs for the used setup were 300.

---

## Overview

The core components of the pipeline include:

### 1. Batch Execution System

- **`batch_manager_IO.py`**
  - Central controller that loads configuration from `batch_config.json` and launches the appropriate processor.
  - Handles shutdowns, retries, job monitoring, and logging.

### 2. Processor Modules

These modules implement different strategies for legal question generation:

- **`processor_Article.py`**
  - Extracts questions from legal paragraphs using additional retrieved context via SearXNG.
  - Generates one or more questions per paragraph, with structured and ranked supporting text snippets.
  - Good for generating realistic real-world QA pairs based on internet-style context.

- **`processor_Chunks.py`**
  - Processes existing QA source chunks from prior hybrid search pipelines (e.g., from `output/hybrid_search`).
  - Each source chunk is processed `reps` times to maximise question diversity.
  - Optimized for preprocessed datasets where retrieval is already done.

- **`processor_Commentary.py`**
  - Designed to generate interpretative questions focused on understanding legal norms ("Kommentarfragen").
  - Focuses more on detailed context and comprehension of legal ideas than procedural steps.

Each processor:
- Supports parallelized execution via `multiprocessing` or `ThreadPoolExecutor`
- Loads articles from JSON or JSONL
- Flushes QA results to disk incrementally
- Balances requests across multiple LLM endpoints

### 3. Additional (external) services

- **Embedding and Token Counting Servers** (external, but required):
   - `/embed` (port 8000): Embedding server powered by default by `intfloat/multilingual-e5-large`
   - `/tokenize`, `/count_tokens` (port 8001): Tokenization server using by default `Mistral-Large-Instruct`

- **SearXNG Instance** (external, but required):
   - Required for retrieving web search results.
   - Must be running locally (optimally on `http://localhost:8080`)
   - For instructions and documentation, see https://github.com/searxng/searxng or https://docs.searxng.org/ 

---

## Directory Structure

```text
.
├── batch_config.json             # Main configuration for IO jobs
├── batch_manager_IO.py           # Orchestrates execution of IO processors
├── processor_Article.py          # Context-aware generation from raw articles
├── processor_Chunks.py           # Generates multiple QAs from chunked sources
├── processor_Commentary.py       # Produces commentary-style understanding questions
├── input/                        # Input articles or old tuples (JSONL / JSON)
│   └── *.json                    # Input law or chunk files
├── output/                       # Output QA results and logs
│   └── *.json                    # Output question-answer files
```

---

## Requirements

Install all dependencies using (untested):

```bash
pip install -r requirements.txt
```

Or manually (untested):

```bash
pip install requests numpy beautifulsoup4 PyPDF2>=3.0.1 nltk torch tqdm
```

---

## Execution Steps

### 1. Start Required API Services

```bash
python tokenization_server.py
python embedding_server.py
```

These must be accessible at (or change the hardcoded variable):
- `http://localhost:8000/embed`
- `http://localhost:8001/tokenize`
- `http://localhost:8001/count_tokens`

> **Note**: Ensure your SearXNG instance is available (optimally at `http://localhost:8080`) if `processor_Article.py` or `processor_Commentary.py` is used. For instructions and documentation, see https://github.com/searxng/searxng or https://docs.searxng.org/. 

### 2. Define Your Configuration

Configure your job in `batch_config.json`. Example:

```json
{
  "input_file": "./input/articles",
  "output_file": "./output/qa_results.jsonl",
  "endpoints": [
    "http://10.0.0.1:6000/v1/chat/completions",
    "http://10.0.0.2:6000/v1/chat/completions"
  ],
  "num_jobs": 10,
  "model": "/path/to/model",
  "amount_questions": 1,
  "flush_interval": 180,
  "reps": 2
}
```

### 3. Run the Batch Manager

```bash
python batch_manager_IO.py batch_config.json
```

This will automatically launch one of the processors depending on the input path and configuration.

---

## Input / Output Format

### Input (`.json` or `.jsonl`)

Each article must follow this structure:

```json
{
  "title_main": "§ 14 AO",
  "title": "Vertreter",
  "text": "Vertreter ist, wer im Namen eines anderen handelt...",
  "num": "§ 14",
  "id": "AO_14"
}
```

Or, for chunk-based generation (`processor_Chunks.py`):

```json
{
  "question": "...",
  "answer": "...",
  "sources": [
    {"text": "...", "score": 0.84, "source_info": {"url": "...", "title": "..."}},
    {"text": "...", "score": 0.75, "source_info": {"url": "...", "title": "..."}}
  ]
}
```

### Output (JSONL)

```json
{
  "article": {
    "id": "AO_14",
    "title_main": "§ 14 AO",
    "text": "..."
  },
  "question": "Was bedeutet der Begriff 'Vertreter' im Sinne des § 14 AO im Zusammenhang mit der Körperschaftsteuer?",
  "answer": "...",
  "context": [ ... ]
}
```

Or for chunk-based generation (`processor_Chunks.py`):

```json
{
  "question": "Wie wirkt sich die Regelung des § 6 EStG auf die Abschreibung von Wirtschaftsgütern aus?",
  "answer": "...",
  "id": "tuple_id_xyz"
}
```

---

## Processor Comparison

| Processor              | Input Type        | Retrieval | Parallelism      | Use Case                                                  |
|------------------------|-------------------|-----------|------------------|-----------------------------------------------------------|
| `processor_Article.py`  | JSONL Articles     | Yes       | Multiprocessing  | Rich QA pairs with retrieved external context             |
| `processor_Chunks.py`   | JSON Chunks        | No        | ThreadPool       | QA generation from pre-chunked hybrid search results      |
| `processor_Commentary.py` | JSONL Articles     | Yes       | Multiprocessing  | Commentary-style questions focusing on legal interpretation |

### Notes on Switching Processors

- All processors can use the same endpoints and output format.
- Only `processor_Chunks.py` expects source-chunked data (with `sources` field).
- `Article`-based processors (`Article`, `Commentary`) expect metadata and full legal texts (see Folder `Data`).
- Retrieval servers (SearXNG, token counter) are required for `Article` and `Commentary`, but not for `Chunks`.

---

## Intended Use

This pipeline is optimized for:

- Generating legally relevant, realistic questions
- Building datasets for legal QA fine-tuning
- Creating interpretable, context-rich QA examples from real legal texts
- Scaling across distributed models and multi-GPU systems

> **Note:** All questions are context-based. Generation does not rely on internal model knowledge or hallucination.

