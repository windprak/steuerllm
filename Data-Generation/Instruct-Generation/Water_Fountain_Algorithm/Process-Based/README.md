# Water Fountain Algorithm: Large-Scale Question Answering and Retrieval Pipeline for German Tax Law

This repository provides a modular pipeline for generating high-quality legal questions and answers on German tax law using a hybrid search mechanism. It orchestrates semantic web search (SearXNG), embedding similarity, rule-based scoring logic, and large language models (LLMs). It is optimised for high-throughput processing using parallel batch job execution. The pipeline employs **process-based parallelisation** via the `multiprocessing` module, allowing jobs to run concurrently and independently across multiple CPU cores. The maximum started parallel jobs with the used setup were 59.

---

## Overview

The system architecture consists of the following components:

1. **`batch_manager.py`**
   - Master controller that coordinates execution of parallel job configurations from a `batch_config.json` file.

2. **`hybrid_search_*.py` (e.g. `Standard`, `EstG`, `Diversity`)**
   - The core algorithm (Water Fountain Algorithm) that handles retrieval, ranking, and answer generation per job.
   - Fetches and chunks search results, performs similarity matching, calls LLM APIs, and outputs question-answer pairs with traceable context.

3. **Embedding and Token Counting Servers** (external, but required):
   - `/embed` (port 8000): Embedding server powered by default by `intfloat/multilingual-e5-large`
   - `/tokenize`, `/count_tokens` (port 8001): Tokenization server using by default `Mistral-Large-Instruct`

4. **SearXNG Instance** (external, but required):
   - Required for retrieving web search results.
   - Must be running locally (optimally on `http://localhost:8080`)
   - For instructions and documentation, see https://github.com/searxng/searxng or https://docs.searxng.org/ 
---

## Directory Structure

```text
.
├── batch_config.json            # Configuration file for parallel jobs
├── batch_manager.py             # Main orchestrator script
├── hybrid_search_Standard.py    # Core algorithm (alternatives: _EstG.py, _Diversity.py)
├── tokenization_server.py       # Token counting & truncation microservice (port 8001)
├── embedding_server.py          # Embedding microservice (port 8000)
├── input/                       # Contains question batches
│   └── *.json                   # Input question files
├── output/                      # Stores model responses
│   └── *.json                   # Output question-answer files
```

---

## Configuration (`batch_config.json`)

Each job is defined with dedicated input/output files, endpoints, model names, and runtime settings. Example:

```json
{
  "global_settings": {
    "data_dir": "",
    "max_concurrent_jobs": "59",
    "script": "hybrid_search_Standard.py"
  },
  "jobs": [
    {
      "job_id": "job1",
      "enabled": true,
      "fau_endpoint": "http://<your-fau-endpoint>",
      "searxng_endpoint": "not working currently, use environmental variable 'SEARXNG_ENDPOINT' or use standard Port 8080",
      "input_file": "input/questions_batch1.json",
      "output_file": "results/qa_results_batch1.json",
      "embeddings_cache": "",
      "max_output_file_size_gb": "7",
      "model": "gpt-4-turbo",
      "flush_settings": {
        "qa_results_interval_minutes": "15",
        "questions_interval_minutes": "60"
      },
      "runtime_settings": {
        "duration_hours": "4",
        "sleep_after_question_seconds": "1.5",
        "sleep_after_generation_seconds": "3"
      },
      "resource_limits": {
        "max_memory_gb": "16",
        "max_cpu_percent": "90"
      }
    }
  ]
}
```

---

## Execution

### Step 1: Start API services

```bash
# Token counter server
python tokenization_server.py

# Embedding server
python embedding_server.py
```

These must be accessible at (or change the hardcoded variable):
- `http://localhost:8000/embed`
- `http://localhost:8001/tokenize`
- `http://localhost:8001/count_tokens`

> **Note**: You also need a running local SearXNG instance (optimally on `http://localhost:8080`). For instructions and documentation, see https://github.com/searxng/searxng or https://docs.searxng.org/. 

### Step 2: Run batch manager

```bash
python batch_manager.py batch_config.json
```

Each job defined in the config will be executed concurrently.

---

## Requirements

Python 3.9 or newer. Recommended packages:

```bash
pip install -r requirements.txt
```

Or manually (untested):

```bash
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 transformers==4.46.3 tokenizers==0.20.3 scikit-learn nltk requests PyPDF2 beautifulsoup4 --index-url https://download.pytorch.org/whl/cu124
```

---

## Key Features

- **Modular Search + Reranking:** Uses embedding similarity to score scraped search results.
- **Adaptive Context Truncation:** Dynamically truncates text to fit LLM token limits using a dedicated tokenization API.
- **Query Regeneration:** Automatically improves search queries if no useful result is found.
- **Robust Question Generation:** Optional generation of additional questions based on prior answers using curated prompts.
- **Job Isolation:** Each job can have individual resource, endpoint, and timing settings.
- **Resilience:** All external calls implement exponential backoff and retry mechanisms.

---

## Intended Use

The Water Fountain Algorithm is optimised for:

- Generating realistic, diverse legal questions
- Context-based answer generation using web-retrieved documents
- High-volume, parallel processing of tax-related input questions
- Creating traceable and reproducible QA data for benchmarking or fine-tuning

> **Note:** The algorithm does not rely on internal model knowledge. It is designed to simulate real-world legal workflows using search and source-backed context.

---

## Input

Each job consumes a JSON input file located in the input/ directory. The input consists of a list of questions with optional metadata. Each entry must follow the structure:
```json
[
  {
    "frage": "Welche steuerlichen Auswirkungen hat eine verdeckte Gewinnausschüttung bei einer GmbH?",
    "typ": "Verständnis- bzw. Kommentarfragen",
    "quelle": ""
  },
  {
    "frage": "Ein Unternehmen kauft eine Maschine für 100.000 €. Wie lautet der Buchungssatz?",
    "typ": "Buchungssätze / Bilanzkonten",
    "quelle": ""
  }
]
```
- `frage` is the actual question (mandatory).

- `typ` defines the question type (optional).

- `quelle` is optional and currently unused. It is a legacy field and can remain empty ("").

---

## Output

Each job will produce a file (or multiple `*_vN.json` files) in the `results/` directory, containing entries of the following structure:

```json
{
  "question": "...",
  "answer": "...",
  "sources": [
    {"text": "...", "score": 0.84, "source_info": {"url": "...", "title": "..."}},
    ...
  ]
}
```

This output can be directly reused for training, evaluation, or quality auditing.

