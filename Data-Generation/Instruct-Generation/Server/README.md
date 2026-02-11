# Tokenization and Embedding API Servers using Hugging Face Transformers

This repository provides two standalone FastAPI-based HTTP services:

1. A **tokenization server** for encoding text using a specified tokenizer and returning either tokenized strings or token counts.
2. An **embedding server** for generating normalized sentence embeddings using a multilingual transformer model.

Both services are designed to be used as modular preprocessing components in larger LLM-based systems.

---

## Overview

### Tokenization Server (`tokenization_server.py`)

- Uses a Hugging Face tokenizer (default: `TechxGenus/Mistral-Large-Instruct-2407-AWQ`, for other models change the hardcoded variable in the script)
- Supports:
  - Truncation with optional `max_length`
  - Batch processing
  - Clean re-decoding of tokenized outputs
  - Token count calculation

Endpoints:

- `POST /tokenize`: Returns decoded tokenized version of texts
- `POST /count_tokens`: Returns number of tokens in a single input string

### Embedding Server (`embedding_server.py`)

- Uses `intfloat/multilingual-e5-large` (for other models change the hardcoded variable in the script) for sentence-level embeddings
- Embeddings are:
  - Generated via average pooling
  - L2-normalized
  - Returned as explicit float lists

Endpoint:

- `POST /embed`: Accepts a list of texts and returns their embeddings

---

## Execution

To run both servers locally, you can either use the `python` command directly (which uses the default ports) or start them manually via `uvicorn` with custom ports.

### Tokenization Server

- **Option 1:** Run with default port `8001`

```bash
python tokenization_server.py
```
APIs will be accessible at `http://0.0.0.0:<PORT>`.

- **Option 2:** Run with custom port using `uvicorn`

```bash
uvicorn tokenization_server:app --port 8081
```
APIs will be accessible at `http://127.0.0.1:<PORT>`.

### Embedding Server

- **Option 1:** Run with default port `8000`

```bash
python embedding_server.py
```
APIs will be accessible at `http://0.0.0.0:<PORT>`.

- **Option 2:** Run with custom port using `uvicorn`

```bash
uvicorn embedding_server:app --port 8080
```
APIs will be accessible at `http://127.0.0.1:<PORT>`.

> **Note 1:** Option 1 should be preferred.

>**Note 2:** When using `uvicorn` directly, make sure the Python file is in your working directory and ends with `:app` to expose the FastAPI application correctly.

---

## Example Usage

### Tokenization Request

```http
POST /tokenize
Content-Type: application/json

{
  "texts": ["Beispieltext für Tokenisierung."],
  "truncation": true,
  "max_length": 20
}
```

### Token Count Request

```http
POST /count_tokens
Content-Type: application/json

{
  "text": "Kurzer Beispielsatz."
}
```

### Embedding Request

```http
POST /embed
Content-Type: application/json

{
  "texts": ["This is an example sentence.", "Das ist ein Beispielsatz."]
}
```

---

## Requirements

Python 3.9 or higher is recommended.

Required Python packages:

- `fastapi`
- `uvicorn`
- `pydantic`
- `transformers`
- `torch`
- `numpy`
- `argparse`
- `socket`

To install all required dependencies, use (untested):

```bash
pip install fastapi uvicorn pydantic transformers torch numpy argparse socket
```

---

## Notes

- Both servers are independent and can be run in parallel.
- The tokenization server uses a causal language model tokenizer; adjust `MODEL_NAME` if needed.
- The embedding server supports GPU acceleration via PyTorch if available.
- Input is always assumed to be UTF-8 encoded plain text.

---
