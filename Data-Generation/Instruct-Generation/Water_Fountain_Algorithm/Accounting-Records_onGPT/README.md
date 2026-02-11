# Accounting Record Questions

This module generates realistic, detailed questions on accounting-related business records in German tax law, and answers them using external sources and a hybrid semantic search and LLM workflow. It is especially suited for producing training data for LLMs in legal-tax domains.

---

## Features

- Prompt-based question generation using curated accounting scenarios
- Embedding-based semantic retrieval of supporting information (via SearXNG)
- Source-aware answer generation using Azure OpenAI (GPT)
- Automatic search query rewriting for failed lookups
- Source scraping for HTML, XML, and PDF content
- Token-aware text selection and truncation
- Continuous generation loop with result deduplication and saving

---

## Usage

Run this script directly:

```bash
python generator_AccountingRecords.py
```

Make sure the following services are running:

- `embedding_server.py` on `http://localhost:8000/embed`
- `tokenization_server.py` on `http://localhost:8001/count_tokens`

> **Note**: You also need a running local SearXNG instance (optimally on `http://localhost:8080`). For instructions and documentation, see https://github.com/searxng/searxng or https://docs.searxng.org/. 

And you filled out the AzureAPI Requirements:

- `API Key`
- `API Endpoint`
- `API Version`
- `API Deployment ID`

And you have `Ollama` running and adjusted the hardcoded model (Line 453) to a suitable.
---

## Output Format

The script will append QA pairs to a file named, but can also be adjusted within the script:

```
accounting_records.jsonl
```

Each line is a JSON object:

```json
{
  "frage": "...",
  "antwort": "..."
}
```

---

## Prompt Logic

- Prompts are randomly sampled from a large set of accounting scenarios.
- Each prompt includes a realistic business context (e.g. leasing, IAB, Rückstellungen).
- Generated questions must be long, indirect, and contain no direct address (no "Du", "Hallo" etc.).
- Questions aim for high diversity and domain relevance.

---

## Search + Retrieval Pipeline

1. Search query is generated from question using `generate_new_query()`
2. Top SearXNG results are scraped and split into chunks
3. Embedding similarity is used to rank chunks
4. Best-ranked passages are token-counted and truncated if needed
5. Selected context is sent to Azure OpenAI with the question for answering

---

## Requirements

Python 3.9 or newer. You can install the dependencies via (untested):

```bash
pip install -r requirements.txt
```

Or manually (untested):

```bash
pip install openai ollama PyPDF2 beautifulsoup4 numpy scikit-learn nltk requests
```

Ensure nltk punkt tokenizer is downloaded:

```python
import nltk
nltk.download('punkt')
```

---

## Notes

- Azure OpenAI credentials must be configured via:
  - `AZURE_API_KEY`
  - `AZURE_ENDPOINT`
  - `AZURE_API_VERSION`
  - `AZURE_DEPLOYMENT_ID`

- LLM model used for generation: `llama3:8b-instruct-q8_0`

