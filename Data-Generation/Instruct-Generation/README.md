# Water Fountain Algorithm – German Tax Law QA System

This repository provides a modular, extensible, and high-throughput pipeline for the generation of instruction-style question-answer (QA) pairs rooted in German tax legislation. The system is designed to support the automated construction of training corpora for fine-tuning large language models (LLMs) in specialized legal domains.

The architecture supports heterogeneous input formats (e.g., legislative paragraphs, web-sourced snippets, and commentaries) and integrates multiple generation strategies (e.g., context-conditioned prompting, diversification logic, and commentary elicitation). The pipeline enables fine-grained control over input-output (I/O) flows and distributed execution via both multiprocessing and threading.

![SteuerLLM Pipeline](../../figures/steuerLLM_figure.png)

> **Note:** Due to the efficiency-driven development of this project, some components may lack extensive documentation or general-purpose abstraction. The scripts are optimized for throughput rather than modularity. 

---

## Repository Structure

### `Water_Fountain_Algorithm/`
Contains all logic related to QA generation. Subdivided into:

- **`Accounting-Records/`**  
  Thread-based routines for QA generation from accounting and booking scenarios (*Buchungssätze*). Focuses on practical business transactions and domain-specific terminology. Uses GPT-4o with AzureOpenAI.

- **`Cleansing/`**  
  Scripts for post-processing and consolidation of intermediate QA outputs. Includes deduplication, context extraction, and conversion to JSONL. See the subdirectory README for step-wise instructions.

- **`Process-Based/`**  
  Multiprocessing-enabled module tailored for large-scale generation. Enables batch-level distribution of legislative texts across multiple endpoints.

- **`Thread-Based/`**  
  Lightweight alternative to the process-based pipeline using `ThreadPoolExecutor`. Designed for use cases with pre-retrieved or chunked contextual data.


---

### `Data/`
Raw and preprocessed legal sources, including Tax laws (e.g., EStG, AO) in machine-readable formats.

---

### `Server/`
Services required for local embedding computation, token counting, and semantic chunk filtering. Communicate via REST APIs and are used by processors to maintain efficiency and reproducibility.

---

### `Utils/`
Assorted utility scripts used throughout the project for ad-hoc data manipulation, formatting, and server checking. These scripts are not documented individually and were created for internal purposes only.

---

## External Requirements

To enable retrieval-based contextualization, a **local instance of SearXNG** must be hosted and accessible at `http://localhost:8080`

SearXNG is used to retrieve supplementary legal or web content based on search queries related to legal articles or tax law terminology. The retrieved snippets are chunked, embedded, and semantically ranked for integration into the prompting context.

For instructions and documentation, see https://github.com/searxng/searxng or https://docs.searxng.org/. 

> If SearXNG is not available or misconfigured, the fallback will result in empty or degraded context, significantly affecting QA quality.

---

## Workflow Summary

1 **Preparation of the input data**
   The basis for the generation can be legal primary sources (e.g. EStG, AO), stored in the directory `Data/`. Or, additional seed questions can be integrated in order to specifically control thematic diversity and semantic depth during generation.

2 **Execution of the Water Fountain Algorithm**
   Depending on the objective, one or more of the following modular pipelines can be executed:
   - `Thread-Based`: context- or article-based
   - `Process-Based`: Parallelised via multiprocessing
   - `Accounting-Records`: accounting cases and scenarios

   These components generate initial question-answer pairs based on the respective input.

3 **Infrastructure requirements**
   Operation requires the following local services:
   - A local instance of **SearXNG** on port `8080` for semantic web search and context enrichment
   - A **Tokenization server** on port `8001` for token-based segmentation and context truncation
   - An **Embedding server** on port `8000` for similarity assessment of text passages
   - At least one active HTTP-exposed LLM endpoint for question generation

4. **Data cleansing and consolidation**
   Multi-stage cleansing takes place after generation:
   - Removal of incorrect or irrelevant QA pairs
   - Semantic deduplication
   - Consolidation of all subsets into a standardised format

   The result is a `JSONL` file with cleansed and structured question-answer pairs.


## Final Output Format

After cleansing, the training dataset is stored in a conversational JSONL format (`convo_qa.jsonl`), where each line represents a question-answer pair encoded as a two-turn dialogue:

```json
{
  "conversations": [
    {"role": "user", "content": "Was versteht man unter einer verdeckten Gewinnausschüttung nach § 8 KStG?"},
    {"role": "assistant", "content": "Eine verdeckte Gewinnausschüttung liegt vor, wenn..."}
  ]
}
