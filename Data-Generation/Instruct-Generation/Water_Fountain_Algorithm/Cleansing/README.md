# Final Data Cleansing & Consolidation Pipeline

This directory contains all scripts used to perform the final data cleaning, filtering, and formatting of the question-answer pairs generated throughout the work. The goal is to consolidate various output formats into one clean and consistent dataset for training purposes.

> **Note:** This pipeline was developed with a strong focus on efficiency and quick integration rather than optimal code structure or reusability. As such, it is somewhat fragmented and not designed for general-purpose use. For questions or clarification, please contact: **laurin.schmid@fau.de**

---

## Overview of the Processing Pipeline

The final dataset is created by sequentially executing the following four scripts:

### 1. `context_retriever.py`
- **Purpose**:
  - Processes QA outputs from diversity-based algorithms.
  - Filters QA pairs based on source quality (minimum 3 SearXNG chunks).
- **Output directory**: `.../final/`
- **Outputs**: 
    - A list of QA pairs (`qaComm_training_data_part_X.json`)
    - A corresponding list of all chunk passages per question (`qaComm_chunk_data_part_X.json`)

### 2. `retrieve_context_Comm.py`
- **Purpose**:
  - Processes QA pairs from the Buchungssatz generator.
  - Filters faulty answers and saves valid QAs + their source context.
- **Output directory**: `.../final/`
- **Outputs**:
  - `qaAccountingRecords_training_data.json`
  - `qaAccountingRecords_chunk_data.json`

### 3. `combine_all_qa.py`
- **Purpose**:
  - Merges all QA datasets from multiple processors and stages.
  - Removes duplicate questions (based on normalized strings).
  - Filters out predefined seed questions and low-quality entries (e.g. generic error answers).
- **Inputs**:
  - A list of JSON files from earlier QA pipelines (hardcoded paths)
  - Path to seed questions (hardcoded path)
- **Output**:
  - `final_qa.json`

### 4. `convert_to_convo.py`
- **Purpose**:
  - Converts the merged QA list into a conversation-style JSONL file suitable for chat-based LLM finetuning.
  - Each line contains a user-assistant exchange in the `conversations` format.
- **Output**:
  - `convo_qa.jsonl`

---

## Important Notes

- All scripts rely on fixed input/output paths that are **hardcoded**. Adjust them before use if working in a different environment.
- The scripts assume that previous outputs exist and are properly structured.
- Output datasets are saved in either `.json` or `.jsonl` format depending on their intended use.
- Within each step there is often a hardcoded `ID`, which needs to be set in order to have an unique identifier.

---

## Execution Order
Run the scripts in the following order:

```bash
python context_retriever.py
python retrieve_context_Comm.py
python combine_all_qa.py
python convert_to_convo.py
```

The result is a clean, deduplicated, and structured QA dataset at:
```text
Models/content/Training_Data/QAs/convo_qa.jsonl
```

---

## Structure of `convo_qa.jsonl`
The final dataset is saved in JSONL format with one conversation per line. Each line has the following structure:

```json
{
  "conversations": [
    {"role": "user", "content": "<question>"},
    {"role": "assistant", "content": "<answer>"}
  ]
}
```

- Each QA pair is represented as a single-turn dialogue between a user and an assistant.
- All lines are valid JSON and can be loaded using typical JSONL parsers.
- The dataset is ready for training dialogue-based language models.

---

## Contact
For questions or troubleshooting, please reach out to:
laurin.schmid@fau.de