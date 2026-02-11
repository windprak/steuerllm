# Migration Guide

This document explains the changes made to clean up and organize the codebase for publication.

## Summary of Changes

The codebase has been reorganized from a collection of experimental scripts with hardcoded values into a production-ready, configurable pipeline.

## File Mapping: Old → New

### Core Worker Scripts
- `worker_json.py` → `src/worker.py` (cleaned, configurable)
- `worker_json_skip.py` → `src/worker.py` (resume logic integrated)
- `worker_json_tokenizer.py` → `src/worker.py` (consolidated)
- `worker_json_cases.py` → `src/worker.py` (consolidated)
- `worker.py` (old version) → **Removed**
- `worker_json copy.py` → **Removed** (duplicate)

### Orchestrator/Submitter Scripts
- `autosubmitter_balance.py` → `src/orchestrator.py` (cleaned, configurable)
- `autosubmitter.py` → **Removed** (older version)
- `autosubmitter2.py` → **Removed** (older version)
- `autosubmitter3.py` → **Removed** (older version)
- `autosubmitter4.py` → **Removed** (older version)
- `autosubmitter_balance_cases.py` → **Removed** (older version)
- `aquasub.py` → **Removed** (experimental)

### Data Processing Scripts
- `json_chunker.py` → `src/data_chunker.py` (cleaned, configurable)
- `enrich_exported_data_final.py` → `src/data_enricher.py` (cleaned, configurable)
- `enrich_exported_data.py` → **Removed** (older version)
- `enrich_exported_data_v2.py` → **Removed** (older version)
- `enrich_exported_data_v3.py` → **Removed** (older version)
- `enrich_exported_data_memory_safe.py` → **Removed** (older version)

### Utility Scripts
- `process_conversations.py` → **Removed** (one-off script)
- `process_conversations_new.py` → **Removed** (one-off script)
- `process_llmbench.py` → **Removed** (one-off script)
- `match_and_format.py` → **Removed** (one-off script)
- `fixer.py` → **Removed** (one-off script)
- `parse_logs.py` → **Removed** (one-off script)
- `example.py` → **Removed** (test script)
- `export.py` → **Removed** (empty file)

### Data Files
- `chunk1.json` → **Removed** (test data)
- `out.jsonl` → **Removed** (output data)
- `output_01.jsonl` → **Removed** (output data)
- `results.txt` → **Removed** (test results)
- `explore.ipynb` → **Removed** (exploratory notebook)

## Major Changes

### 1. Removed Hardcoded Values

**Old (worker_json.py):**
```python
base_url = None
headers = {'content-type': 'application/json', 'Authorization': 'Bearer xFhGltj52Gn'}
API_KEY = 'xFhGltj52Gn'
MODEL = "/anvme/workspace/unrz103h-helma/base_models/MistralAWQ"
```

**New (src/worker.py + config/config.yaml):**
```yaml
api:
  base_url: "http://localhost:6000"
  api_key: "your-api-key-here"
  model: "your-model-name"
```

### 2. Removed Hardcoded Paths

**Old (enrich_exported_data_final.py):**
```python
SOURCE_DIR = Path("/Users/sebastian/PycharmProjects/q&a/q&a/data/fixed_4gb")
EXPORT_DIR = Path("/Users/sebastian/PycharmProjects/q&a/q&a/exportdata")
OUTPUT_DIR = Path("/Users/sebastian/PycharmProjects/q&a/q&a/data/datajsonlexport")
```

**New (src/data_enricher.py):**
```bash
python src/data_enricher.py \
  --source-dir data/original_source \
  --conversation-dir data/output \
  --output-file data/final/enriched_conversations.jsonl
```

### 3. Removed IP Address Hardcoding

**Old (autosubmitter_balance.py):**
```python
pattern = r'IP Address: (10\.28\.89\.\d+)'
with open('results.txt', 'r') as f:
    results = f.read()
ip_addresses = parse_ip_addresses(results)
```

**New (src/orchestrator.py + config/config.yaml):**
```yaml
orchestrator:
  enabled: true
  endpoints:
    - "192.168.1.10"
    - "192.168.1.11"
```

### 4. Unified Configuration

All configuration is now centralized in `config/config.yaml`:
- API settings (endpoint, key, model)
- Worker parameters (concurrency, temperature)
- Data paths (input/output directories)
- Orchestrator settings (endpoints, workers)
- Logging configuration

### 5. New Directory Structure

```
Old (flat):                    New (organized):
├── worker_json.py            ├── src/
├── autosubmitter.py          │   ├── worker.py
├── json_chunker.py           │   ├── orchestrator.py
├── enrich_*.py               │   ├── data_chunker.py
├── chunk1.json               │   └── data_enricher.py
├── out.jsonl                 ├── config/
└── results.txt               │   └── config.yaml
                              ├── data/
                              │   ├── input/
                              │   ├── output/
                              │   └── final/
                              ├── examples/
                              │   └── sample_input.json
                              ├── logs/
                              ├── requirements.txt
                              └── README.md
```

## How to Migrate Your Workflow

### Old Workflow

```bash
# 1. Chunk data
python json_chunker.py

# 2. Run workers (hardcoded IPs in results.txt)
python autosubmitter_balance.py

# 3. Enrich output (hardcoded paths)
python enrich_exported_data_final.py
```

### New Workflow

```bash
# 1. Configure once
cp config/config.yaml config/config.yaml
# Edit config.yaml with your settings

# 2. Chunk data
python src/data_chunker.py \
  --input-dir data/input \
  --output-dir data/chunks

# 3. Generate conversations
# Option A: Single worker
python src/worker.py \
  --input-files data/chunks/chunk1.json \
  --output-dir data/output

# Option B: Distributed (configure endpoints in config.yaml first)
python src/orchestrator.py

# 4. Enrich output
python src/data_enricher.py \
  --source-dir data/source \
  --conversation-dir data/output \
  --output-file data/final/result.jsonl
```

## Benefits of New Structure

1. **No hardcoded secrets** - API keys are in config files (gitignored)
2. **Portable** - Works on any machine without path changes
3. **Configurable** - All settings in one place
4. **Resumable** - Automatic skip of processed records
5. **Documented** - Clear README with examples
6. **Maintainable** - Single source of truth for each component
7. **Production-ready** - Proper error handling and logging

## Configuration Tips

### For Local Development
```yaml
api:
  base_url: "http://localhost:6000"
orchestrator:
  enabled: false
```

### For Distributed Processing
```yaml
orchestrator:
  enabled: true
  endpoints: ["gpu-node-1", "gpu-node-2", "gpu-node-3"]
  workers_per_endpoint: 4
```

### For Large Scale Processing
```yaml
worker:
  max_concurrent: 20
data:
  chunk_size_mb: 500
  skip_processed: true
```

## Questions?

Refer to the main README.md for detailed usage instructions.
