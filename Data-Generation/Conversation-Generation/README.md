# Conversational Dataset Generation Pipeline

A production-ready pipeline for generating instructional conversational datasets from structured data using Large Language Models (LLMs). Originally designed for generating tax law consultations but adaptable to any domain.

## 📋 Overview

This pipeline takes structured JSON data and generates realistic multi-turn conversations between a user and an AI assistant. The system is designed to:

- Process large JSON datasets efficiently through chunking
- Generate natural, domain-specific conversations via LLM APIs
- Support distributed processing across multiple GPU endpoints
- Handle resume/retry logic for robust production use
- Enrich outputs with original metadata

## 🏗️ Architecture

```
Input JSON → Data Chunker → Worker(s) → Generated Conversations → Data Enricher → Final Dataset
```

### Components

1. **Data Chunker** (`src/data_chunker.py`) - Splits large JSON files into manageable chunks
2. **Worker** (`src/worker.py`) - Generates conversations by calling LLM API
3. **Orchestrator** (`src/orchestrator.py`) - Distributes work across multiple endpoints (optional)
4. **Data Enricher** (`src/data_enricher.py`) - Merges conversations with original metadata

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Access to an OpenAI-compatible LLM API endpoint
- Input data in JSON format with required fields

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Conversation-Generation

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy the example configuration:
```bash
cp config/config.yaml config/config.yaml
```

2. Edit `config/config.yaml` with your settings:
```yaml
api:
  base_url: "http://your-llm-endpoint:6000"
  api_key: "your-api-key"
  model: "your-model-name"
```

## 📊 Usage

### Step 1: Prepare Your Data

Your input JSON files should be arrays of objects with these fields:

**Required:**
- `_recordid`: Unique identifier for each record
- `Volltext`: Full text content to generate conversations from

**Optional (metadata):**
- `Themengebiet`: Topic area
- `TopicFilter`: Topic filter/category
- `Titel`: Title
- `Autor`: Author
- `text`: Additional text

Example:
```json
[
  {
    "_recordid": "rec_001",
    "Volltext": "Your content here...",
    "Titel": "Example Title",
    "Themengebiet": "Tax Law"
  }
]
```

### Step 2: Chunk Large Files (Optional)

If you have very large JSON files (>1GB), split them into chunks:

```bash
python src/data_chunker.py \
  --input-dir data/input \
  --output-dir data/chunks \
  --chunk-size-mb 200
```

### Step 3: Generate Conversations

**Single Worker (Local):**
```bash
python src/worker.py \
  --input-files data/chunks/chunk1.json,data/chunks/chunk2.json \
  --output-dir data/output \
  --config config/config.yaml
```

**Distributed Processing:**

First, configure multiple endpoints in `config/config.yaml`:
```yaml
orchestrator:
  enabled: true
  endpoints:
    - "192.168.1.10"
    - "192.168.1.11"
  workers_per_endpoint: 3
  chunks_per_worker: 3
```

Then run:
```bash
python src/orchestrator.py --config config/config.yaml
```

### Step 4: Enrich Output (Optional)

Merge generated conversations with original source metadata:

```bash
python src/data_enricher.py \
  --source-dir data/original_source \
  --conversation-dir data/output \
  --output-file data/final/enriched_conversations.jsonl
```

## 📁 Directory Structure

```
.
├── config/
│   ├── config.yaml           # Main configuration file
│   └── config.example.yaml   # Example configuration
├── src/
│   ├── worker.py            # Core conversation generation worker
│   ├── data_chunker.py      # Splits large JSON files
│   ├── orchestrator.py      # Distributed work manager
│   └── data_enricher.py     # Merges metadata with conversations
├── data/
│   ├── input/               # Place your input JSON files here
│   ├── output/              # Generated conversations output here
│   └── final/               # Final enriched datasets
├── examples/
│   └── sample_input.json    # Example input format
├── logs/                     # Worker logs (created automatically)
├── requirements.txt
└── README.md
```

## ⚙️ Configuration Reference

### API Settings
- `base_url`: LLM API endpoint (without `/v1/`)
- `api_key`: Authentication key
- `model`: Model name or path
- `timeout`: Request timeout in seconds (null = no timeout)
- `max_retries`: Number of retry attempts for failed requests

### Worker Settings
- `max_concurrent`: Number of simultaneous API requests
- `temperature`: LLM temperature parameter (0.0-1.0)
- `top_p`: LLM top-p sampling parameter

### Data Settings
- `input_dir`: Input data directory
- `output_dir`: Output directory for conversations
- `chunk_size_mb`: Size of chunks in MB
- `skip_processed`: Skip already processed records (resume support)

### Orchestrator Settings
- `enabled`: Enable distributed processing
- `endpoints`: List of worker endpoint IPs/hostnames
- `workers_per_endpoint`: Number of workers per endpoint
- `chunks_per_worker`: Chunks assigned to each worker

## 🔄 Resume Functionality

The worker automatically tracks processed records and skips them on restart. To resume interrupted processing:

```bash
# Simply run the same command again
python src/worker.py \
  --input-files data/chunks/chunk1.json \
  --output-dir data/output
```

## 📝 Output Format

Generated conversations are saved as JSONL (one JSON object per line):

```json
{"_recordid": "rec_001", "conversation": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## 🐛 Troubleshooting

### Connection Errors
- Verify `base_url` is correct and accessible
- Check if API endpoint is running
- Ensure firewall allows connections

### Out of Memory
- Reduce `chunk_size_mb` in config
- Reduce `max_concurrent` workers
- Use the data chunker for large files

### Low Success Rate
- Check API endpoint logs for errors
- Increase `timeout` value
- Reduce `max_concurrent` to avoid overloading

### Missing Fields
- Ensure input JSON has `_recordid` and `Volltext` fields
- Check logs for specific error messages

## 📊 Monitoring

Monitor progress in real-time:

```bash
# Watch worker logs
tail -f logs/worker_1_err.log

# Count processed records
wc -l data/output/*.jsonl

# Check success rate in logs
grep "Success rate" logs/worker_1_err.log
```

## 🔧 Customization

### Modify Conversation Prompts

Edit the system and user messages in `src/worker.py`:
- `get_sys_msg()`: System prompt defining assistant behavior
- `get_usr_msg()`: User prompt template for conversation generation

### Add Custom Fields

Update the field mapping in `parse_file()` function to include additional metadata.

## 🧪 Testing

Test the pipeline without a real LLM API using the included mock server:

```bash
# Run complete integration test (automated)
./tests/run_integration_test.sh
```

This will:
- Start a mock LLM server on localhost:6000
- Process 3 test records
- Validate output format
- Clean up automatically

**Manual testing:**
```bash
# Terminal 1: Start mock server
python tests/mock_llm_server.py

# Terminal 2: Run worker with test data
python src/worker.py \
  --input-files tests/test_data_sample.json \
  --output-dir tests/output \
  --config tests/config_test.yaml
```

**Unit tests:**
```bash
pip install pytest
pytest tests/test_worker.py -v
```

See [`tests/README.md`](tests/README.md) for detailed testing documentation.

## 🙏 Citation

If you use this pipeline in your research, please cite:

```bibtex
[Add citation information]
```
