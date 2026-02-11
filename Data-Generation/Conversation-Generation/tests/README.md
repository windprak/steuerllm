# Testing Guide

This directory contains tests and testing infrastructure for the conversation generation pipeline.

## Quick Start

### Run All Tests (Automated)

```bash
# Run the complete integration test with mock LLM server
./tests/run_integration_test.sh
```

This script will:
1. Install dependencies (including Flask for mock server)
2. Start the mock LLM server
3. Run the worker with test data
4. Validate the output
5. Clean up automatically

### Run Unit Tests Only

```bash
# Install pytest if not already installed
pip install pytest

# Run unit tests
pytest tests/test_worker.py -v
```

## Test Components

### 1. Mock LLM Server (`mock_llm_server.py`)

A lightweight Flask server that simulates an OpenAI-compatible API endpoint.

**Features:**
- Returns realistic German tax law conversations
- Supports the `/v1/chat/completions` endpoint
- Adds random delays (0.1-0.5s) to simulate real API
- Health check endpoint at `/health`

**Run standalone:**
```bash
python tests/mock_llm_server.py
```

The server will start on `http://localhost:6000`

### 2. Test Data (`test_data_sample.json`)

Sample input data with 3 test records covering:
- Einkommensteuer (Income Tax)
- Umsatzsteuer (VAT)
- Abschreibungen (Depreciation)

All records include the required `_recordid` and `Volltext` fields plus optional metadata.

### 3. Test Configuration (`config_test.yaml`)

Test-specific configuration that:
- Points to `localhost:6000` for the mock server
- Uses small concurrency (3) for quick tests
- Sets short timeout (30s)
- Outputs to `tests/output/`

### 4. Integration Test Script (`run_integration_test.sh`)

Bash script that orchestrates the full pipeline test:
- ✓ Dependency installation
- ✓ Mock server lifecycle management
- ✓ Worker execution
- ✓ Output validation
- ✓ Automatic cleanup

### 5. Unit Tests (`test_worker.py`)

Python unit tests using pytest:
- Message generation functions
- Configuration loading
- Data format validation
- JSON parsing logic

## Manual Testing

### Test Individual Components

**1. Test Mock Server:**
```bash
# Terminal 1: Start server
python tests/mock_llm_server.py

# Terminal 2: Test endpoint
curl http://localhost:6000/health
```

**2. Test Worker:**
```bash
# With mock server running
python src/worker.py \
  --input-files tests/test_data_sample.json \
  --output-dir tests/output \
  --config tests/config_test.yaml
```

**3. Check Output:**
```bash
# View generated conversations
cat tests/output/test_data_sample.jsonl | python -m json.tool
```

## Expected Output

After a successful test run, you should see:

```
tests/output/
  └── test_data_sample.jsonl    # 3 lines, one per input record
```

Each line in the output file contains:
```json
{
  "_recordid": "test_001",
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ]
}
```

## Validation Checks

The integration test validates:
- ✓ All input records are processed
- ✓ Output is valid JSON (one object per line)
- ✓ Each record has `_recordid` and `conversation` fields
- ✓ Conversations contain user and assistant messages
- ✓ No errors in processing

## Troubleshooting

### Mock Server Won't Start

**Port already in use:**
```bash
# Find and kill process on port 6000
lsof -ti:6000 | xargs kill -9
```

**Missing Flask:**
```bash
pip install flask
```

### Tests Fail

**Check logs:**
```bash
# Mock server logs
cat tests/logs/mock_server.log

# Worker output (if saved)
ls tests/logs/
```

**Cleanup and retry:**
```bash
rm -rf tests/output tests/logs
./tests/run_integration_test.sh
```

### Permission Issues

```bash
# Make test script executable
chmod +x tests/run_integration_test.sh
```

## Testing with Real API

To test with a real LLM API instead of the mock:

1. Copy the test config:
   ```bash
   cp tests/config_test.yaml config/config_real.yaml
   ```

2. Edit with real API details:
   ```yaml
   api:
     base_url: "http://your-real-api:8000"
     api_key: "your-real-key"
     model: "your-real-model"
   ```

3. Run worker with real config:
   ```bash
   python src/worker.py \
     --input-files tests/test_data_sample.json \
     --output-dir tests/output_real \
     --config config/config_real.yaml
   ```

## Continuous Integration

To integrate with CI/CD pipelines (GitHub Actions, GitLab CI, etc.):

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    pip install -r requirements.txt
    pip install pytest flask
    pytest tests/test_worker.py
    ./tests/run_integration_test.sh
```

## Adding New Tests

### Add Unit Test

Edit `tests/test_worker.py`:
```python
def test_new_feature():
    """Test description."""
    result = your_function()
    assert result == expected_value
```

### Add Integration Test Data

Edit `tests/test_data_sample.json` to add more test cases:
```json
{
  "_recordid": "test_004",
  "Volltext": "Your test content...",
  ...
}
```

### Modify Mock Responses

Edit `tests/mock_llm_server.py` to customize conversation templates:
```python
CONVERSATION_TEMPLATES = [
    # Add your custom template here
    [{"role": "user", "content": "..."}, ...]
]
```

## Performance Testing

For load testing the pipeline:

```bash
# Generate larger test dataset
python -c "
import json
data = [{'_recordid': f'perf_{i}', 'Volltext': f'Test content {i}'} 
        for i in range(100)]
with open('tests/test_large.json', 'w') as f:
    json.dump(data, f)
"

# Run with high concurrency
python src/worker.py \
  --input-files tests/test_large.json \
  --output-dir tests/output_perf \
  --config tests/config_test.yaml \
  --max-concurrent 20
```

## Clean Up

```bash
# Remove all test outputs
rm -rf tests/output tests/logs

# Remove generated test data
rm -f tests/test_large.json
```
