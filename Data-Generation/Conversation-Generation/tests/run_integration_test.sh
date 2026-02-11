#!/bin/bash
# Integration Test Script
# Tests the entire pipeline with mock LLM server

set -e

echo "========================================"
echo "Conversation Generation - Integration Test"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ ! -z "$MOCK_SERVER_PID" ]; then
        kill $MOCK_SERVER_PID 2>/dev/null || true
        echo "✓ Mock server stopped"
    fi
    rm -rf tests/output tests/logs
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Detect Python executable
if command -v python &> /dev/null; then
    PYTHON=python
elif command -v python3 &> /dev/null; then
    PYTHON=python3
else
    echo -e "${RED}✗ Python not found${NC}"
    exit 1
fi

echo "Using Python: $PYTHON"

echo "Step 1: Installing dependencies..."
$PYTHON -m pip install -q -r requirements.txt
$PYTHON -m pip install -q flask pytest
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Cleanup from previous runs
rm -rf tests/output tests/logs
mkdir -p tests/output tests/logs

echo "Step 2: Starting mock LLM server..."
$PYTHON tests/mock_llm_server.py > tests/logs/mock_server.log 2>&1 &
MOCK_SERVER_PID=$!
echo "Mock server PID: $MOCK_SERVER_PID"

# Wait for server to start
sleep 2

# Check if server is running
if ! kill -0 $MOCK_SERVER_PID 2>/dev/null; then
    echo -e "${RED}✗ Mock server failed to start${NC}"
    cat tests/logs/mock_server.log
    exit 1
fi

# Test server health
if curl -s http://localhost:6000/health | grep -q "healthy"; then
    echo -e "${GREEN}✓ Mock server is running${NC}"
else
    echo -e "${RED}✗ Mock server health check failed${NC}"
    exit 1
fi
echo ""

echo "Step 3: Running worker with test data..."
$PYTHON src/worker.py \
    --input-files tests/test_data_sample.json \
    --output-dir tests/output \
    --config tests/config_test.yaml \
    --max-concurrent 3

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Worker completed successfully${NC}"
else
    echo -e "${RED}✗ Worker failed${NC}"
    exit 1
fi
echo ""

echo "Step 4: Validating output..."

# Check if output file exists
OUTPUT_FILE="tests/output/test_data_sample.jsonl"
if [ ! -f "$OUTPUT_FILE" ]; then
    echo -e "${RED}✗ Output file not found: $OUTPUT_FILE${NC}"
    exit 1
fi

# Count records in output
RECORD_COUNT=$(wc -l < "$OUTPUT_FILE" | tr -d ' ')
echo "Records generated: $RECORD_COUNT"

if [ "$RECORD_COUNT" -eq 3 ]; then
    echo -e "${GREEN}✓ All 3 test records processed${NC}"
else
    echo -e "${YELLOW}⚠ Expected 3 records, got $RECORD_COUNT${NC}"
fi

# Validate JSON structure
echo "Validating JSON structure..."
VALID=true
while IFS= read -r line; do
    if ! echo "$line" | $PYTHON -m json.tool > /dev/null 2>&1; then
        echo -e "${RED}✗ Invalid JSON line${NC}"
        VALID=false
        break
    fi
    
    # Check for required fields
    if ! echo "$line" | grep -q '"_recordid"'; then
        echo -e "${RED}✗ Missing _recordid field${NC}"
        VALID=false
        break
    fi
    
    if ! echo "$line" | grep -q '"conversation"'; then
        echo -e "${RED}✗ Missing conversation field${NC}"
        VALID=false
        break
    fi
done < "$OUTPUT_FILE"

if $VALID; then
    echo -e "${GREEN}✓ JSON structure is valid${NC}"
else
    echo -e "${RED}✗ JSON validation failed${NC}"
    exit 1
fi

# Show sample output
echo ""
echo "Sample conversation from output:"
echo "--------------------------------"
head -n 1 "$OUTPUT_FILE" | $PYTHON -m json.tool | head -n 20
echo "..."
echo ""

echo "========================================"
echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
echo "========================================"
echo ""
echo "Generated conversations: $OUTPUT_FILE"
echo "Logs: tests/logs/"
echo ""
