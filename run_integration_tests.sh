#!/bin/bash

# Integration Tests Runner Script
# Starts the API server, runs integration tests, and stops the server

set -e  # Exit on error

echo "=========================================="
echo "Michman PDF Extractor - Integration Tests"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if test PDFs exist
PDF_COUNT=$(find tests/test_pdfs -name "*.pdf" 2>/dev/null | wc -l)
if [ "$PDF_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}Warning: No PDF files found in tests/test_pdfs/${NC}"
    echo "Integration tests will be skipped."
    echo "Add PDF files to tests/test_pdfs/ to run integration tests."
    echo ""
    echo "Example:"
    echo "  cp your_document.pdf tests/test_pdfs/"
    exit 0
fi

echo -e "${GREEN}Found $PDF_COUNT test PDF(s)${NC}"
echo ""

# Check if server is already running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${YELLOW}API server already running at http://localhost:8000${NC}"
    SERVER_STARTED=false
else
    # Start the API server in background
    echo "Starting API server..."
    python run.py > /tmp/api_server.log 2>&1 &
    SERVER_PID=$!
    SERVER_STARTED=true

    # Wait for server to be ready
    echo "Waiting for server to start..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}Server ready!${NC}"
            break
        fi
        echo -n "."
        sleep 1
    done

    # Check if server started successfully
    if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${RED}Failed to start server${NC}"
        echo "Check logs in /tmp/api_server.log"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Running Integration Tests"
echo "=========================================="
echo ""

# Run integration tests
pytest tests/integration/ -v -s --tb=short

TEST_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Test Results"
echo "=========================================="

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
fi

echo ""
echo "Test outputs saved to: tests/integration_output/"
echo ""

# Stop server if we started it
if [ "$SERVER_STARTED" = true ]; then
    echo "Stopping API server..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo -e "${GREEN}Server stopped${NC}"
fi

echo ""
echo "=========================================="

exit $TEST_EXIT_CODE
