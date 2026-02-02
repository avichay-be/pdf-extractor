#!/usr/bin/env python3
"""
Integration Tests Runner

Cross-platform script to run integration tests:
1. Checks for test PDFs
2. Starts API server if not running
3. Runs integration tests
4. Stops API server
5. Reports results
"""
import subprocess
import sys
import time
import signal
import requests
from pathlib import Path


# Configuration
API_URL = "http://localhost:8000"
TEST_PDFS_DIR = Path("tests/test_pdfs")
OUTPUT_DIR = Path("tests/integration_output")
SERVER_STARTUP_TIMEOUT = 30  # seconds


class Colors:
    """ANSI color codes."""
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

    @staticmethod
    def green(text):
        return f"{Colors.GREEN}{text}{Colors.NC}"

    @staticmethod
    def yellow(text):
        return f"{Colors.YELLOW}{text}{Colors.NC}"

    @staticmethod
    def red(text):
        return f"{Colors.RED}{text}{Colors.NC}"


def print_header(text):
    """Print a header."""
    print("\n" + "=" * 50)
    print(text)
    print("=" * 50 + "\n")


def check_test_pdfs():
    """Check if test PDFs exist."""
    if not TEST_PDFS_DIR.exists():
        TEST_PDFS_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(TEST_PDFS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(Colors.yellow("Warning: No PDF files found in tests/test_pdfs/"))
        print("Integration tests will be skipped.")
        print("Add PDF files to tests/test_pdfs/ to run integration tests.")
        print("\nExample:")
        print("  cp your_document.pdf tests/test_pdfs/")
        return False

    print(Colors.green(f"Found {len(pdf_files)} test PDF(s)"))
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    return True


def check_server_running():
    """Check if API server is already running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def start_server():
    """Start the API server."""
    print("Starting API server...")

    # Start server process
    process = subprocess.Popen(
        [sys.executable, "run.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True  # Create new process group
    )

    # Wait for server to be ready
    print("Waiting for server to start", end="")
    for i in range(SERVER_STARTUP_TIMEOUT):
        if check_server_running():
            print()
            print(Colors.green("Server ready!"))
            return process
        print(".", end="", flush=True)
        time.sleep(1)

    print()
    print(Colors.red("Failed to start server"))
    print("Check if port 8000 is already in use")
    process.terminate()
    return None


def stop_server(process):
    """Stop the API server."""
    if process:
        print("\nStopping API server...")
        try:
            # Try graceful shutdown first
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if graceful shutdown failed
            process.kill()
            process.wait()
        print(Colors.green("Server stopped"))


def run_tests():
    """Run integration tests."""
    print_header("Running Integration Tests")

    # Run pytest
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/integration/", "-v", "-s", "--tb=short"],
        capture_output=False
    )

    return result.returncode


def main():
    """Main execution."""
    print_header("Michman PDF Extractor - Integration Tests")

    # Check for test PDFs
    if not check_test_pdfs():
        return 0

    # Check if server is already running
    server_process = None
    server_started = False

    if check_server_running():
        print(Colors.yellow(f"API server already running at {API_URL}"))
    else:
        server_process = start_server()
        if not server_process:
            return 1
        server_started = True

    try:
        # Run tests
        exit_code = run_tests()

        # Print results
        print_header("Test Results")

        if exit_code == 0:
            print(Colors.green("✓ All tests passed!"))
        else:
            print(Colors.red("✗ Some tests failed"))

        print(f"\nTest outputs saved to: {OUTPUT_DIR}/")

        return exit_code

    finally:
        # Stop server if we started it
        if server_started and server_process:
            stop_server(server_process)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
