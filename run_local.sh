#!/bin/bash
# Run the application locally
export PYTHONPATH=$PYTHONPATH:$(pwd)
uvicorn main:app --reload --host 0.0.0.0 --port 8002
