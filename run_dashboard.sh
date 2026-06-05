#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define color variables for pretty console printing
GREEN='\033[0;32m'
NC='\033[0m' # No Color
BLUE='\033[0;34m'
YELLOW='\033[1;33m'

echo -e "${BLUE}[System] Initializing AOI Defect Inspection & RAG Console...${NC}"

# Navigate to project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Verify virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}[Warning] Virtual environment (.venv) not found. Initializing with uv...${NC}"
    uv venv
    uv pip install -r requirements.txt
fi

# Run the Streamlit application
echo -e "${GREEN}[Success] Starting Streamlit App on http://localhost:8501${NC}"
.venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --browser.gatherUsageStats false
