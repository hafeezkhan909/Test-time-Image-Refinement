#!/bin/bash
# Initialize the project environment

# Create virtual environment
python3 -m venv diff
source diff/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt