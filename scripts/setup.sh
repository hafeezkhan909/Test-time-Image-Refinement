#!/bin/bash
# Initialize the project environment

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p scripts/extract
mkdir -p scripts/post_process
mkdir -p generated_images
mkdir -p new_outputs