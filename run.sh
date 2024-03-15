#!/bin/bash

# Define variables
VENV_NAME="myenv"
REQUIREMENTS_FILE="requirements.txt"
COMMAND1="python ingest.py"
COMMAND2="streamlit run main.py"

# Create virtual environment
echo "Creating virtual environment..."
python -m venv $VENV_NAME

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r $REQUIREMENTS_FILE

# Run command 1
echo "Running ingestion: $COMMAND1"
$COMMAND1

# Run command 2
echo "Running streamlit: $COMMAND2"
$COMMAND2


echo "Script execution complete."