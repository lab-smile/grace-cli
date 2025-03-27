#!/bin/bash

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh <input_nifti_file.nii.gz>"
    exit 1
fi

# Check if the input file exists and has .nii.gz extension
if [ ! -f "$1" ] || [[ "$1" != *.nii.gz ]]; then
    echo "Error: Input file does not exist or is not a .nii.gz file"
    exit 1
fi

# Create and activate Python virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Make sure grace.py is executable
chmod +x grace.py

# Run the grace.py script with the input file
python ./grace.py "$1"

# Deactivate virtual environment
deactivate 