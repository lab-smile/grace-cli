# Use an official PyTorch CUDA runtime so torch + GPU "just works"
# (Change the tag if you need a different CUDA / torch combo)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Prevent Python from writing .pyc files & keep stdout/stderr unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && rm -rf /var/lib/apt/lists/*

COPY grace.py /app/grace.py

# Install Python dependencies
RUN pip install numpy nibabel scipy monai einops --quiet --no-cache-dir

# Copy the rest of the application
COPY . /app

# Make sure grace.py is executable
RUN chmod +x /app/grace.py

# Use tini as PID 1 to handle signals cleanly
ENTRYPOINT ["python3", "./grace.py"]

# This lets users pass arguments directly:
# e.g. docker run --gpus all grace-cli ./samples/brain.nii.gz