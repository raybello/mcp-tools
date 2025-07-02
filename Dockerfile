FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV UVICORN_PORT=8070
# Set the Hugging Face home directory for better model caching
ENV HF_HOME=/app/hf_cache

# Update package list and install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a symbolic link for `python` if it's not already present
RUN ln -s /usr/bin/python3 /usr/bin/python 

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Create virtual environment and install dependencies inside it
RUN python -m venv .venv && \
    .venv/bin/python -m pip install --upgrade pip && \
    .venv/bin/python -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY mcp_tools.py .
COPY mcp_yt.py .

RUN mkdir -p assets

# Expose the application port
EXPOSE 8070

# Run the application using the venv's Python
CMD [".venv/bin/python", "mcp_tools.py"]
