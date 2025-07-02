FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV UVICORN_PORT=8070
# Set the Hugging Face home directory for better model caching
ENV HF_HOME=/app/hf_cache


# Setup packages
RUN apt-get update && apt-get install -y wget curl git
RUN wget \
    https://developer.download.nvidia.com/compute/cudnn/9.10.2/local_installers/cudnn-local-repo-debian12-9.10.2_1.0-1_amd64.deb \
    && dpkg -i cudnn-local-repo-debian12-9.10.2_1.0-1_amd64.deb \
    && cp /var/cudnn-local-repo-debian12-9.10.2/cudnn-*-keyring.gpg /usr/share/keyrings/

# Update package list and install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    cudnn \
    cudnn-cuda-12 \
    ffmpeg \
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

RUN mkdir -p assets hf_cache

# Expose the application port
EXPOSE 8070

# Run the application using the venv's Python
CMD [".venv/bin/python", "mcp_tools.py"]
