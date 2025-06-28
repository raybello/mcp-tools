# Multi-tool MCP Server

A FastAPI-based MCP (Model Context Protocol) server that provides YouTube transcript extraction capabilities.

## Features

- YouTube transcript extraction via POST API
- FastAPI with automatic OpenAPI documentation
- Docker containerization with Ubuntu base
- FFmpeg support for media processing
- Health checks and logging

## Quick Start

1. **Clone/Create the project directory:**
   ```bash
   mkdir mcp-server && cd mcp-server
   ```

2. **Add all the files** (Dockerfile, docker-compose.yml, requirements.txt, mcp_tools.py, etc.)

3. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build -d
   ```

4. **Access the API:**
   - API Base: http://localhost:8070
   - API Docs: http://localhost:8070/docs
   - Redoc: http://localhost:8070/redoc

## API Endpoints

### POST /v1/transcript/
Extract transcript from a YouTube video.

**Request Body:**
```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

**Response:**
```json
{
  "transcript": "Full video transcript text..."
}
```

### GET /v1/sample/
Sample endpoint returning a hello world message.

### GET /
Health check endpoint.

## Configuration

### Environment Variables

- `PORT`: Server port (default: 8070)
- `UVICORN_HOST`: Host address (default: 0.0.0.0)
- `UVICORN_PORT`: Uvicorn port (default: 8070)
- `UVICORN_RELOAD`: Enable auto-reload (default: false)

### Docker Volumes

- `./logs:/app/logs` - Application logs
- `./data:/app/data` - Persistent data storage

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python mcp_tools.py
```

### Docker Development
```bash
# Build the image
docker compose build

# Run in development mode
docker compose up

# View logs
docker compose logs -f

# Stop the service
docker compose down
```

## Dependencies

- FastAPI: Web framework
- fastapi-mcp: MCP integration
- youtube-transcript-api: YouTube transcript extraction
- uvicorn: ASGI server
- pydantic: Data validation

## System Requirements

- Docker and Docker Compose
- Python 3.8+ (for local development)
- FFmpeg (installed in Docker container)

## MCP Server Directory Structure

```
mcp-server/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── mcp_tools.py
├── .dockerignore
├── .gitignore
├── README.md
├── logs/          # Will be created by Docker
└── data/          # Will be created by Docker
```

## Files Overview

- **Dockerfile**: Defines the container image with Ubuntu base, Python, and FFmpeg
- **docker-compose.yml**: Orchestrates the container with port mapping and volumes
- **requirements.txt**: Python dependencies for the MCP server
- **mcp_tools.py**: Your main application file
- **.dockerignore**: Excludes unnecessary files from Docker build context
- **.gitignore**: Git ignore patterns for development
- **logs/**: Directory for application logs (mounted as volume)
- **data/**: Directory for any persistent data (mounted as volume)

## Usage

1. Place all files in the `mcp-server/` directory
2. Run `docker-compose up --build` to build and start the service
3. Access the API at `http://localhost:8070`
4. View API docs at `http://localhost:8070/docs`