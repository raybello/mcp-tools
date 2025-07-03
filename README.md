
# 🎬 MCP Tools — Video Processing & Clipping API

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/containerized-docker-blue)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

A full-stack, containerized video automation toolkit. Download YouTube videos, generate transcripts and subtitles, create clips with overlays — all in one fast, extensible Python API.

---

## 📽️ How It Works (Visual)

| 🎯 Input (YouTube URL) | 🔎 Download & Transcribe | ✂️ Clip + Subtitles | ✅ Output (Ready-to-upload Video) |
|------------------------|--------------------------|---------------------|----------------------------------|
| ![](docs/assets/url_input.png) | ![](docs/assets/download_transcribe.gif) | ![](docs/assets/clip_subtitle.gif) | ![](docs/assets/final_video.gif) |

> YouTube → FFmpeg + Whisper + SRT → Clip → Subbed Video → Done ✅

---

## 🚀 Quick Start

### 🐳 Run with Docker Compose

```bash
docker-compose up --build
```

Then visit:  
🔗 [http://localhost:8070/docs](http://localhost:8070/docs) — for Swagger UI

---

## 🧠 Features

- 🔽 Download YouTube **video or audio** via `yt-dlp`
- 🧠 Transcribe audio with **Faster Whisper**
- 📝 Auto-generate `.srt` **subtitles**
- ✂️ Create video **subclips** with hardcoded subtitles
- 📜 Generate full **transcripts with timestamps**
- 📤 Append video metadata to `uploads.csv`
- ⚙️ Fully **containerized** for consistent environments

---

## 🛠 Project Structure

```text
.
├── Dockerfile
├── docker-compose.yml
├── mcp_tools.py          # Main FastAPI app
├── assets/
│   ├── output/           # Output clips and videos
│   ├── uploads/          # Upload logs (CSV)
│   └── tasks/
│       └── video_gen.csv # Source video jobs
```

---

## 📬 API Endpoints

> Full docs at `/docs` once the server is running.

### ✅ `/get_sample`

Simple test endpoint.

---

### 📝 `/get_transcript`

**POST**: Get YouTube transcript (optionally with timestamps)

```json
{
  "url": "https://www.youtube.com/watch?v=abc123",
  "timestamp": "true"
}
```

---

### 🔽 `/download_video`

**POST**: Download video or audio

```json
{
  "url": "https://www.youtube.com/watch?v=abc123",
  "quality": "1080p",
  "audio_only": false
}
```

---

### 🎥 `/create_video`

**POST**: Download, transcribe, and process `N` videos listed in `video_gen.csv`

```json
{
  "video_type": "video_clip",
  "number_of_videos": 2
}
```

---

### ✂️ `/create_clips`

**POST**: Create subtitle-embedded subclips from a video

```json
{
  "clips": [
    {
      "clip_title": "highlight",
      "start": 10.0,
      "end": 25.0,
      "description": "Tech news highlight",
      "keywords": ["tech", "highlight"]
    }
  ],
  "video_path": "assets/output/sample.mp4",
  "subtitles": [[[10.0, 25.0], "Welcome to the future of tech!"]]
}
```

---

### 📤 `/add_upload`

**POST**: Append metadata for tracking uploads

```json
{
  "title": "Tech Highlight",
  "description": "A short clip on the latest in AI.",
  "upload_file": "assets/output/sample/highlight_subbed.mp4",
  "keywords": ["AI", "shorts"]
}
```

---

## 🧱 Internals

### Services

| Service Name            | Role                                                  |
|-------------------------|--------------------------------------------------------|
| `WhisperService`        | Transcribe audio to text using Faster Whisper         |
| `VideoDownloadService`  | Uses yt-dlp to download video/audio                   |
| `SubtitleService`       | Handles `.srt` creation and parsing                   |
| `VideoProcessingService`| Subclip creation and subtitle hardcoding (FFmpeg)     |
| `TranscriptService`     | Pulls YouTube transcript with proxy support           |
| `CSVService`            | Reads tasks and appends upload logs                   |

---

## 🧰 Configuration

Located in `mcp_tools.py`:

```python
@dataclass
class Config:
    OUTPUT_PATH = "assets/output"
    UPLOAD_PATH = "assets/uploads"
    MODEL_SIZE = "large-v2"
    PROXY_USERNAME = "your_username"
    PROXY_PASSWORD = "your_password"
```

---

## 🔐 Requirements

Automatically handled in the container:

- `yt-dlp`
- `faster-whisper`
- `youtube-transcript-api`
- `ffmpeg`
- `uvicorn`, `fastapi`, `pydantic`
- Python 3.10+

---

## 🧹 File Cleanup

Video/audio files are deleted after sending via `BackgroundTasks`.

---

## 🧪 Development Tips

To run locally outside Docker:

```bash
pip install -r requirements.txt
uvicorn mcp_tools:app --reload --port 8070
```

---

## 📈 Roadmap

- [ ] Web search & retrieval
- [ ] S3/Drive upload support
- [ ] Web frontend
- [ ] Multi-language transcription
- [ ] AI-based auto-clipping

---

## 👩🏽‍💻 Author

**Ray Bello**  
🔗 [github.com/raybello](https://github.com/raybello)

---

## 📄 License

MIT License. See `LICENSE` file.
