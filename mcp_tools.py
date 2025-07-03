from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi_mcp import FastApiMCP
from fastapi.responses import FileResponse, JSONResponse
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
import uvicorn
import logging
import os
import csv
import subprocess
import re
import math
import asyncio
import sys
from faster_whisper import WhisperModel

# User Modules
from mcp_yt import ytDownloader

#########################################################
# Configuration and Constants
#########################################################

@dataclass
class Config:
    OUTPUT_PATH: str = "assets/output"
    UPLOAD_PATH: str = "assets/uploads"
    MODEL_SIZE: str = "large-v2"
    PROXY_USERNAME: str = "bwucltsa"
    PROXY_PASSWORD: str = "i21gi2m2lwdi"

config = Config()

#########################################################
# Logging Configuration
#########################################################

class ColorFormatter(logging.Formatter):
    COLORS = {
        "INFO": "\033[92m",
        "DEBUG": "\033[95m",
        "ERROR": "\033[91m",
        "RESET": "\033[0m",
    }

    def format(self, record):
        levelname = record.levelname
        color = self.COLORS.get(levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        record.levelname = f"{color}{levelname}{reset}"
        return super().format(record)

def setup_logging():
    handler = logging.StreamHandler()
    formatter = ColorFormatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = [handler]
    return logger

logger = setup_logging()

#########################################################
# Models
#########################################################

class TranscriptRequest(BaseModel):
    url: str
    timestamp: str

class SearchRequest(BaseModel):
    url: str

class VideoRequest(BaseModel):
    url: HttpUrl
    quality: Optional[str] = "720p"
    audio_only: Optional[bool] = False

class CreateVideoRequest(BaseModel):
    video_type: str
    number_of_videos: int

class CreateClipsRequest(BaseModel):
    clips: List[Dict[str, Any]]
    video_path: str
    subtitles: List[Tuple[Tuple[float, float], str]]

class AddUploadRequest(BaseModel):
    title: str
    description: str
    upload_file: str
    keywords: List[str]

#########################################################
# Core Services
#########################################################

class WhisperService:
    def __init__(self, model_size: str = "large-v2"):
        self.model = self._initialize_model(model_size)
    
    def _initialize_model(self, model_size: str) -> WhisperModel:
        if sys.platform == 'darwin':
            logger.info("Initializing Whisper model for macOS")
            return WhisperModel(model_size, device="cpu", compute_type="int8")
        elif sys.platform == 'linux':
            logger.info("Initializing Whisper model for Linux")
            return WhisperModel(model_size, device="cuda", compute_type="int8")
        else:
            logger.info("Initializing Whisper model for Windows")
            return WhisperModel(model_size, device="cuda", compute_type="float16")
    
    def transcribe(self, audio_file: Path) -> List[Tuple[Tuple[float, float], str]]:
        segments, info = self.model.transcribe(audio_file, beam_size=5)
        subtitles = []
        for segment in segments:
            logger.info(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            subtitles.append(((segment.start, segment.end), segment.text))
        return subtitles

class TranscriptService:
    def __init__(self, proxy_username: str, proxy_password: str):
        self.api = YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=proxy_username,
                proxy_password=proxy_password,
            )
        )
    
    def get_transcript(self, url: str, include_timestamps: bool = False) -> str:
        video_id = url.split("=")[1]
        fetched_transcript = self.api.fetch(video_id)
        
        if include_timestamps:
            result = []
            time = 0
            for item in fetched_transcript:
                if (item.start - time) > 10:
                    time = item.start
                    result.append(f"[time:{time}s] {item.text}")
                else:
                    result.append(item.text)
            return " ".join(result)
        else:
            return " ".join([snippet.text for snippet in fetched_transcript])

class VideoDownloadService:
    def __init__(self, output_path: str):
        self.downloader = ytDownloader(output_path)
        self.output_path = Path(output_path)
    
    def download_video(self, url: str, quality: str = "720p", audio_only: bool = False) -> Dict[str, Any]:
        result = self.downloader.download_video(url, quality=quality, audio_only=audio_only)
        
        if result["success"]:
            # Update file path based on URL
            video_id = str(url).split('=')[1]
            extension = ".m4a" if audio_only else ".mp4"
            corrected_path = self.output_path / f"{video_id}{extension}"
            result["file_path"] = corrected_path
            logger.info(f"Downloaded {'audio' if audio_only else 'video'}: {corrected_path}")
        
        return result

class SubtitleService:
    @staticmethod
    def format_time(seconds: float) -> str:
        hours = math.floor(seconds / 3600)
        seconds %= 3600
        minutes = math.floor(seconds / 60)
        seconds %= 60
        milliseconds = round((seconds - math.floor(seconds)) * 1000)
        seconds = math.floor(seconds)
        return f"{hours:02d}:{minutes:02d}:{seconds:01d},{milliseconds:03d}"
    
    @staticmethod
    def load_srt_to_segments(srt_path: Path) -> List[Tuple[Tuple[float, float], str]]:
        segments = []
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        blocks = content.split("\n\n")
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue
            
            time_line = lines[1]
            text_lines = lines[2:]
            text = " ".join(text_lines).strip()
            
            start_str, end_str = time_line.split(" --> ")
            
            def parse_time_str(s):
                h, m, rest = s.split(":")
                s, ms = rest.split(",")
                return float(h) * 3600 + float(m) * 60 + float(s) + float(ms) / 1000
            
            start = parse_time_str(start_str)
            end = parse_time_str(end_str)
            segments.append(((start, end), text))
        
        return segments
    
    @staticmethod
    def generate_srt_file(subtitles: List[Tuple[Tuple[float, float], str]], 
                         output_file: Path, start: float, end: Optional[float] = None):
        if end is None:
            end = max(sub[0][1] for sub in subtitles) if subtitles else start
        
        filtered_subs = []
        for sub in subtitles:
            sub_start, sub_end = sub[0]
            text = sub[1]
            
            if sub_end > start and sub_start < end:
                adjusted_start = max(0, sub_start - start)
                adjusted_end = min(end - start, sub_end - start)
                
                if adjusted_end > adjusted_start:
                    filtered_subs.append(((adjusted_start, adjusted_end), text))
        
        srt_content = ""
        for index, segment in enumerate(filtered_subs):
            segment_start = SubtitleService.format_time(segment[0][0])
            segment_end = SubtitleService.format_time(segment[0][1])
            srt_content += f"{index + 1}\n"
            srt_content += f"{segment_start} --> {segment_end}\n"
            srt_content += f"{segment[1].strip()}\n\n"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(srt_content)
        
        logger.info(f"Generated subtitles with {len(filtered_subs)} segments")
        logger.info(f"Time range: {start}s to {end}s (duration: {end - start}s)")

class VideoProcessingService:
    @staticmethod
    def make_subclip(input_video: Path, start_time: float, end_time: float, output_path: Path):
        duration = end_time - start_time
        command = [
            "ffmpeg", "-y", "-ss", str(start_time), "-i", str(input_video),
            "-t", str(duration), "-c", "copy", str(output_path)
        ]
        logger.info(" ".join(command))
        subprocess.run(command, check=True)
    
    @staticmethod
    def add_subtitles_to_video(video_file: Path, subtitle_file: Path, output_path: Path):
        command = [
            "ffmpeg", "-i", str(video_file), "-vf", f"subtitles={subtitle_file}",
            str(output_path)
        ]
        logger.info(" ".join(command))
        subprocess.run(command, check=True)

class CSVService:
    @staticmethod
    def read_video_tasks(csv_path: Path) -> List[Dict[str, str]]:
        videos = []
        with open(csv_path, mode="r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row["STATUS"] != "COMPLETED":
                    videos.append({
                        "video_id": row["UUID"],
                        "title": row["VIDEO_TITLE"],
                        "url": row["VIDEO_URL"],
                    })
        return videos
    
    @staticmethod
    def append_upload(upload_path: Path, title: str, description: str, 
                     upload_file: str, keywords: List[str]):
        with open(upload_path / "uploads.csv", mode='a', encoding='utf-8') as f:
            f.write(f'{title}, "{description}", "{upload_file}", "{" ".join(keywords)}"\n')

#########################################################
# Business Logic Layer
#########################################################

class VideoProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.whisper_service = WhisperService(config.MODEL_SIZE)
        self.transcript_service = TranscriptService(config.PROXY_USERNAME, config.PROXY_PASSWORD)
        self.download_service = VideoDownloadService(config.OUTPUT_PATH)
        self.subtitle_service = SubtitleService()
        self.video_processing_service = VideoProcessingService()
        self.csv_service = CSVService()
    
    def process_transcript_request(self, url: str, include_timestamps: bool) -> str:
        return self.transcript_service.get_transcript(url, include_timestamps)
    
    async def process_video_download(self, url: str, quality: str, audio_only: bool) -> Dict[str, Any]:
        result = self.download_service.download_video(url, quality, audio_only)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    
    async def process_video_creation(self, video_type: str, number_of_videos: int) -> List[Dict[str, Any]]:
        if video_type != "video_clip":
            raise HTTPException(status_code=400, detail="Only video_clip type is supported")
        
        base_dir = Path(__file__).parent.absolute()
        csv_path = base_dir / "assets" / "tasks" / "video_gen.csv"
        
        videos = self.csv_service.read_video_tasks(csv_path)[:number_of_videos]
        
        if not videos:
            raise HTTPException(status_code=404, detail="No available videos to work on")
        
        results = []
        for video_item in videos:
            result = await self._process_single_video(video_item)
            results.append(result)
        
        return results
    
    async def _process_single_video(self, video_item: Dict[str, str]) -> Dict[str, Any]:
        # Download audio
        loop = asyncio.get_running_loop()
        audio_result = await loop.run_in_executor(
            None, self.download_service.download_video, video_item["url"], "1080p", True
        )
        
        if not audio_result["success"]:
            raise HTTPException(status_code=400, detail="Error downloading audio")
        
        # Process subtitles
        file_name = "_".join(video_item["title"].lower().split(" "))
        file_name = re.sub(r"[|\\/]", "", file_name)
        srt_file_path = Path(self.config.OUTPUT_PATH) / f"{file_name}.srt"
        
        if srt_file_path.exists():
            logger.info(f"Loading existing SRT file: {srt_file_path}")
            subtitles = self.subtitle_service.load_srt_to_segments(srt_file_path)
        else:
            logger.info(f"Transcribing audio file: {audio_result['file_path']}")
            subtitles = self.whisper_service.transcribe(audio_result["file_path"])
            self.subtitle_service.generate_srt_file(subtitles, srt_file_path, 0)
        
        # Download video
        video_result = self.download_service.download_video(video_item["url"], "1080p", False)
        if not video_result["success"]:
            raise HTTPException(status_code=400, detail=video_result["error"])
        
        # Generate full transcript with timestamps
        transcript_full = self._generate_timestamped_transcript(subtitles)
        
        return {
            **video_item,
            "transcript": subtitles,
            "srt_path": str(srt_file_path),
            "transcript_full": transcript_full,
            "video_path": str(video_result["file_path"])
        }
    
    def _generate_timestamped_transcript(self, subtitles: List[Tuple[Tuple[float, float], str]]) -> str:
        timestamp_marks = 0
        result_parts = []
        
        for segment in subtitles:
            if (segment[0][0] - timestamp_marks) >= 10:
                result_parts.append(f"({segment[0][0]:.2f}s) {segment[1]}")
                timestamp_marks = segment[0][0]
            else:
                result_parts.append(segment[1])
        
        transcript_full = " ".join(result_parts)
        cleaned = re.sub(r"[\n\t'\"]", "", transcript_full)
        cleaned = re.sub(r"[^\x20-\x7E]", "", cleaned)
        
        return cleaned
    
    def process_clip_creation(self, clips: List[Dict[str, Any]], video_path: str, 
                            subtitles: List[Tuple[Tuple[float, float], str]]) -> List[Dict[str, Any]]:
        video_path_obj = Path(video_path)
        clip_dir = video_path_obj.parent / video_path_obj.stem
        clip_dir.mkdir(parents=True, exist_ok=True)
        
        clip_results = []
        
        for i, clip in enumerate(clips):
            clip_base = "_".join(clip["clip_title"].lower().split(" "))
            clip_base = re.sub(r"[,\.:;\\/|\'\"]", "", clip_base)
            clip_name = f"{clip_base}.mp4"
            clip_subbed_name = f"{clip_base}_subbed.mp4"
            clip_sub = f"{clip_base}.srt"
            
            clip_path = clip_dir / clip_name
            clip_subbed_path = clip_dir / clip_subbed_name
            clip_sub_path = clip_dir / clip_sub
            
            # Create clip
            self.video_processing_service.make_subclip(
                video_path_obj, clip["start"], clip["end"], clip_path
            )
            
            # Generate subtitles for clip
            self.subtitle_service.generate_srt_file(
                subtitles, clip_sub_path, clip["start"], clip["end"]
            )
            
            # Add subtitles to clip
            self.video_processing_service.add_subtitles_to_video(
                clip_path, clip_sub_path, clip_subbed_path
            )
            
            clip_results.append({
                "clip_path": str(clip_subbed_path),
                "description": clip["description"],
                "clip_title": clip["clip_title"],
                "keywords": clip["keywords"],
            })
            
            logger.info(f"Created clip #{i}: {clip_subbed_path}")
        
        return clip_results

#########################################################
# Utility Functions
#########################################################

def cleanup_file(file_path: str):
    """Background task to cleanup downloaded files after response"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup file {file_path}: {str(e)}")

#########################################################
# FastAPI Application
#########################################################

app = FastAPI(title="Video Processing API", version="1.0.0")

mcp = FastApiMCP(
    app,
    name="Multi-tool MCP Server",
    description="Current MCP server",
    describe_all_responses=True,
    describe_full_response_schema=True,
)

mcp.mount()

# Initialize the video processor
video_processor = VideoProcessor(config)

#########################################################
# API Endpoints
#########################################################

@app.get("/get_sample", operation_id="get_sample", response_model=Dict[str, str])
async def get_sample():
    return {"message": "Hello, world!"}

@app.post("/get_transcript", operation_id="get_transcript", response_model=Dict[str, str])
async def get_transcript(request: TranscriptRequest):
    try:
        include_timestamps = request.timestamp.lower() == "true"
        transcript = video_processor.process_transcript_request(request.url, include_timestamps)
        return {"transcript": transcript}
    except Exception as e:
        logger.error(f"Error getting transcript: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_web", operation_id="search_web")
async def search_web(request: SearchRequest):
    return {"search": "Function not yet implemented"}

@app.post("/download_video", operation_id="download_video", response_model=Dict[str, str])
async def download_video(request: VideoRequest, background_tasks: BackgroundTasks):
    try:
        result = await video_processor.process_video_download(
            str(request.url), request.quality, request.audio_only
        )
        
        file_path = result["file_path"]
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="Downloaded file not found")
        
        _, ext = os.path.splitext(file_path)
        media_type = "video/mp4" if ext.lower() == ".mp4" else "application/octet-stream"
        
        background_tasks.add_task(cleanup_file, str(file_path))
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=f"{result['title']}{ext}",
            headers={
                "Content-Disposition": f"attachment; filename=\"{result['title']}{ext}\"",
                "X-Video-Title": result["title"],
                "X-Video-Duration": str(result["duration"]),
                "X-File-Size": str(result["file_size"]),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/create_video", operation_id="create_video")
async def create_video(request: CreateVideoRequest):
    try:
        if request.video_type not in ["video_clip", "video_gen"]:
            raise HTTPException(
                status_code=400, 
                detail=f"{request.video_type} is not valid. Use ['video_clip', 'video_gen']"
            )
        
        if request.video_type == "video_gen":
            raise HTTPException(status_code=501, detail="Not yet implemented")
        
        result = await video_processor.process_video_creation(
            request.video_type, request.number_of_videos
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create_clips", operation_id="create_clips")
async def create_clips(request: CreateClipsRequest):
    try:
        clips = video_processor.process_clip_creation(
            request.clips, request.video_path, request.subtitles
        )
        return clips
    except Exception as e:
        logger.error(f"Error creating clips: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_upload", operation_id="add_upload")
async def add_upload(request: AddUploadRequest):
    try:
        upload_path = Path(config.UPLOAD_PATH)
        video_processor.csv_service.append_upload(
            upload_path, request.title, request.description, 
            request.upload_file, request.keywords
        )
        return {"uploads_list": str(upload_path / "uploads.csv")}
    except Exception as e:
        logger.error(f"Error adding upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

#########################################################
# Application Setup
#########################################################

mcp.setup_server()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8070)
