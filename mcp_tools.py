from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi_mcp import FastApiMCP
from fastapi.responses import FileResponse, JSONResponse
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from pydantic import BaseModel, HttpUrl
from typing import Optional

from moviepy import (
    VideoFileClip,
    TextClip,
    ImageClip,
    CompositeVideoClip,
    clips_array,
    vfx,
)
from moviepy.video.tools.subtitles import SubtitlesClip

from faster_whisper import WhisperModel

import uvicorn
import logging
import os
from pathlib import Path
import csv
import subprocess
import re
import math
import asyncio
import sys

# User Modules
from mcp_yt import ytDownloader

#########################################################
# Globals
#########################################################


# Configure logging
class ColorFormatter(logging.Formatter):
    # Define color codes
    COLORS = {
        "INFO": "\033[92m",  # Green
        "DEBUG": "\033[95m",  # Purple
        "ERROR": "\033[91m",  # Red
        "RESET": "\033[0m",  # Reset to default
    }

    def format(self, record):
        levelname = record.levelname
        color = self.COLORS.get(levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        record.levelname = f"{color}{levelname}{reset}"
        return super().format(record)

# Configure logger with the custom formatter
handler = logging.StreamHandler()
formatter = ColorFormatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # or INFO
logger.handlers = [handler]

# Global downloader instance
OUTPUT_PATH = "assets/output"
UPLOAD_PATH = "assets/uploads"
downloader = ytDownloader(OUTPUT_PATH)

MODEL_SIZE = "large-v2"
# Run on GPU with FP16
# MODEL = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
# or run on GPU with INT8
# MODEL = WhisperModel(MODEL_SIZE, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
if sys.platform == 'darwin': # for macOS
    print("macOS")
    MODEL = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
elif sys.platform == 'linux': # for Linux
    print("Linux")
    MODEL = WhisperModel(MODEL_SIZE, device="cuda", compute_type="int8")
else: # for Windows
    print("Windows")
    MODEL = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")


app = FastAPI()

mcp = FastApiMCP(
    app,
    name="Multi-tool MCP Server",
    description="Current MCP server",
    describe_all_responses=True,
    describe_full_response_schema=True,
)

mcp.mount()

#########################################################
# Models
#########################################################


# Pydantic model for the POST request body
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
    clips: list
    video_path: str
    subtitles: list

class AddUploadRequest(BaseModel):
    title: str
    description: str
    upload_file: str
    keywords: list

#########################################################
# Functions
#########################################################
def format_time(seconds):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:01d},{milliseconds:03d}"
    return formatted_time

def load_srt_to_segments(srt_path):
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

        # Parse time like: 00:00:01,000 --> 00:00:04,000
        start_str, end_str = time_line.split(" --> ")

        def parse_time_str(s):
            h, m, rest = s.split(":")
            s, ms = rest.split(",")
            return float(h) * 3600 + float(m) * 60 + float(s) + float(ms) / 1000

        start = parse_time_str(start_str)
        end = parse_time_str(end_str)

        segments.append(((start, end), text))
    return segments

def cleanup_file(file_path: str):
    """Background task to cleanup downloaded files after response"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup file {file_path}: {str(e)}")

def make_subclip(input_video, start_time, end_time, output_path):
    """
    Creates a subclip using ffmpeg without re-encoding.

    :param input_video: Path to the source video
    :param start_time: Start time in format 'HH:MM:SS.xxx' or seconds (float/int)
    :param end_time: End time in the same format
    :param output_path: Path to save the clipped video
    """
    duration = float(end_time) - float(start_time)

    command = [
        "ffmpeg",
        "-y",  # Overwrite without asking
        "-ss",
        str(start_time),
        "-i",
        str(input_video),
        "-t",
        str(duration),
        "-c",
        "copy",  # Copy codec (no re-encoding)
        str(output_path),
    ]
    logger.info(" ".join(command))
    subprocess.run(command, check=True)

def gen_srt_file(subs, output_file, start, end=None):
    """
    Generate an SRT subtitle file from subtitle data.

    Args:
        subs: Array of subtitles in format [[[start, end], "text"], ...]
              Example: [[[0,3], "Good morning dear"], [[3,5], "Not too bad"]]
        output_file: Path to output SRT file
        start: Start time in seconds to extract subtitles from
        end: End time in seconds (optional). If None, uses last subtitle's end time
    """

    # Given an array of subtitle of format
    # [[[start, end], "text"]]
    # [[[0,3], "Good morning dear"], [[3,5], "Not too bad"]]

    # User-specifies the start_time and end_time that subtitles
    # Should be created for.
    # When generating a srt file, for example from 4.5 to 79.1
    # subtitles in srt should start from 0 seconds up to (79.1 - 4.5) = 74.6 seconds
    # When No end time is specified, subtitle should be generated from start_time to last subtitle item

    # Filter subtitles within the specified time range
    filtered_subs = []

    # Determine actual end time
    if end is None:
        end = max(sub[0][1] for sub in subs) if subs else start

    for sub in subs:
        sub_start, sub_end = sub[0]
        text = sub[1]

        # Check if subtitle overlaps with our time range
        if sub_end > start and sub_start < end:
            # Adjust timing relative to our start time
            adjusted_start = max(0, sub_start - start)
            adjusted_end = min(end - start, sub_end - start)

            # Only include if there's actual duration after adjustment
            if adjusted_end > adjusted_start:
                filtered_subs.append([[adjusted_start, adjusted_end], text])

    # Generate SRT content
    text = ""
    for index, segment in enumerate(filtered_subs):
        segment_start = format_time(segment[0][0])
        segment_end = format_time(segment[0][1])
        text += f"{str(index+1)} \n"
        text += f"{segment_start} --> {segment_end} \n"
        text += f"{segment[1].strip('\n')} \n"
        text += "\n"

    # Write to file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

        logger.info(f"Generated subtitles with {len(filtered_subs)} segments")
        logger.info(f"Time range: {start}s to {end}s (duration: {end - start}s)")
        logger.info(f"Subtitle file: {output_file}")

    except IOError as e:
        logger.error(f"Error writing subtitle file {output_file}: {e}")
        raise

def gen_srt_video(video_file, sub_path, output_path=Path(OUTPUT_PATH, "sample.mp4").as_posix()):
    # ffmpeg -i mymovie.mp4 -vf subtitles=subtitles.srt mysubtitledmovie.mp4

    command = [
        "ffmpeg", 
        "-i", f"{video_file}",
        "-vf", 
        f"subtitles={sub_path}",
        f"{output_path}"
    ]
    logger.info(" ".join(command))
    subprocess.run(command, check=True)

#########################################################
# Endpoints
#########################################################

# This endpoint will not be registered as a tool, since it was added after the MCP instance was created
@app.get("/get_sample", operation_id="get_sample", response_model=dict[str, str])
async def get_sample():
    return {"message": "Hello, world!"}


@app.post(
    "/get_transcript", operation_id="get_transcript", response_model=dict[str, str]
)
async def get_transcript(request: TranscriptRequest):
    url = request.url
    timestamp = request.timestamp
    ytt_api = YouTubeTranscriptApi(
        proxy_config=WebshareProxyConfig(
            proxy_username="bwucltsa",
            proxy_password="i21gi2m2lwdi",
        )
    )

    fetched_transcript = ytt_api.fetch(url.split("=")[1])

    if timestamp.lower() == "true":
        result = []
        time = 0
        for item in fetched_transcript:
            # print(item)
            if (item.start - time) > 10:
                time = item.start
                result.append(f"[time:{time}s] {item.text}")
            else:
                result.append(f"{item.text}")

        # print(result)
        transcript = " ".join(result)
    else:
        transcript = " ".join([snippet.text for snippet in fetched_transcript])

    return {"transcript": f"{transcript}"}


# Endpoint that given a search topic, searches google and formats the content of the first 3 responses
# as a text file.
@app.post("/search_web", operation_id="search_web")
async def search_web(request: SearchRequest):
    # TODO: Implement this
    return {"search": "Function not yet implemented"}


# @app.post("/download_video", operation_id="download_video")
@app.post(
    "/download_video", operation_id="download_video", response_model=dict[str, str]
)
async def download_video(request: VideoRequest, background_tasks: BackgroundTasks):
    """Download a YouTube video and return it as a file response"""
    try:
        url = str(request.url)
        quality = request.quality
        logger.info(f"Downloading video: {url} in quality: {quality}")

        # Download the video
        result = downloader.download_video(url, quality=quality, audio_only=False)

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])

        file_path = result["file_path"]

        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="Downloaded file not found")

        # Get file extension for proper media type
        _, ext = os.path.splitext(file_path)
        media_type = (
            "video/mp4" if ext.lower() == ".mp4" else "application/octet-stream"
        )

        # Schedule file cleanup after response
        background_tasks.add_task(cleanup_file, file_path)

        # Return file response
        return FileResponse(
            path=file_path,
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

    valid_types = ["video_gen", "video_clip"]
    if request.video_type not in valid_types:
        return JSONResponse(
            {
                "error": f"{request.video_type} is not valid. Use {valid_types.__str__()}"
            },
            status_code=500,
        )

    logger.info(f"Using flow for {request.video_type}")

    if request.video_type == "video_clip":
        base_dir = Path(__file__).parent.absolute()
        assets_dir = Path(base_dir, "assets")
        tasks_dir = Path(assets_dir, "tasks")

        csv_filename = Path(tasks_dir, f"video_gen.csv")

        videos = []
        count = 0

        with open(csv_filename, mode="r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                if row["STATUS"] != "COMPLETED":
                    if count < request.number_of_videos:
                        # print(row["UUID"])
                        videos.append(
                            {
                                "video_id": row["UUID"],
                                "title": row["VIDEO_TITLE"],
                                "url": row["VIDEO_URL"],
                            }
                        )
                        count += 1
                    else:
                        break

        logger.info(videos)

        if len(videos) == 0:
            return JSONResponse(
                {"error": f"No available videos to work on"},
                status_code=500,
            )

        response = []

        for video_item in videos:
            response_item = video_item
            
            audio_file = None
            audio_result = None
            
            file_path = None
            result = None
            
            subtitles = []
            
            # Download the audio file
            loop = asyncio.get_running_loop()
            audio_result = await loop.run_in_executor(None, downloader.download_video, video_item["url"], "1080p", True)
            
            if not audio_result["success"]:
                return JSONResponse(
                    {"error": "error downloading audio"},
                    status_code=400,
                )
            # audio_file = audio_result["file_path"]
            audio_file = str(video_item["url"]).split('=')[1]
            audio_file = Path(OUTPUT_PATH, f"{audio_file}.m4a")
            audio_result["file_path"] = audio_file
            
            logger.info(f"Downloaded audio for: {audio_result["title"]}")
            logger.info(f"Downloaded audio file: {audio_file}")
            
            # Prepare name
            _file = "_".join(str(video_item["title"]).lower().split(" "))
            _file = re.sub(r"[|\\/]", "", _file)

            srt_file_path = Path(OUTPUT_PATH, f"{_file}.srt")

            # === If subtitle file exists, load it ===
            if srt_file_path.exists():
                logger.info(
                    f"SRT file already exists: {srt_file_path.as_posix()}, loading instead of transcribing..."
                )
                subtitles = load_srt_to_segments(srt_file_path.as_posix())
            else:
                # === Transcribe audio if no SRT ===
                logger.info(f"No existing SRT found. Transcribing audio file: {audio_file}")
                fetched_transcript, info = MODEL.transcribe(Path(audio_file), beam_size=5)

                subtitles = []
                for segment in fetched_transcript:
                    print(
                        "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
                    )
                    subtitles.append(((segment.start, segment.end), segment.text))

                # Create srt file
                gen_srt_file(subtitles, srt_file_path.as_posix(), 0)

            response_item["transcript"] = subtitles
            response_item["srt_path"] = srt_file_path.as_posix()

            timestamp_marks = 0
            result_parts = []
            for segment in subtitles:
                if (segment[0][0] - timestamp_marks) >= 10:
                    result_parts.append(f"({segment[0][0]:.2f}s) {segment[1]}")
                    timestamp_marks = segment[0][0]
                else:
                    result_parts.append(segment[1])

            transcript_full = " ".join(result_parts)
            cleaned = re.sub(
                r"[\n\t\'\"]", "", transcript_full
            )  # Remove newlines, tabs, quotes
            cleaned = re.sub(
                r"[^\x20-\x7E]", "", cleaned
            )  # Remove non-ASCII printable chars

            response_item["transcript_full"] = f"{cleaned}"
            logger.info(f"Created full transcript to 'transcript_full'")

            # Download the video
            result = downloader.download_video(
                video_item["url"], quality="1080p", audio_only=False
            )
            if not result["success"]:
                return JSONResponse(
                    {"error": result["error"]},
                    status_code=400,
                )
                
            video_file = str(video_item["url"]).split('=')[1]
            video_file = Path(OUTPUT_PATH, f"{video_file}.m4a")
            result["file_path"] = video_file
            
            logger.info(f"Downloaded video for: {video_item["title"]}")
            logger.info(f"Downloaded video file: {video_file}")

            # Set video path
            response_item["video_path"] = f"{video_file}"
            response.append(response_item)

        return response

    if request.video_type == "video_gen":
        return JSONResponse(
            {"error": f"Not yet implemented"},
            status_code=500,
        )

    return request


@app.post("/create_clips", operation_id="create_clips")
async def create_clips(request: CreateClipsRequest):

    logger.info(f"Number of clips {len(request.clips)}")
    logger.info(f"Video to edit {request.video_path}")
    logger.info(f"Length of subtitles {len(request.subtitles)}")

    # Create directory to store clips
    video_path = request.video_path
    clip_dir = Path(request.video_path.split(".")[0])
    print(f"Created folder {clip_dir}\n")

    # Create directories if they don't exist
    clip_dir.mkdir(parents=True, exist_ok=True)

    # list of clips
    list_of_clips = []

    # Iterate over the number of clips and create them from the main video
    for i, clip in enumerate(request.clips):
        clip_base = f"{'_'.join(str(clip['clip_title']).lower().split(' '))}"
        clip_name = f"{clip_base}.mp4"
        clip_subbed_name = f"{clip_base}_subbed.mp4"
        clip_sub = f"{clip_base}.srt"

        logger.info(f"Clip#{i} file: {clip_name}")

        clip_path = Path(clip_dir, clip_name)
        clip_subbed_path = Path(clip_dir, clip_subbed_name)
        clip_sub_path = Path(clip_dir, clip_sub)

        clip_start = clip["start"]
        clip_end = clip["end"]

        make_subclip(
            video_path, start_time=clip_start, end_time=clip_end, output_path=clip_path
        )
        logger.info(f"Source video: {video_path}")
        logger.info(f"Created clip: {clip_path}")

        gen_srt_file(request.subtitles, clip_sub_path, clip_start, clip_end)
        logger.info(f"Created subtitles: {clip_sub_path}")

        gen_srt_video(clip_path, clip_sub_path, output_path=clip_subbed_path)
        logger.info(f"Added subtitles to clip: {clip_subbed_path}")
        
        # os.remove(clip_path)
        # logger.info(f"Deleted base clip: {clip_path}\n")
        
        list_of_clips.append(
            {
                "clip_path": clip_subbed_path,
                "description": clip["description"],
                "clip_title": clip["clip_title"],
                "keywords": clip["keywords"],
            }
        )

    return list_of_clips

@app.post("/add_upload", operation_id="add_upload")
async def add_upload(request: AddUploadRequest):
    
    # Append to upload file
    with open(Path(UPLOAD_PATH, "uploads.csv"),mode='a',encoding='utf-8') as f:
        f.write(f'{request.title}, "{request.description}", "{request.upload_file}", "{" ".join(list(request.keywords))}" \n')
    
    return {"uploads_list": Path(UPLOAD_PATH, "uploads.csv").as_posix()}

#########################################################
# Main App
#########################################################

# But if you re-run the setup, the new endpoints will now be exposed.
mcp.setup_server()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8070)
