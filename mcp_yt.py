import yt_dlp
import os
import tempfile
import subprocess
from datetime import datetime
import sys
from pathlib import Path


class ytDownloader:
    """
    A class to handle YouTube video downloads using yt-dlp.
    Provides methods to download videos and return file paths or file data.
    """

    def __init__(self, output_path=None):
        """
        Initialize the ytDownloader.

        Args:
            output_path (str): Directory to save downloads. If None, uses temp directory.
        """
        self.output_path = output_path or tempfile.mkdtemp()
        self.ffmpeg_available = self._check_ffmpeg()

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def _check_ffmpeg(self):
        """
        Check if FFmpeg is installed and accessible.
        Returns True if FFmpeg is available, False otherwise.
        """
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def _format_size(self, bytes_size):
        """
        Convert bytes to human readable format.
        """
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_size < 1024:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.2f} GB"

    def _progress_hook(self, d):
        """
        Display download progress.
        """
        if d["status"] == "downloading":
            downloaded = d.get("downloaded_bytes", 0)
            total = d.get("total_bytes", 0) or d.get("total_bytes_estimate", 0)

            if total:
                percentage = (downloaded / total) * 100
                speed = d.get("speed", 0)
                speed_str = self._format_size(speed) + "/s" if speed else "N/A"

                progress = f"Progress: {percentage:.1f}% | Speed: {speed_str}"
                print(progress)

    def _get_best_format(self, target_height=720):
        """
        Select the best video format based on desired quality and FFmpeg availability.
        """
        if self.ffmpeg_available:
            return f"bestvideo[height<={target_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={target_height}][ext=mp4]/best"
        else:
            return f"best[height<={target_height}][ext=mp4]/best[ext=mp4]/best"

    def get_video_info(self, url):
        """
        Get video information without downloading.

        Args:
            url (str): YouTube video URL

        Returns:
            dict: Video information
        """
        ydl_opts = {"quiet": True, "no_warnings": True}

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    "title": info.get("title", "Unknown"),
                    "duration": info.get("duration", 0),
                    "uploader": info.get("uploader", "Unknown"),
                    "view_count": info.get("view_count", 0),
                    "description": info.get("description", ""),
                    "upload_date": info.get("upload_date", ""),
                    "thumbnail": info.get("thumbnail", ""),
                }
        except Exception as e:
            raise Exception(f"Failed to get video info: {str(e)}")

    def download_video(self, url, quality="720p", audio_only=False):
        """
        Download a YouTube video.

        Args:
            url (str): YouTube video URL
            quality (str): Preferred video quality (e.g., '720p', '1080p')
            audio_only (bool): If True, download only audio

        Returns:
            dict: Download result with file path and metadata
        """
        try:
            # Configure output filename template
            if audio_only:
                outtmpl = os.path.join(self.output_path, "%(id)s.%(ext)s")
                format_selector = "bestaudio[ext=m4a]/bestaudio/best"
            else:
                outtmpl = os.path.join(self.output_path, "%(id)s.%(ext)s")
                height = int(quality.replace("p", ""))
                format_selector = self._get_best_format(height)

            ydl_opts = {
                "format": format_selector,
                "outtmpl": outtmpl,
                "progress_hooks": [self._progress_hook],
                "quiet": False,
                "no_warnings": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info first
                info = ydl.extract_info(url, download=False)

                # Download the video
                ydl.download([url])

                # Find the downloaded file
                title = info.get("title", "video")
                # Clean title for filename
                safe_title = "".join(
                    c for c in title if c.isalnum() or c in (" ", "-", "_")
                ).rstrip()


                # Find the actual downloaded file
                downloaded_file = None
                for file in os.listdir(self.output_path):
                    if (
                        safe_title.lower() in file.lower()
                        or title.lower() in file.lower()
                    ):
                        downloaded_file = os.path.join(self.output_path, file)
                        break

                # If we can't find by title, get the most recent file
                if not downloaded_file:
                    files = [
                        os.path.join(self.output_path, f)
                        for f in os.listdir(self.output_path)
                    ]
                    if files:
                        downloaded_file = max(files, key=os.path.getctime)

                if downloaded_file and os.path.exists(downloaded_file):
                    file_size = os.path.getsize(downloaded_file)
                    return {
                        "success": True,
                        "file_path": downloaded_file,
                        "title": info.get("title", "Unknown"),
                        "duration": info.get("duration", 0),
                        "file_size": file_size,
                        "file_size_formatted": self._format_size(file_size),
                        "format": "audio" if audio_only else quality,
                    }
                else:
                    raise Exception("Downloaded file not found")

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_file_data(self, file_path):
        """
        Read file data as bytes.

        Args:
            file_path (str): Path to the file

        Returns:
            bytes: File content
        """
        try:
            with open(file_path, "rb") as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Failed to read file: {str(e)}")

    def cleanup_file(self, file_path):
        """
        Remove a downloaded file.

        Args:
            file_path (str): Path to the file to remove
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except Exception:
            pass
        return False

    def cleanup_all(self):
        """
        Remove all files in the output directory.
        """
        try:
            for file in os.listdir(self.output_path):
                file_path = os.path.join(self.output_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            return True
        except Exception:
            return False
