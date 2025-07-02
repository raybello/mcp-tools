import csv
import uuid
import os
from pathlib import Path
from pytube import Playlist
from tools.mcp_yt import ytDownloader

# Get the base directory where this script is located
base_dir = Path(__file__).parent.absolute()

# Define absolute paths
assets_dir = Path(base_dir, "assets")
tasks_dir = Path(assets_dir, "tasks")
subtitles_dir = Path(assets_dir, "subtitles")

# Create directories if they don't exist
tasks_dir.mkdir(parents=True, exist_ok=True)
subtitles_dir.mkdir(parents=True, exist_ok=True)

playlist_url = "https://youtube.com/playlist?list=PL-o4ZQ01xkrK1ERV6f9ZugWIWwqcvir0i&si=YeNtL84_Kjyp4kCG"
# playlist_url = "https://youtube.com/playlist?list=PL-o4ZQ01xkrL3aenYcHq2zjJ0pSgm2gHD&si=i7D80glA5Obj-lBp"

yt = ytDownloader()
p = Playlist(playlist_url)

print(f"Playlist Title: {p.title}")
print(f"Number of videos in playlist: {len(p.video_urls)}")

# Absolute path for CSV file
csv_filename = Path(tasks_dir, "video_gen.csv")

print(f"\nSaving video list to: {csv_filename}")

with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write header with the new column order
    csv_writer.writerow(["UUID", "VIDEO_URL", "VIDEO_TITLE", "SRT_FILE_PATH", "STATUS"])

    for video in p.videos:
        title = yt.get_video_info(video.watch_url)["title"]
        video_uuid = str(uuid.uuid4())[:8]
        status = "pending"

        # Create absolute paths for SRT file
        srt_filename = f"{video_uuid}.srt"
        srt_file_path = Path(subtitles_dir, srt_filename).absolute()

        # Write row with the new column order
        csv_writer.writerow(
            [video_uuid, video.watch_url, title, str(srt_file_path), status]
        )
        print(
            f"Added to CSV: {video_uuid}, {video.watch_url}, {title}, {srt_file_path}, {status}"
        )

print(
    f"\nCSV file '{csv_filename}' created successfully with {len(p.video_urls)} entries."
)
