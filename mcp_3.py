import csv
import os
from pathlib import Path

# Path configuration (make sure these match your setup)
BASE_DIR = Path(__file__).parent.absolute()
CSV_FILE = Path(BASE_DIR, "assets", "tasks", "video_gen.csv")


def get_pending():
    """
    Returns a list of dictionaries representing videos with STATUS not 'COMPLETED'
    Each dictionary contains all the video data for the pending item.
    """
    pending_videos = []

    if not CSV_FILE.exists():
        return pending_videos

    with open(CSV_FILE, mode="r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row["STATUS"].upper() != "COMPLETED":
                pending_videos.append(row)

    return pending_videos


def set_status(uuid, status):
    """
    Updates the status of a video entry in the CSV file.

    Args:
        uuid (str): The UUID of the video to update
        status (str): The new status value

    Returns:
        bool: True if update was successful, False otherwise
    """
    if not CSV_FILE.exists():
        return False

    # Read all rows first
    all_rows = []
    with open(CSV_FILE, mode="r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            all_rows.append(row)

    # Find and update the matching UUID
    updated = False
    for row in all_rows:
        if row["UUID"] == uuid:
            row["STATUS"] = status
            updated = True
            break

    if not updated:
        return False

    # Write all rows back to the file
    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = all_rows[0].keys() if all_rows else []
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    return True


def set_upload_result(uuid, video_url):
    """
    Updates a video entry in the CSV file to mark it as completed
    and sets the new video URL (presumably after uploading somewhere).

    Args:
        uuid (str): The UUID of the video to update
        video_url (str): The new video URL (typically after upload)

    Returns:
        bool: True if update was successful, False otherwise
    """
    if not CSV_FILE.exists():
        return False

    # Read all rows first
    all_rows = []
    with open(CSV_FILE, mode="r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            all_rows.append(row)

    # Find and update the matching UUID
    updated = False
    for row in all_rows:
        if row["UUID"] == uuid:
            row["VIDEO_URL"] = video_url
            row["STATUS"] = "COMPLETED"
            updated = True
            break

    if not updated:
        return False

    # Write all rows back to the file
    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = all_rows[0].keys() if all_rows else []
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    return True
