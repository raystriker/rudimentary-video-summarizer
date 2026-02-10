import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

AUDIO_OUTPUT_DIR = BASE_DIR / "audio_files"
TRANSCRIPTION_OUTPUT_DIR = BASE_DIR / "transcriptions"
SUMMARY_OUTPUT_DIR = BASE_DIR / "summaries"

AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)
TRANSCRIPTION_OUTPUT_DIR.mkdir(exist_ok=True)
SUMMARY_OUTPUT_DIR.mkdir(exist_ok=True)


def sanitize_filename(filename: str) -> str:
    """Remove invalid characters and replace spaces with underscores."""
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    filename = re.sub(r"\s+", "_", filename)
    return filename
