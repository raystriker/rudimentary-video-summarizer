import logging

from pytube import YouTube
from pytube.exceptions import PytubeError

from misc_utils import AUDIO_OUTPUT_DIR, sanitize_filename

logger = logging.getLogger(__name__)


def download_youtube_audio(youtube_url: str) -> str:
    """Download the audio track from a YouTube video.

    Returns the path to the downloaded MP3 file.
    Raises RuntimeError if the download fails.
    """
    try:
        yt = YouTube(youtube_url)
        stream = yt.streams.filter(only_audio=True).first()
        if stream is None:
            raise RuntimeError(f"No audio stream found for {youtube_url}")

        audio_filename = sanitize_filename(f"{yt.title}.mp3")
        stream.download(output_path=str(AUDIO_OUTPUT_DIR), filename=audio_filename)
        logger.info("Downloaded YouTube audio: %s", audio_filename)
        return str(AUDIO_OUTPUT_DIR / audio_filename)
    except PytubeError as exc:
        raise RuntimeError(f"Failed to download YouTube audio: {exc}") from exc
