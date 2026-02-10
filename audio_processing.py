import logging

from moviepy.editor import VideoFileClip

from misc_utils import AUDIO_OUTPUT_DIR

logger = logging.getLogger(__name__)


def extract_audio_from_local_video(video_file_path: str, output_audio_path: str) -> str:
    """Extract the audio track from a local video file.

    Returns the path to the extracted audio file.
    """
    output = AUDIO_OUTPUT_DIR / output_audio_path
    video = VideoFileClip(video_file_path)
    try:
        video.audio.write_audiofile(str(output))
        logger.info("Extracted audio to %s", output)
        return str(output)
    finally:
        video.close()
