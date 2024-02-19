from pytube import YouTube
import os
from misc_utils import audio_output_dir, sanitize_filename


def download_youtube_audio(youtube_url):
    try:
        yt = YouTube(youtube_url)
        video = yt.streams.filter(only_audio=True).first()
        audio_filename = sanitize_filename(f"{yt.title}.mp3")
        video.download(output_path=audio_output_dir, filename=audio_filename)
        print(f"Downloaded YouTube audio: {audio_filename}")
        return os.path.join(audio_output_dir, audio_filename)
    except Exception as e:
        print(f"Error downloading YouTube audio: {e}")
        return None
