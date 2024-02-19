from moviepy.editor import VideoFileClip
import os
from misc_utils import audio_output_dir

def extract_audio_from_local_video(video_file_path, output_audio_path):
    video = VideoFileClip(video_file_path)
    audio = video.audio
    audio.write_audiofile(os.path.join(audio_output_dir, output_audio_path))