import os
import re

def sanitize_filename(filename):
    # Remove invalid characters and replace spaces
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    filename = re.sub(r'\s+', '_', filename)
    return filename

# Define directories for storing audio files, transcriptions, and summaries
audio_output_dir = 'audio_files'
transcription_output_dir = 'transcriptions'
summary_output_dir = 'summaries'
os.makedirs(audio_output_dir, exist_ok=True)
os.makedirs(transcription_output_dir, exist_ok=True)
os.makedirs(summary_output_dir, exist_ok=True)