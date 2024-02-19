import os
import pprint as pp
import subprocess

from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

def transcribe_cuda(audio_file):
    import torch
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
        torch_dtype=torch.float16,
        device="cuda:0",  # or mps for Mac devices
        model_kwargs={
            "attn_implementation": "flash_attention_2"
        } if is_flash_attn_2_available() else {
            "attn_implementation": "sdpa"
        },
    )

    try:
        outputs = pipe(
            audio_file,
            chunk_length_s=30,
            batch_size=24,
            return_timestamps=True,
        )
        pp.pprint(outputs)
        actual_output = outputs['text']

        del pipe
        torch.cuda.empty_cache()

        return actual_output
    except Exception as e:
        print(f"Error in transcription: {e}")
        del pipe
        torch.cuda.empty_cache()
        return ""


def transcribe_whispercpp_cpu(audio_file, inference_engine):
    input_file = audio_file
    wav_file = audio_file.split('.')[0] + '.wav'

    print(input_file, wav_file)

    # Convert audio mp3 to wav
    def convert_mp3_to_wav_ffmpeg(input_mp3_path, output_wav_path):
        abs_path_project_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            # Construct the ffmpeg command to convert MP3 to WAV
            command = [
                'ffmpeg',
                '-i', abs_path_project_dir + '/' + input_mp3_path,  # Input file
                '-ar', '16000',  # Set audio sample rate to 16000 Hz
                '-ac', '1',  # Set audio channels to 1 (mono)
                '-c:a', 'pcm_s16le',  # Set audio codec to PCM 16-bit little-endian
                abs_path_project_dir + '/' + output_wav_path,  # Output file
                '-y'  # Overwrite output file without asking
            ]

            # Execute the command
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"Conversion successful.")
        except subprocess.CalledProcessError as e:
            print(f"Error during conversion: {e.stderr.decode()}")

    convert_mp3_to_wav_ffmpeg(input_file, wav_file)

    # Transcribe using whispercpp with various backend inference engines
    # Command components
    abs_path_project_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"This is using {inference_engine} inference engine.")

    match inference_engine.lower():
        case "openvino":
            executable = f'{abs_path_project_dir}/whisperinference/openvino/build/bin/main'
            thread_count = '6'
        case "x86openblas":
            executable = f'{abs_path_project_dir}/whisperinference/x86openBLAS/main'
            thread_count = '6'
        case "avx2":
            executable = f'{abs_path_project_dir}/whisperinference/avx2/main'
            thread_count = '6'
        case "aarch64":
            executable = f'{abs_path_project_dir}/whisperinference/aarch64/main'
            thread_count = '4'
        case "aarch64openBLAS":
            executable = f'{abs_path_project_dir}/whisperinference/aarch64openBLAS/main'
            thread_count = '4'

    threads = '-t'
    # thread_count = '6'
    model_flag = '-m'
    model_path = f'{os.path.dirname(os.path.abspath(__file__))}/models/ggml-base.en.bin'
    file_flag = '-f'
    audio_file = wav_file
    output_type = '-otxt'
    output_flag = '-of'
    output_path = wav_file.split('.')[0].replace('audio_files', 'transcriptions') + "_transcription"

    # Constructing the command
    command = [
        executable,
        threads, thread_count,
        model_flag, model_path,
        file_flag, audio_file,
        output_type,
        output_flag, output_path
    ]

    # Executing the command
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Command output:", result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e.stderr.decode())

    print("Transcription completed.")