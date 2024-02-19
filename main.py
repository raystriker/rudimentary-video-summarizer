import time
from tqdm import tqdm
import os
import argparse
from misc_utils import transcription_output_dir, summary_output_dir
from transcription import transcribe_cuda, transcribe_whispercpp_cpu
from audio_processing import extract_audio_from_local_video
from youtube_downloading import download_youtube_audio
from text_summarization import summarize_text


def main(youtube_url, inference_engine, llm_host_ip):
    try:
        start_time = time.time()
        youtube_audio_file = download_youtube_audio(youtube_url)
        download_time = time.time() - start_time
        print(f"Audio download completed in {download_time:.2f} seconds.")

        transcription_filename = os.path.splitext(os.path.basename(youtube_audio_file))[0] + "_transcription.txt"

        start_time = time.time()
        if inference_engine != "cuda":
            youtube_transcribed_text = transcribe_whispercpp_cpu(youtube_audio_file, inference_engine)
        else:
            youtube_transcribed_text = transcribe_cuda(youtube_audio_file)
        transcription_time = time.time() - start_time

        if youtube_transcribed_text:
            with open(os.path.join(transcription_output_dir, transcription_filename), 'w') as f:
                f.write("YouTube Video Transcription:\n")
                f.write(youtube_transcribed_text)

        print(f"Transcription completed in {transcription_time:.2f} seconds.")
        print("Waiting for 10 seconds...")
        for _ in tqdm(range(10), desc="waiting..."):
            time.sleep(1)

        with open(os.path.join(transcription_output_dir, transcription_filename), 'r') as f:
            transcribed_text = f.read()

        start_time = time.time()
        summarized_text = summarize_text(transcribed_text, llm_host_ip)

        summarization_time = time.time() - start_time
        print(f"Summarization completed in {summarization_time:.2f} seconds.")

        summary_filename = os.path.splitext(os.path.basename(youtube_audio_file))[0] + "_summary.txt"
        with open(os.path.join(summary_output_dir, summary_filename), 'w') as f:
            f.write(summarized_text)

        print("Summary saved successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YouTube Video Transcription and Summarization")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--inference", help="Specify the inference engine",
                        choices=['openvino', 'x86openBLAS', 'avx2','aarch64', 'aarch64openBLAS', 'cuda'], default='avx2')
    parser.add_argument("--llm_host_ip", help="Specify the LLM host IP",default="100.88.35.147")
    args = parser.parse_args()

    main(args.url, args.inference, args.llm_host_ip)
