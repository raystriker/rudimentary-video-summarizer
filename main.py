import argparse
import logging
import os
import time

from misc_utils import TRANSCRIPTION_OUTPUT_DIR, SUMMARY_OUTPUT_DIR
from transcription import transcribe_cuda, transcribe_whispercpp_cpu
from youtube_downloading import download_youtube_audio
from text_summarization import summarize_text

logger = logging.getLogger(__name__)

DEFAULT_LLM_HOST = "127.0.0.1"


def main(youtube_url: str, inference_engine: str, llm_host_ip: str) -> None:
    """Download, transcribe, and summarize a YouTube video."""
    start = time.time()
    audio_path = download_youtube_audio(youtube_url)
    logger.info("Audio download completed in %.2f seconds", time.time() - start)

    start = time.time()
    if inference_engine == "cuda":
        transcribed_text = transcribe_cuda(audio_path)
    else:
        transcribed_text = transcribe_whispercpp_cpu(audio_path, inference_engine)
    logger.info("Transcription completed in %.2f seconds", time.time() - start)

    if not transcribed_text:
        logger.error("Transcription produced no output — aborting")
        return

    # Save transcription
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    transcription_file = TRANSCRIPTION_OUTPUT_DIR / f"{stem}_transcription.txt"
    transcription_file.write_text(
        f"YouTube Video Transcription:\n{transcribed_text}", encoding="utf-8"
    )

    # Summarize
    start = time.time()
    summary = summarize_text(transcription_file.read_text(encoding="utf-8"), llm_host_ip)
    logger.info("Summarization completed in %.2f seconds", time.time() - start)

    summary_file = SUMMARY_OUTPUT_DIR / f"{stem}_summary.txt"
    summary_file.write_text(summary, encoding="utf-8")
    logger.info("Summary saved to %s", summary_file)


def cli() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="YouTube Video Transcription and Summarization",
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--inference",
        help="Inference engine to use for transcription",
        choices=["openvino", "x86openBLAS", "avx2", "aarch64", "aarch64openBLAS", "cuda"],
        default="avx2",
    )
    parser.add_argument(
        "--llm-host-ip",
        help="LLM server IP address (default: $LLM_HOST_IP or 127.0.0.1)",
        default=os.environ.get("LLM_HOST_IP", DEFAULT_LLM_HOST),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    main(args.url, args.inference, args.llm_host_ip)


if __name__ == "__main__":
    cli()
