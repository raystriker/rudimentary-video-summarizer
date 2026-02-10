import logging
import subprocess
from pathlib import Path

from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

from misc_utils import BASE_DIR

logger = logging.getLogger(__name__)


def transcribe_cuda(audio_file: str) -> str:
    """Transcribe audio using the Whisper model on a CUDA GPU."""
    import torch

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
        torch_dtype=torch.float16,
        device="cuda:0",
        model_kwargs=(
            {"attn_implementation": "flash_attention_2"}
            if is_flash_attn_2_available()
            else {"attn_implementation": "sdpa"}
        ),
    )

    try:
        outputs = pipe(
            audio_file,
            chunk_length_s=30,
            batch_size=24,
            return_timestamps=True,
        )
        return outputs["text"]
    except Exception:
        logger.exception("Error during CUDA transcription")
        return ""
    finally:
        del pipe
        torch.cuda.empty_cache()


def _convert_mp3_to_wav(input_mp3: str, output_wav: str) -> None:
    """Convert an MP3 file to 16 kHz mono WAV using FFmpeg."""
    command = [
        "ffmpeg",
        "-i", str(BASE_DIR / input_mp3),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(BASE_DIR / output_wav),
        "-y",
    ]
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logger.info("MP3 to WAV conversion successful")
    except subprocess.CalledProcessError as exc:
        logger.error("FFmpeg conversion failed: %s", exc.stderr.decode())
        raise


_ENGINE_CONFIG: dict[str, tuple[str, str]] = {
    "openvino": ("whisperinference/openvino/build/bin/main", "6"),
    "x86openblas": ("whisperinference/x86openBLAS/main", "6"),
    "avx2": ("whisperinference/avx2/main", "6"),
    "aarch64": ("whisperinference/aarch64/main", "4"),
    "aarch64openblas": ("whisperinference/aarch64openBLAS/main", "4"),
}


def transcribe_whispercpp_cpu(audio_file: str, inference_engine: str) -> str:
    """Transcribe audio using whisper.cpp on CPU with the given inference engine.

    Returns the transcribed text, or an empty string on failure.
    """
    wav_file = Path(audio_file).with_suffix(".wav").as_posix()

    _convert_mp3_to_wav(audio_file, wav_file)

    engine_key = inference_engine.lower()
    if engine_key not in _ENGINE_CONFIG:
        logger.error("Unknown inference engine: %s", inference_engine)
        return ""

    rel_executable, thread_count = _ENGINE_CONFIG[engine_key]
    executable = str(BASE_DIR / rel_executable)
    model_path = str(BASE_DIR / "models" / "ggml-base.en.bin")

    output_path = (
        Path(wav_file).with_suffix("").as_posix().replace("audio_files", "transcriptions")
        + "_transcription"
    )

    command = [
        executable,
        "-t", thread_count,
        "-m", model_path,
        "-f", wav_file,
        "-otxt",
        "-of", output_path,
    ]

    logger.info("Transcribing with %s engine", inference_engine)
    try:
        result = subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logger.debug("whisper.cpp output: %s", result.stdout.decode())
    except subprocess.CalledProcessError as exc:
        logger.error("whisper.cpp failed: %s", exc.stderr.decode())
        return ""

    # whisper.cpp writes output to <output_path>.txt
    output_file = Path(f"{output_path}.txt")
    if output_file.exists():
        return output_file.read_text(encoding="utf-8")

    logger.warning("Expected transcription file not found: %s", output_file)
    return ""
