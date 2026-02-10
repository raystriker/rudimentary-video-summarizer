# Rudimentary Video Summarizer

A Python CLI tool that downloads YouTube videos, transcribes the audio using [Whisper](https://github.com/openai/whisper), and summarizes the transcript with a local LLM.

## Features

- **YouTube audio download** via pytube
- **Audio transcription** — CUDA (GPU) via HuggingFace Transformers, or CPU via [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with multiple backend options (OpenVINO, x86 OpenBLAS, AVX2, AArch64)
- **Text summarization** using any OpenAI-compatible local LLM server
- **Local video support** — extract and process audio from local files

## Requirements

- Python 3.10+
- FFmpeg installed and on `PATH`
- A local LLM server exposing an OpenAI-compatible API (e.g. [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.com/), [vLLM](https://github.com/vllm-project/vllm))
- *(Optional)* CUDA-compatible GPU for GPU-accelerated transcription

## Installation

```bash
git clone https://github.com/raystriker/rudimentary-video-summarizer.git
cd rudimentary-video-summarizer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For GPU support, also install the CUDA packages:

```bash
pip install ".[gpu]"
```

### whisper.cpp setup (CPU transcription only)

1. Clone and build [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
2. Place the compiled binaries under `whisperinference/<backend>/main` (e.g. `whisperinference/avx2/main`)
3. Download the Whisper model to `models/ggml-base.en.bin`

## Usage

```bash
python main.py <YOUTUBE_URL> [--inference ENGINE] [--llm-host-ip IP] [-v]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--inference` | `avx2` | Transcription backend: `openvino`, `x86openBLAS`, `avx2`, `aarch64`, `aarch64openBLAS`, `cuda` |
| `--llm-host-ip` | `$LLM_HOST_IP` or `127.0.0.1` | IP address of the LLM server |
| `-v` / `--verbose` | off | Enable debug logging |

### Environment variables

| Variable | Description |
|----------|-------------|
| `LLM_HOST_IP` | Default LLM server IP (overridden by `--llm-host-ip`) |
| `LLM_PORT` | LLM server port (default: `1234`) |

### Example

```bash
# Transcribe on CPU (AVX2) and summarize via local LLM on localhost
python main.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Use CUDA GPU and a remote LLM server
python main.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --inference cuda --llm-host-ip 192.168.1.50

# With debug logging
python main.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -v
```

Output files are saved to `transcriptions/` and `summaries/`.

## Development

```bash
pip install ".[dev]"
pytest
ruff check .
```

## License

MIT License — see [LICENSE](https://opensource.org/licenses/MIT).

The CUDA transcription path uses [Insanely Fast Whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) patterns, licensed under Apache 2.0.
