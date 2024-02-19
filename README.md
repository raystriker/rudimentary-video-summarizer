# Rudimentary Video Summarizer

## Introduction
The Rudimentary Video Summarizer is a Python-based tool designed to streamline the process of extracting, transcribing, and summarizing audio content from YouTube videos. Leveraging advanced machine learning models and audio processing techniques, this tool offers a quick and efficient way to get the essence of video content without the need for manual transcription and summarization.

## Features
- **YouTube Audio Downloading**: Downloads audio from a specified YouTube video URL.
- **Audio Transcription**: Transcribes audio content using various inference engines, including CUDA for NVIDIA GPUs, as well as CPU-based options like OpenVINO, x86 OpenBLAS, AVX2, AArch64, and AArch64 OpenBLAS.
- **Text Summarization**: Summarizes the transcribed text, providing a concise version of the video's content.
- **Flexible Inference Engine Selection**: Allows users to choose the inference engine that best fits their hardware setup.
- **Local Large Language Model Support**: Integrates with a local LLM server for text summarization, ensuring fast and private processing.

## Installation

### Prerequisites
- Python 3.6 or later
- CUDA-compatible GPU (for CUDA inference engine option)

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/raystriker/rudimentary-video-summarizer.git
   ```
2. Navigate to the project directory:
   ```
   cd rudimentary-video-summarizer
   ```
3. Install the required Python dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set up Whisper.cpp for audio transcription:
   - Clone the Whisper.cpp repository: `git clone https://github.com/ggerganov/whisper.cpp.git`
   - Follow the build instructions in the Whisper.cpp README to compile the project.
   - Place the compiled inference engine binaries in the respective directories under `/whisperinference` (e.g., `/whisperinference/aarch64`, `/whisperinference/openvino`, `/whisperinference/avx2`, etc.).
   - Download the Whisper model files and place them in the `/models` directory.

## Usage

To use the Rudimentary Video Summarizer, follow these steps:

1. Run the script with the necessary arguments:
   ```
   python main.py [YouTube Video URL] --inference [Inference Engine] --llm_host_ip [LLM Host IP]
   ```
   
   Example:
   ```
   python main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --inference cuda --llm_host_ip 100.88.35.147
   ```

2. The script will download the audio, transcribe it, wait for 10 seconds (with a progress bar), and then summarize the content. The transcription and summary files will be saved in the specified output directories.

## License
This project is made available under the terms of the MIT License, with the exception of the CUDA-based transcription functionality which leverages "Insanely Fast Whisper," licensed under the Apache License 2.0.

### MIT License
The core of Rudimentary Video Summarizer, excluding its dependencies on external libraries or frameworks not developed as part of this project, is licensed under the MIT License. The MIT License is a permissive free software license originating at the Massachusetts Institute of Technology (MIT). It allows for commercial use, modification, distribution, private use, and sublicensing.

### Apache License 2.0
"Insanely Fast Whisper" is used under the Apache License 2.0, which is a free, open-source license administered by the Apache Software Foundation. The Apache License allows for free use, distribution, and modification of the software, provided that the original copyright and license notices are preserved, and changes to the original software are clearly marked.

For more details on the licenses, please see the LICENSE files in the respective repositories or the following summaries:
- [MIT License](https://opensource.org/licenses/MIT)
- [Apache License 2.0](https://opensource.org/licenses/Apache-2.0)