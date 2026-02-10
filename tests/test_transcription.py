from unittest.mock import patch, MagicMock
from pathlib import Path

from transcription import _ENGINE_CONFIG, transcribe_whispercpp_cpu


class TestEngineConfig:
    def test_all_engines_present(self):
        expected = {"openvino", "x86openblas", "avx2", "aarch64", "aarch64openblas"}
        assert set(_ENGINE_CONFIG.keys()) == expected

    def test_each_engine_has_path_and_threads(self):
        for key, (path, threads) in _ENGINE_CONFIG.items():
            assert isinstance(path, str) and path, f"{key} has empty path"
            assert threads in ("4", "6"), f"{key} has unexpected thread count: {threads}"


class TestTranscribeWhispercppCpu:
    @patch("transcription.subprocess.run")
    @patch("transcription._convert_mp3_to_wav")
    def test_returns_empty_for_unknown_engine(self, mock_convert, mock_run):
        result = transcribe_whispercpp_cpu("audio_files/test.mp3", "nonexistent_engine")
        assert result == ""
        mock_run.assert_not_called()

    @patch("transcription.subprocess.run")
    @patch("transcription._convert_mp3_to_wav")
    def test_returns_text_on_success(self, mock_convert, mock_run, tmp_path):
        # Create a fake transcription output file where the function expects it
        audio_file = "audio_files/test.mp3"
        output_path = Path("audio_files/test").as_posix().replace(
            "audio_files", "transcriptions"
        ) + "_transcription"
        output_file = Path(f"{output_path}.txt")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("hello world", encoding="utf-8")

        mock_run.return_value = MagicMock(stdout=b"", stderr=b"")

        try:
            result = transcribe_whispercpp_cpu(audio_file, "avx2")
            assert result == "hello world"
        finally:
            output_file.unlink(missing_ok=True)
