from unittest.mock import patch, MagicMock

from text_summarization import summarize_text, SYSTEM_PROMPT


class TestSummarizeText:
    @patch("text_summarization.OpenAI")
    def test_returns_summary_content(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_openai_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_choice = MagicMock()
        mock_choice.message.content = "This is a summary."
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        result = summarize_text("some transcript", "127.0.0.1")
        assert result == "This is a summary."

    @patch("text_summarization.OpenAI")
    def test_sends_correct_messages(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_openai_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_choice = MagicMock()
        mock_choice.message.content = "summary"
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        summarize_text("my text", "10.0.0.1")

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert "my text" in messages[1]["content"]

    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 0
