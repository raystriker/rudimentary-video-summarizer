import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_LLM_PORT = "1234"
SYSTEM_PROMPT = (
    "You are an expert script summarizer who has a knack for important "
    "details related to the main content of the script."
)


def summarize_text(text: str, llm_host_ip: str) -> str:
    """Send transcribed text to a local LLM server and return the summary."""
    port = os.environ.get("LLM_PORT", DEFAULT_LLM_PORT)
    base_url = f"http://{llm_host_ip}:{port}/v1"

    with OpenAI(base_url=base_url, api_key="not-needed") as client:
        completion = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Here is the script of the video: {text}"},
            ],
            temperature=0.7,
        )

    return completion.choices[0].message.content
