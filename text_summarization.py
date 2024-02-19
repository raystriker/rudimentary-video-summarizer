from openai import OpenAI


def summarize_text(text, llm_host_ip):
    client = OpenAI(base_url=f"http://{llm_host_ip}:1234/v1", api_key="not-needed")

    completion = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": "You are an expert script summarizer who has a knack for important "
                                          "details related to the main content of the script."},
            {"role": "user", "content": f"Here is the script of the video: {text}"}
        ],
        temperature=0.7,
    )
    client.close()
    summary = completion.choices[0].message.content
    return summary

