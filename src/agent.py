from openai import OpenAI


def build_client(base_url, api_key):

    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )


def call_llm(
    client,
    model,
    messages,
    n_ctx,
    max_tokens,
    temperature,
    tools=None,
):

    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=None,
        extra_body={
            "options": {
                "num_ctx": 8192
            }
        }
    )