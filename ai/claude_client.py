import anthropic
from config import ANTHROPIC_API_KEY, INSTITUTIONAL_SYSTEM_PROMPT, DEFAULT_MODEL, MAX_TOKENS


def generate_analysis(json_payload: str, model: str = DEFAULT_MODEL) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    with client.messages.stream(
        model=model,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": INSTITUTIONAL_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    f"<data>\n{json_payload}\n</data>\n\n"
                    "Generate the full 7-section institutional analysis report based on the provided data. "
                    "Follow the exact section format defined in your instructions."
                ),
            }
        ],
    ) as stream:
        message = stream.get_final_message()

    # Extract text content from response blocks
    report_text = ""
    for block in message.content:
        if hasattr(block, "text"):
            report_text += block.text

    usage = message.usage
    cache_info = ""
    if hasattr(usage, "cache_read_input_tokens") and usage.cache_read_input_tokens:
        cache_info = f" [cache hit: {usage.cache_read_input_tokens:,} tokens saved]"
    elif hasattr(usage, "cache_creation_input_tokens") and usage.cache_creation_input_tokens:
        cache_info = f" [cache created: {usage.cache_creation_input_tokens:,} tokens cached]"

    return report_text, cache_info
