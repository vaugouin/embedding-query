# Using Anthropic Claude with embedding-query

This document describes how to integrate Anthropic Claude (text-generation/assistant) workflows with this project. The repository primarily uses OpenAI for embeddings; Claude can be used for text generation, instruction-following, or assistant-style prompts alongside embedding-based search.

## Purpose

- Provide guidance for configuring and calling Anthropic Claude for text processing tasks.
- Show simple examples (curl and Python) to get started.

## Requirements

- An Anthropic API key. Set it in your environment as `ANTHROPIC_API_KEY`.
- Network access to Anthropic's API endpoint (e.g., `https://api.anthropic.com`).

## Environment

Add to your `.env` or environment:

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Optionally set an alternate base URL:

```bash
ANTHROPIC_API_BASE_URL=https://api.anthropic.com
```

## Quick curl Example

Replace `YOUR_PROMPT` with the text you want Claude to process.

```bash
curl -s -X POST "${ANTHROPIC_API_BASE_URL:-https://api.anthropic.com}/v1/complete" \
  -H "x-api-key: ${ANTHROPIC_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-2","prompt":"YOUR_PROMPT","max_tokens":300}'
```

Note: Anthropic's API surface and model names can change. Consult Anthropic's official docs for the latest endpoints and parameters.

## Python (requests) Example

```python
import os
import requests

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
BASE = os.environ.get("ANTHROPIC_API_BASE_URL", "https://api.anthropic.com")

def call_claude(prompt: str, model: str = "claude-2") -> str:
    url = f"{BASE}/v1/complete"
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    payload = {"model": model, "prompt": prompt, "max_tokens": 300}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

# Example usage
if __name__ == "__main__":
    resp = call_claude("Summarize the following text: ...")
    print(resp)
```

## Integration Notes

- This repository uses embedding vectors (OpenAI by default) for similarity search. Claude can be used as a complementary text-generation/assistant model for tasks such as:
  - Summarization of retrieved documents
  - Rewriting or expanding user queries
  - Generating human-readable answers from search results
- If you want to use Claude for embeddings (if supported), follow Anthropic's docs for embedding endpoints and adapt the embedding creation code accordingly.

## Privacy & Safety

- Be mindful of sending sensitive or personal data to third-party APIs.
- Review Anthropic's terms and data policies before sending production data.

## References

- Anthropic API documentation: https://www.anthropic.com/docs
