# tools/list_groq_models.py
"""
List the chat models your Groq key can use (Oct 3, 2026). Needed because Groq retires models:
`llama-3.1-8b-instant` returned 404 model_not_found in step 16 (LLM-size study).

Uses the first key from GROQ_API_KEYS / GROQ_API_KEYS_FILE / GROQ_API_KEY (.env). Prints model ids
only - never a key. One request, no tokens used.
    python tools/list_groq_models.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv  # noqa: E402

load_dotenv()


def first_key():
    from utils.groq_key_pool import _split_keys
    keys = _split_keys(os.environ.get("GROQ_API_KEYS", ""))
    f = os.environ.get("GROQ_API_KEYS_FILE", "").strip()
    if f and Path(f).is_file():
        keys += _split_keys(Path(f).read_text())
    if not keys and os.environ.get("GROQ_API_KEY", "").strip():
        keys = [os.environ["GROQ_API_KEY"].strip()]
    return keys[0] if keys else None


def main():
    key = first_key()
    if not key:
        print("no Groq key found in the environment / .env")
        return
    from groq import Groq
    models = Groq(api_key=key, max_retries=0).models.list().data
    skip = ("whisper", "tts", "guard", "playai", "orpheus", "distil")
    rows = []
    for m in models:
        mid = getattr(m, "id", "")
        if any(s in mid.lower() for s in skip):
            continue
        rows.append((mid, getattr(m, "owned_by", ""), getattr(m, "context_window", ""),
                     getattr(m, "active", "")))
    print(f"{'model id':45s} {'owner':18s} {'context':>8s} active")
    for r in sorted(rows):
        print(f"{r[0]:45s} {str(r[1]):18s} {str(r[2]):>8s} {r[3]}")
    print("\nFor step 16 pick one model clearly SMALLER and one LARGER than openai/gpt-oss-20b if available.")


if __name__ == "__main__":
    main()
