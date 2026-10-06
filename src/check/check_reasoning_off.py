"""
Per ciascun modello prova a disabilitare il reasoning con una chiamata minima.
Dice quale endpoint RIFIUTA enabled:False (reasoning obbligatorio).
Richiede OPENROUTER_API_KEY nell'ambiente.
"""
import os
from openai import OpenAI

client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"])

MODELS = [
    "qwen/qwen3.7-plus",
    "google/gemini-3.1-flash-lite",
    "openai/gpt-4o-mini",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-v4-pro",
]

for m in MODELS:
    try:
        r = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "Reply with the word OK."}],
            temperature=0,
            extra_body={"reasoning": {"enabled": False}},
        )
        det = getattr(r.usage, "completion_tokens_details", None)
        rt = getattr(det, "reasoning_tokens", 0) or 0
        print(f"[OK ] {m:<32} enabled:False accettato  reasoning_tokens={rt}")
    except Exception as e:
        msg = str(e).split("message':")[-1][:80]
        print(f"[!! ] {m:<32} RIFIUTA disabilitazione -> {msg}")
