"""
Per ciascuno dei 5 modelli elenca i provider (endpoint) disponibili su
OpenRouter, con slug esatto e quantizzazione. Serve a compilare a mano
PROVIDER_POLICY in utils.py: scegli UN provider per modello e copiane lo slug.

Nessuna API key necessaria.
"""
import requests

MY_MODELS = [
    "qwen/qwen3.7-plus",
    "google/gemini-3.1-flash-lite",
    "openai/gpt-4o-mini",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-v4-pro",
]

for model_id in MY_MODELS:
    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
    try:
        data = requests.get(url, timeout=30).json()["data"]
    except Exception as e:
        print(f"{model_id}: errore ({e})\n")
        continue

    print("=" * 70)
    print(model_id)
    print("=" * 70)
    for ep in data.get("endpoints", []):
        slug = ep.get("provider_name") or ep.get("name")
        tag  = ep.get("tag")            # a volte lo slug utile e' qui (es. 'deepinfra/turbo')
        quant = ep.get("quantization", "?")
        ctx   = ep.get("context_length", "?")
        print(f"  provider='{slug}'  tag='{tag}'  quant={quant}  ctx={ctx}")
    print()
