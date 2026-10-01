"""
Interroga l'endpoint pubblico di OpenRouter e, per ciascuno dei tuoi 5 slug,
riporta i segnali che dicono se il modello RAGIONA e se puoi CONTROLLARLO.

Non serve API key: /api/v1/models e' pubblico.

Cosa guardare:
- reasoning in supported_parameters  -> puoi passare il parametro reasoning
- internal_reasoning nel pricing > 0  -> il modello FATTURA token di ragionamento
  (cioe' ragiona di default: e' il tell piu' affidabile)
- se supported_parameters NON contiene reasoning ma internal_reasoning > 0,
  il modello ragiona e NON puoi spegnerlo -> candidato alla sostituzione.
"""
import requests

MY_MODELS = [
    "qwen/qwen3.7-plus",
    "google/gemini-3.1-flash-lite",
    "openai/gpt-4o-mini",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-v4-pro",
]

data = requests.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]
index = {m["id"]: m for m in data}

def flag(model_id):
    m = index.get(model_id)
    if m is None:
        return f"{model_id:<32} !! NON TROVATO (slug errato o ritirato)"

    params = set(m.get("supported_parameters") or [])
    can_control = "reasoning" in params
    internal = m.get("pricing", {}).get("internal_reasoning", "0")
    try:
        reasons = float(internal) > 0
    except (TypeError, ValueError):
        reasons = False

    if reasons and can_control:
        verdict = "RAGIONA, ma controllabile -> imposta reasoning off/minimal"
    elif reasons and not can_control:
        verdict = "RAGIONA e NON controllabile -> SOSTITUISCI"
    elif not reasons and can_control:
        verdict = "ok (reasoning opzionale, off di default)"
    else:
        verdict = "ok (non-reasoning)"

    ctx = m.get("context_length", "?")
    return (f"{model_id:<32} reasoning_param={'Y' if can_control else 'N'}  "
            f"internal_reasoning={internal:<10} ctx={ctx}\n"
            f"{'':<32} -> {verdict}")

print("=" * 78)
for mid in MY_MODELS:
    print(flag(mid))
    print("-" * 78)
