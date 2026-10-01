import os
import time
import shutil

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    timeout=60,
)

# number of retries for TEMPORARY errors (rate limit, 5xx, timeout)
# systematic errors (400, 404) are not retried: see is_retryable
MAX_RETRIES = 4
RETRY_SLEEP = 10  # secondi between tries

'''
 ---------------------------------------------------------------------------
 Routing policy for models 

  Compiles "only" with the provider identified in check_provider.py
  If you don't want to lock the provider, set only to "None", the rest of the 
  policy will still work with the provider not locked.

  allow_fallbacks=False   -> Forced to not fallback in order to enforce the same provider for the experiment and not use other endpoints, you can set it to true or delete it if not needed
 ---------------------------------------------------------------------------
'''

PROVIDER_POLICY = {
    "qwen/qwen3.7-plus": {
        "only": ["alibaba"],                 
        "allow_fallbacks": False,
    },
    "google/gemini-3.1-flash-lite": {
        "only": ["Google"],                 
        "allow_fallbacks": False,
    },
    "openai/gpt-4o-mini": {
        "only": ["OpenAI"],
        "allow_fallbacks": False,
    },
    "meta-llama/llama-4-maverick": {
        "only": ["DeepInfra"],                
        "allow_fallbacks": False,
    },
    "deepseek/deepseek-v4-pro": {
        "only": ["StreamLake"],                 
        "allow_fallbacks": False,
    },
}

REASONING_EFFORT = None   # default None = off; "low"/"medium"/"high" = on 



#llama does not support reasoning parameter
NO_REASONING_PARAM = {"meta-llama/llama-4-maverick"}


def clean_policy(model_name):
    # Removes empty keys before sending the request to OpenRouter (e.g. only=None)
    raw = PROVIDER_POLICY.get(model_name, {})
    return {k: v for k, v in raw.items() if v is not None}


def extra_body(model_name):
    body = {"provider": clean_policy(model_name)}
    if model_name not in NO_REASONING_PARAM:
        if REASONING_EFFORT is None:
            body["reasoning"] = {"enabled": False}      # spegnimento esplicito (come prima)
        else:
            body["reasoning"] = {"effort": REASONING_EFFORT}
    return body


def log_usage(response):
    # Extracts tokens and provider to check costs and validity
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "provider": getattr(response, "provider", None),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
    }


def is_retryable(e):
    """
     True only for TEMPORARY errors, where you can retry after some time (rate limits, 5xx, timeouts).
     SYSTEMATIC errors (400 and 404) are not retriable.
    """
    code = getattr(e, "status_code", None)
    if code in (400, 404, 401, 403):
        return False           # malformed / no endpoint / auth error
    if code in (429, 500, 502, 503, 504):
        return True            # rate limit / server maybe reachable after some time
    return code is None  #no code but a retryable error


def get_completion_from_router(messages, model_name, temperature=0):
    for retry in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                extra_body=extra_body(model_name),
            )
            if not getattr(response, "choices", None):
                raise ValueError(f"Answer with no 'choices': {response}")
            return response.choices[0].message.content
        except Exception as e:
            if not is_retryable(e):
                print(f"Error not retryable for {model_name}: {e}")
                break
            print(f"Error: {e}\nRetry {retry + 1}/{MAX_RETRIES}...")
            time.sleep(RETRY_SLEEP)
    print(f"WARNING: completion failed for {model_name}.")
    return "Cannot update memory: connection error."


def get_completion_from_router_json(messages, model_name, temperature=0):
    # token returns are for logging and cost analysis
    for retry in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                extra_body=extra_body(model_name),
            )
            if not getattr(response, "choices", None):
                raise ValueError(f"Answer with no 'choices': {response}")
            return response.choices[0].message.content, log_usage(response)
        except Exception as e:
            if not is_retryable(e):
                print(f"Error not retryable for {model_name}: {e}")
                break
            print(f"Error: {e}\nRetry {retry + 1}/{MAX_RETRIES}...")
            time.sleep(RETRY_SLEEP)
    print(f"WARNING: completion JSON failed for {model_name}.")
    return None, {"provider": None, "prompt_tokens": None,
                      "completion_tokens": None, "reasoning_tokens": 0}


def clear_cache():
    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")