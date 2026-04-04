"""
Shared LLM helper functions.

This file supports two classroom paths:
1. Ollama or OpenAI for stronger prompt-following results
2. Local Hugging Face models as an offline fallback
"""

import os
from urllib import error, request
from typing import Any

from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Load values from a local .env file if one exists.
load_dotenv()


# Prefer local Ollama automatically. Fall back to the local Hugging Face path
# so the project still works offline even without the Ollama server.
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5-mini")

# We keep the Hugging Face model name in one place so it is easy to change
# during class when running locally.
DEFAULT_MODEL_NAME = os.getenv("HF_MODEL_NAME", "google/flan-t5-large")
FALLBACK_MODEL_NAME = os.getenv("HF_FALLBACK_MODEL_NAME", "google/flan-t5-base")


def resolve_provider() -> str:
    """
    Decide which LLM backend the demo should use.

    Parameters:
    - None.

    Returns:
    - str: The provider name to use, such as `ollama`, `openai`, or
      `huggingface`.
    """
    if DEFAULT_PROVIDER in {"ollama", "openai", "huggingface"}:
        return DEFAULT_PROVIDER

    if is_ollama_available():
        return "ollama"

    return "huggingface"


def is_ollama_available() -> bool:
    """
    Check whether the local Ollama HTTP server is reachable.

    Parameters:
    - None.

    Returns:
    - bool: `True` if the Ollama server responds successfully, otherwise `False`.
    """
    try:
        with request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2) as response:
            return response.status == 200
    except (error.URLError, TimeoutError):
        return False


def load_ollama_llm(model_name: str = OLLAMA_MODEL_NAME) -> Any:
    """
    Build a small config object for using a local Ollama model.

    Parameters:
    - model_name: The Ollama model tag to use for generation.

    Returns:
    - Any: A dictionary describing the active Ollama provider and model.
    """
    if not is_ollama_available():
        raise RuntimeError(
            "Could not connect to the local Ollama server. "
            "Make sure Ollama is running."
        )

    print(f"\nLoading Ollama model: {model_name}")
    return {
        "provider": "ollama",
        "model_name": model_name,
    }


def load_openai_llm(model_name: str = OPENAI_MODEL_NAME) -> Any:
    """
    Build a config object for using the OpenAI API.

    Parameters:
    - model_name: The OpenAI model name to use for generation.

    Returns:
    - Any: A dictionary containing the OpenAI client and selected model.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI SDK is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    print(f"\nLoading OpenAI model: {model_name}")
    client = OpenAI(api_key=OPENAI_API_KEY)

    return {
        "provider": "openai",
        "client": client,
        "model_name": model_name,
    }


def load_huggingface_llm(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """
    Load a local Hugging Face model for text generation.

    Parameters:
    - model_name: The preferred Hugging Face model name to load first.

    Returns:
    - Any: A dictionary containing the tokenizer, model, provider name, and
      resolved model name.
    """
    candidate_models = [model_name]
    if FALLBACK_MODEL_NAME != model_name:
        candidate_models.append(FALLBACK_MODEL_NAME)

    last_error = None
    for candidate_model in candidate_models:
        print(f"\nLoading Hugging Face model: {candidate_model}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(candidate_model)
            return {
                "provider": "huggingface",
                "tokenizer": tokenizer,
                "model": model,
                "model_name": candidate_model,
            }
        except Exception as exc:
            last_error = exc
            if candidate_model != candidate_models[-1]:
                print(
                    f"Could not load {candidate_model}. "
                    f"Falling back to {candidate_models[-1]}."
                )

    # The first run may fail if the model has not been downloaded yet
    # and the machine does not currently have internet access.
    raise RuntimeError(
        "Could not load the Hugging Face model. "
        "Check your internet connection for the first download "
        "or choose a smaller cached model."
    ) from last_error


def load_llm() -> Any:
    """
    Load the currently selected LLM provider for the classroom demo.

    Parameters:
    - None.

    Returns:
    - Any: A provider-specific dictionary used later by `ask_llm()`.
    """
    # Resolve the provider once so the rest of the demo can use one interface.
    provider = resolve_provider()
    if provider == "ollama":
        return load_ollama_llm()
    if provider == "openai":
        return load_openai_llm()
    return load_huggingface_llm()


def ask_llm(generator: Any, prompt: str, max_new_tokens: int = 128) -> str:
    """
    Send a prompt to the model and return the generated text.

    Parameters:
    - generator: The provider config returned by `load_llm()`.
    - prompt: The instruction text sent to the model.
    - max_new_tokens: The maximum number of new tokens to generate.

    Returns:
    - str: The generated answer text.
    """
    if generator["provider"] == "ollama":
        payload = (
            "{"
            f"\"model\": {json_string(generator['model_name'])}, "
            f"\"prompt\": {json_string(prompt)}, "
            "\"stream\": false, "
            f"\"options\": {{\"num_predict\": {max_new_tokens}}}"
            "}"
        ).encode("utf-8")
        ollama_request = request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(ollama_request, timeout=120) as response:
            response_body = response.read().decode("utf-8")
        return parse_ollama_response(response_body).strip()

    if generator["provider"] == "openai":
        response = generator["client"].responses.create(
            model=generator["model_name"],
            input=prompt,
            max_output_tokens=max_new_tokens,
        )
        return response.output_text.strip()

    tokenizer = generator["tokenizer"]
    model = generator["model"]

    # Convert the text prompt into token ids that the model can process.
    encoded_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    # Use light beam search and repetition controls so the small classroom
    # model produces more stable answers for instructional prompts.
    generated_tokens = model.generate(
        **encoded_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=4,
        early_stopping=True,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
    )

    # Convert the generated token ids back into readable text.
    answer = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True,
    )

    return answer.strip()


def json_string(value: str) -> str:
    """
    Encode a Python string as a JSON-safe string literal.

    Parameters:
    - value: The raw Python string to encode.

    Returns:
    - str: The JSON-encoded string literal.
    """
    import json

    return json.dumps(value)


def parse_ollama_response(response_body: str) -> str:
    """
    Extract the generated answer text from an Ollama API response body.

    Parameters:
    - response_body: The raw JSON response text returned by Ollama.

    Returns:
    - str: The generated text stored in the `response` field.
    """
    import json

    # Ollama returns a JSON object whose `response` field contains the text.
    return json.loads(response_body)["response"]
