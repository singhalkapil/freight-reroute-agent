"""Centralized Google Gemini LLM provider factory for freight-reroute-agent.

Resolves API credentials from environment variables (GEMINI_API_KEY or GOOGLE_API_KEY)
and returns a configured ChatGoogleGenerativeAI instance ready for .with_structured_output()
and streaming via LangGraph astream_events("v2").
"""

import os

from langchain_google_genai import ChatGoogleGenerativeAI


def get_gemini_chat_model(
    model: str = "gemini-1.5-flash",
    temperature: float = 0.0,
) -> ChatGoogleGenerativeAI:
    """Return a configured ChatGoogleGenerativeAI instance.

    Resolves the API key by checking GEMINI_API_KEY first, then GOOGLE_API_KEY as a
    fallback (both are accepted by the Google AI Studio backend).

    Args:
        model: Gemini model identifier (default: ``gemini-1.5-flash``).
        temperature: Sampling temperature; 0.0 for deterministic structured extraction.

    Returns:
        Configured :class:`ChatGoogleGenerativeAI` instance.

    Raises:
        ValueError: If neither ``GEMINI_API_KEY`` nor ``GOOGLE_API_KEY`` is set.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Google Gemini API key not configured. "
            "Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or .env file. "
            "Obtain a free key at https://aistudio.google.com/"
        )

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )
