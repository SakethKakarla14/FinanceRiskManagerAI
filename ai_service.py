import os
import json
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger("AIService")

# ── Module-Level Singleton ──────────────────────────────────────────────
# The LLM client is instantiated ONCE at import time, not per-request.
# This eliminates redundant network handshakes and object allocations.
_llm_instance = None

def _get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatGroq(temperature=0, model_name="openai/gpt-oss-120b")
    return _llm_instance


async def generate_ai_response(ml_data: dict, request_type: str) -> str:
    """
    Generates either a forensic summary or a bank chargeback dispute letter
    using Groq and LangChain based on the provided ML data.

    Uses a module-level singleton LLM client to avoid per-request initialization overhead.
    """
    if "GROQ_API_KEY" not in os.environ:
        return "Error: GROQ_API_KEY environment variable not set."

    try:
        llm = _get_llm()
    except Exception as e:
        return f"Error initializing Groq client: {e}"

    if request_type == "summary":
        prompt = PromptTemplate.from_template(
            "You are an expert Fraud Analyst. Review the following machine learning output for User {user_id}:\n"
            "{ml_data}\n\n"
            "Write a concise, 3-sentence forensic summary explaining exactly why this user is a risk based on the data. "
            "Do not use overly flowery language. Be analytical and direct."
        )
    elif request_type == "chargeback":
        prompt = PromptTemplate.from_template(
            "You are a Corporate Legal AI defending a merchant against a bank chargeback initiated by User {user_id}.\n"
            "Here are the forensic machine learning logs regarding this user's interactions with our platform:\n"
            "{ml_data}\n\n"
            "Draft a formal, professional dispute letter addressed to 'Visa/Mastercard Adjudication Team'.\n"
            "Use the provided data to explicitly prove the user's claim is fraudulent.\n"
            "Keep it under 250 words and maintain a strict legal tone."
        )
    else:
        return "Error: Invalid request_type. Must be 'summary' or 'chargeback'."

    chain = prompt | llm
    user_id = ml_data.get("User_ID", "Unknown")

    try:
        response = await chain.ainvoke({
            "user_id": user_id,
            "ml_data": json.dumps(ml_data, indent=2)
        })
        return response.content
    except Exception as e:
        return f"Error communicating with Groq API: {e}"