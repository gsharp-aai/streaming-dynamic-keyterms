"""
Keyterm generation and refresh logic using LLM Gateway.

This module handles:
- Loading previous conversation history from the database
- Generating initial keyterms using LLM Gateway
- Refreshing keyterms dynamically during conversation
- Fallback keyterms for when LLM is unavailable
"""

import json
import logging

import requests

from config import (
    API_KEY,
    DATABASE_FILE,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_KEYTERMS,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

def load_previous_conversations() -> list[dict]:
    """
    Load previous conversation transcripts from the database.

    In a production environment, this would query your actual database
    (e.g., PostgreSQL, MongoDB, etc.) for the customer's conversation history.

    Returns:
        List of conversation objects with 'text' field containing transcript text.
    """
    logger.info(f"Loading previous conversations from: {DATABASE_FILE}")

    try:
        with open(DATABASE_FILE, "r") as f:
            conversations = json.load(f)
        logger.info(f"Loaded {len(conversations)} previous conversations")
        return conversations
    except FileNotFoundError:
        logger.warning("No previous conversations database found")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing conversations database: {e}")
        return []


# ============================================================================
# LLM GATEWAY INTEGRATION
# ============================================================================

def call_llm_gateway(prompt: str, max_tokens: int = LLM_MAX_TOKENS) -> str:
    """
    Call LLM Gateway to generate responses.

    Args:
        prompt: The prompt to send to the LLM
        max_tokens: Maximum tokens in the response

    Returns:
        The LLM's response text
    """
    url = "https://llm-gateway.assemblyai.com/v1/chat/completions"
    headers = {
        "Authorization": API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": LLM_TEMPERATURE
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        logger.error(f"LLM Gateway request failed: {e}")
        return ""


# ============================================================================
# KEYTERM GENERATION
# ============================================================================

def generate_initial_keyterms(previous_conversations: list[dict]) -> list[str]:
    """
    Generate initial keyterms based on previous conversation history.

    Uses LLM Gateway to analyze previous transcripts and extract entities
    that should be boosted for better ASR recognition.

    Args:
        previous_conversations: List of previous conversation transcripts

    Returns:
        List of up to MAX_KEYTERMS keyterms for ASR boosting
    """
    all_text = "\n\n".join([conv["text"] for conv in previous_conversations])

    prompt = f"""You are helping improve speech recognition accuracy for a housing and healthcare appointment scheduling system.

TASK: Extract ALL proper nouns from the conversation history below and return them as keyterms for ASR boosting.

CRITICAL - THE FIRST 30+ KEYTERMS MUST BE PROPER NOUNS FROM THE TEXT:
1. All PERSON NAMES - extract every single name EXACTLY as spelled:
   - Full names with titles (e.g., "Dr. Firstname Lastname")
   - Full names WITHOUT titles (e.g., for "Dr. Mary Smith-Jones", also include "Mary Smith-Jones")
   - First names alone
   - Last names alone (including hyphenated surnames like "Smith-Jones")
   - CRITICAL FOR HYPHENATED NAMES: ASR processes words individually, so for names like "Mary Smith-Jones":
     * Include the full hyphenated form: "Smith-Jones"
     * ALSO include each component separately: "Smith", "Jones"
     * This ensures each word gets boosted even when ASR doesn't recognize the hyphen
   - Include diacritics/accents if present in the original (é, á, ü, etc.)

2. All PLACE/ORGANIZATION NAMES - extract EXACTLY as spelled:
   - Full organization names
   - The distinctive/unique word from each name (the part ASR would struggle with)
   - CRITICAL FOR HYPHENATED PLACE NAMES: Same rule as person names - for "Winston-Salem":
     * Include full form: "Winston-Salem"
     * ALSO include each part: "Winston", "Salem"

3. All MEDICATION NAMES - extract EXACTLY as spelled

WHY THIS MATTERS: ASR struggles with phonetically ambiguous words:
- Irish names (e.g., "Siobhan" sounds like "Shivawn", "Niamh" sounds like "Neev")
- Polish names (e.g., "Kowalczyk" sounds like "Kovalchik", "Brzezinski" sounds like "Brezhinski")
- African names (e.g., "Oluwaseun" sounds like "Oh-loo-wa-shay-oon")
- Native American place names (e.g., "Natchitoches" sounds like "Nack-a-tish")
- Medical terms (e.g., "Omeprazole" sounds like "oh-MEP-ra-zole")

Extract the EXACT spelling from the conversation history - these keyterms will help ASR recognize tricky words correctly.

PREVIOUS CONVERSATIONS:
{all_text}

OUTPUT FORMAT:
Return ONLY a JSON array of exactly 100 strings. The first 30+ MUST be the exact proper nouns extracted from the conversations above. Fill remaining slots with common healthcare/housing terms.
IMPORTANT: Do NOT include the word "clinic" as it sounds like "calling" and causes transcription errors.
No explanation or markdown - just the JSON array."""

    logger.info("Generating initial keyterms from conversation history...")
    response = call_llm_gateway(prompt)

    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1]
            response = response.rsplit("```", 1)[0]

        keyterms = json.loads(response)

        valid_keyterms = [
            term for term in keyterms
            if isinstance(term, str) and len(term) <= 50 and len(term) > 0
        ][:MAX_KEYTERMS]

        logger.info(f"Generated {len(valid_keyterms)} initial keyterms")
        return valid_keyterms

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse keyterms from LLM response: {e}")
        return get_fallback_keyterms()


def refresh_keyterms(
    current_keyterms: list[str],
    current_transcript: str,
    previous_conversations: list[dict]
) -> list[str]:
    """
    Refresh keyterms based on the current conversation progress.

    Called periodically to dynamically update keyterms based on:
    - What has been said in the current conversation
    - The existing keyterm list
    - Previous conversation context

    Args:
        current_keyterms: The current list of keyterms being used
        current_transcript: Transcript of the current conversation so far
        previous_conversations: Previous conversation history for context

    Returns:
        Updated list of up to MAX_KEYTERMS keyterms
    """
    history_text = "\n".join([conv["text"] for conv in previous_conversations[-3:]])

    prompt = f"""You are helping improve speech recognition accuracy for a housing and healthcare appointment scheduling call in progress.

CURRENT SITUATION:
- This is a live call about housing or healthcare appointment scheduling
- Below is what has been transcribed so far in this call
- You also have access to the current keyterms being used and previous conversation history

TASK: Generate an updated list of exactly 100 keyterms optimized for what might be said next in this conversation.

STRATEGY:
1. Keep keyterms that are still relevant based on the conversation context
2. Add new keyterms based on entities or topics mentioned in the current call that weren't in the original list
3. Remove keyterms that seem unlikely to appear based on the conversation direction
4. Prioritize terms that are likely to come up based on the conversation flow
5. Include any names, locations, medical terms, or housing terms mentioned in the current call

CRITICAL:
- The transcript may contain MISHEARD words. Look for phonetically similar patterns and include the CORRECT spelling from the conversation history, not the misheard version.
- If you see something in the transcript that looks like a mangled version of a name/medication from the history, include the correct version from history - do NOT include the misheard version.

CURRENT KEYTERMS (may keep, modify, or replace):
{json.dumps(current_keyterms[:50])}... (truncated)

CURRENT CALL TRANSCRIPT:
{current_transcript}

RECENT CONVERSATION HISTORY (for context):
{history_text}

OUTPUT FORMAT:
Return ONLY a JSON array of exactly 100 strings, each being a keyterm (50 characters or less).
No explanation or markdown - just the JSON array."""

    logger.info("Refreshing keyterms based on conversation progress...")
    response = call_llm_gateway(prompt, max_tokens=1500)

    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1]
            response = response.rsplit("```", 1)[0]

        keyterms = json.loads(response)

        valid_keyterms = [
            term for term in keyterms
            if isinstance(term, str) and len(term) <= 50 and len(term) > 0
        ][:MAX_KEYTERMS]

        logger.info(f"Refreshed to {len(valid_keyterms)} keyterms")
        return valid_keyterms

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse refreshed keyterms: {e}")
        return current_keyterms


def get_fallback_keyterms() -> list[str]:
    """
    Return fallback keyterms for housing/healthcare scheduling if LLM fails.
    """
    return [
        # Healthcare terms
        "appointment", "reschedule", "follow-up", "consultation",
        "primary care", "specialist", "referral", "prescription",
        "Medicare", "Medicaid", "insurance", "copay", "deductible",
        "cardiology", "orthopedics", "nephrology", "oncology",
        "physical therapy", "occupational therapy", "dialysis",
        "blood pressure", "cholesterol", "diabetes", "Metformin",
        "MRI", "CT scan", "X-ray", "ultrasound", "lab work",
        # Housing terms
        "Section 8", "housing voucher", "HUD", "subsidized housing",
        "affordable housing", "income verification", "lease agreement",
        "rental assistance", "LIHEAP", "weatherization",
        "housing authority", "case worker", "application status",
        "maintenance request", "property manager", "landlord",
        # General scheduling
        "available", "morning", "afternoon", "Tuesday", "Thursday",
        "next week", "tomorrow", "confirm", "cancel", "waiting list",
        # Common entities
        "Social Security", "disability", "SNAP", "food stamps",
        "home health aide", "visiting nurse", "transportation",
    ]
