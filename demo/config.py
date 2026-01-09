"""
Configuration settings for the Dynamic Keyterms Streaming Demo.

Edit these values to customize the demo for your environment.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# API KEY
# ============================================================================

API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
if not API_KEY:
    raise ValueError("ASSEMBLYAI_API_KEY environment variable is required. Set it or create a .env file.")

# ============================================================================
# STREAMING API CONFIGURATION
# API Reference: https://www.assemblyai.com/docs/api-reference/streaming-api/streaming-api#request.query
# Endpointing recommendations: https://www.assemblyai.com/docs/universal-streaming/turn-detection#quick-start-configurations
# ============================================================================

# Sample rate for audio input (Hz). Must match your audio source.
SAMPLE_RATE = 16000

# Speech model to use for transcription
# Options: "universal-streaming-english" (faster) or "universal-streaming-multilingual"
SPEECH_MODEL = "universal-streaming-english"

# Audio encoding format
# Options: "pcm_s16le" (16-bit signed little-endian PCM) or "pcm_mulaw" (mu-law)
ENCODING = "pcm_s16le"

# Confidence threshold (0-1) for detecting end of turn
END_OF_TURN_CONFIDENCE_THRESHOLD = 0.4

# Adds language metadata to turns (only available with multilingual model)
# Options: True or False
LANGUAGE_DETECTION = False

# Minimum silence duration in ms when confident about end of turn
MIN_END_OF_TURN_SILENCE_WHEN_CONFIDENT = 400

# Maximum silence allowed in a turn before end of turn is triggered (ms)
MAX_TURN_SILENCE = 1280

# ============================================================================
# LLM GATEWAY CONFIGURATION
# ============================================================================

# Model to use for keyterm generation
# Options: "claude-sonnet-4-5-20250929", "claude-3-5-haiku-20241022", etc.
# See https://www.assemblyai.com/docs/llm-gateway/overview#available-models
LLM_MODEL = "claude-sonnet-4-5-20250929"

# Maximum tokens for LLM response
LLM_MAX_TOKENS = 2000

# Temperature for LLM (0 = deterministic, higher = more creative)
LLM_TEMPERATURE = 0

# ============================================================================
# KEYTERM REFRESH CONFIGURATION
# ============================================================================

# Number of words between keyterm refreshes during conversation
KEYTERM_REFRESH_THRESHOLD = 50

# Maximum number of keyterms to generate
MAX_KEYTERMS = 100

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Path to conversation history database (JSON file)
DATABASE_FILE = os.path.join(os.path.dirname(__file__), "previous_conversations.json")

# Path to ground truth file for comparison mode
GROUND_TRUTH_FILE = os.path.join(os.path.dirname(__file__), "..", "files", "ground_truth.txt")
