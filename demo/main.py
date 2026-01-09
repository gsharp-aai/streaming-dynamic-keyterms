"""
Dynamic Keyterms Streaming Demo for Housing/Healthcare Appointment Scheduling

This demo shows how to improve speech recognition accuracy by dynamically
boosting domain-specific keyterms during real-time transcription.

Usage:
    python main.py                       # Use microphone (boosted only)
    python main.py ../files/test_file.wav  # Stream file (comparison mode)

For configuration options, see config.py.
For details on extending to a voice agent, see EXTENDING_TO_VOICE_AGENT.md.
"""

import argparse
import logging
import os
import threading
import time
import wave
from typing import Type

import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)

from config import (
    API_KEY,
    ENCODING,
    END_OF_TURN_CONFIDENCE_THRESHOLD,
    GROUND_TRUTH_FILE,
    KEYTERM_REFRESH_THRESHOLD,
    LANGUAGE_DETECTION,
    MAX_TURN_SILENCE,
    MIN_END_OF_TURN_SILENCE_WHEN_CONFIDENT,
    SAMPLE_RATE,
    SPEECH_MODEL,
)
from keyterms import (
    generate_initial_keyterms,
    get_fallback_keyterms,
    load_previous_conversations,
    refresh_keyterms,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONVERSATION STATE TRACKING
# ============================================================================

class ConversationState:
    """Tracks the current conversation state for dynamic keyterm updates."""

    def __init__(self):
        self.current_transcript = ""
        self.word_count = 0
        self.last_keyterm_update_word_count = 0
        self.current_keyterms = []
        self.initial_keyterms_generated = False
        self.final_formatted_turns = []

    def add_transcript(self, text: str) -> int:
        """Add new transcript text and return total word count."""
        self.current_transcript += " " + text
        self.word_count = len(self.current_transcript.split())
        return self.word_count

    def should_refresh_keyterms(self) -> bool:
        """Check if we've hit the word threshold for keyterm refresh."""
        words_since_last_update = self.word_count - self.last_keyterm_update_word_count
        return words_since_last_update >= KEYTERM_REFRESH_THRESHOLD

    def mark_keyterms_updated(self):
        """Mark that keyterms were just updated."""
        self.last_keyterm_update_word_count = self.word_count


# Global state (in production, this would be per-session)
conversation_state = ConversationState()
streaming_client = None


# ============================================================================
# ASYNC KEYTERM FUNCTIONS
# ============================================================================

def generate_initial_keyterms_async(client: StreamingClient):
    """
    Background task to generate LLM-based keyterms and update the stream.
    """
    global conversation_state

    try:
        previous_conversations = load_previous_conversations()

        if not previous_conversations:
            logger.info("No previous conversations found - keeping fallback keyterms")
            return

        print()  # Space before keyterm generation logs
        logger.info("Generating contextual keyterms from conversation history...")
        llm_keyterms = generate_initial_keyterms(previous_conversations)

        if llm_keyterms and len(llm_keyterms) > 0:
            client.set_params(StreamingSessionParameters(keyterms_prompt=llm_keyterms))
            conversation_state.current_keyterms = llm_keyterms

            logger.info(f"KEYTERMS UPDATED: Replaced fallback keyterms with {len(llm_keyterms)} LLM-generated keyterms")
            print(f"\n>>> KEYTERMS UPDATED: Now using {len(llm_keyterms)} contextual keyterms from conversation history")
            print(f">>> ALL KEYTERMS BEING BOOSTED: {llm_keyterms}\n")
        else:
            logger.warning("LLM returned empty keyterms - keeping fallback keyterms")

    except Exception as e:
        logger.error(f"Failed to generate LLM keyterms: {e} - keeping fallback keyterms")


def refresh_keyterms_async(client: StreamingClient):
    """
    Background task to refresh keyterms based on conversation progress.
    """
    global conversation_state

    try:
        previous_conversations = load_previous_conversations()

        new_keyterms = refresh_keyterms(
            conversation_state.current_keyterms,
            conversation_state.current_transcript,
            previous_conversations
        )

        client.set_params(StreamingSessionParameters(keyterms_prompt=new_keyterms))

        conversation_state.current_keyterms = new_keyterms
        conversation_state.mark_keyterms_updated()

        print(f">>> Keyterms refreshed ({len(new_keyterms)} terms)")
        print(f">>> Sample: {new_keyterms[:3]}...\n")

    except Exception as e:
        logger.error(f"Failed to refresh keyterms: {e}")


# ============================================================================
# STREAMING EVENT HANDLERS - WITH BOOSTING
# ============================================================================

def on_begin(client: Type[StreamingClient], event: BeginEvent):
    """Handle session start - start with fallback keyterms, generate LLM keyterms in background."""
    global conversation_state, streaming_client
    streaming_client = client

    print(f"\n{'='*60}")
    print(f"Session started: {event.id}")
    print(f"{'='*60}\n")

    fallback_keyterms = get_fallback_keyterms()
    conversation_state.current_keyterms = fallback_keyterms
    client.set_params(StreamingSessionParameters(keyterms_prompt=fallback_keyterms))

    print(f"Started with {len(fallback_keyterms)} generic keyterms (generating contextual keyterms in background...)\n")

    conversation_state.initial_keyterms_generated = True
    keyterm_thread = threading.Thread(
        target=generate_initial_keyterms_async,
        args=(client,),
        daemon=True
    )
    keyterm_thread.start()


def on_turn(client: Type[StreamingClient], event: TurnEvent):
    """Handle transcription turn events with dynamic keyterm refresh."""
    global conversation_state

    if not event.transcript.strip():
        return

    if event.end_of_turn:
        if event.turn_is_formatted:
            print(f"[FINAL formatted] {event.transcript}\n")
            conversation_state.final_formatted_turns.append(event.transcript)
        else:
            print(f"[FINAL unformatted] {event.transcript}")
    else:
        print(f"[partial] {event.transcript}")

    if event.end_of_turn and not event.turn_is_formatted:
        word_count = conversation_state.add_transcript(event.transcript)

        if conversation_state.should_refresh_keyterms():
            print(f"\n>>> Reached {word_count} words - refreshing keyterms in background...\n")
            conversation_state.mark_keyterms_updated()

            refresh_thread = threading.Thread(
                target=refresh_keyterms_async,
                args=(client,),
                daemon=True
            )
            refresh_thread.start()


def on_terminated(_client: Type[StreamingClient], _event: TerminationEvent):
    """Handle session termination."""
    pass


def on_error(client: Type[StreamingClient], error: StreamingError):
    """Handle streaming errors."""
    print(f"Error occurred: {error}")


# ============================================================================
# STREAMING EVENT HANDLERS - NO BOOSTING (baseline)
# ============================================================================

def on_begin_no_boost(client: Type[StreamingClient], event: BeginEvent):
    """Handle session start for non-boosted session."""
    global conversation_state, streaming_client
    streaming_client = client

    print(f"\n{'='*60}")
    print(f"Session started (NO BOOSTING): {event.id}")
    print(f"{'='*60}\n")


def on_turn_no_boost(client: Type[StreamingClient], event: TurnEvent):
    """Handle transcription turn events for non-boosted session."""
    global conversation_state

    if not event.transcript.strip():
        return

    if event.end_of_turn:
        if event.turn_is_formatted:
            print(f"[FINAL formatted] {event.transcript}\n")
            conversation_state.final_formatted_turns.append(event.transcript)
        else:
            print(f"[FINAL unformatted] {event.transcript}")
    else:
        print(f"[partial] {event.transcript}")


def on_terminated_no_boost(client: Type[StreamingClient], event: TerminationEvent):
    """Handle session termination for non-boosted session."""
    print(f"\n{'='*60}")
    print(f"Session terminated (NO BOOSTING)")
    print(f"Audio duration: {event.audio_duration_seconds} seconds")
    print(f"{'='*60}")


# ============================================================================
# FILE STREAMING
# ============================================================================

def stream_file(filepath: str, sample_rate: int):
    """
    Stream audio file in chunks to simulate real-time audio.

    Args:
        filepath: Absolute path to a WAV file
        sample_rate: Expected sample rate for the streaming connection

    Yields:
        Audio frames in chunks
    """
    chunk_duration = 0.1  # 100ms chunks

    with wave.open(filepath, 'rb') as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("Only mono audio is supported")

        file_sample_rate = wav_file.getframerate()
        if file_sample_rate != sample_rate:
            print(f"Warning: File sample rate ({file_sample_rate}) doesn't match expected rate ({sample_rate})")

        frames_per_chunk = int(file_sample_rate * chunk_duration)

        while True:
            frames = wav_file.readframes(frames_per_chunk)
            if not frames:
                break
            yield frames
            time.sleep(chunk_duration)


# ============================================================================
# STREAMING CLIENT HELPERS
# ============================================================================

def create_streaming_client() -> StreamingClient:
    """Create a configured streaming client."""
    return StreamingClient(
        StreamingClientOptions(
            api_key=API_KEY,
            api_host="streaming.assemblyai.com",
        )
    )


def get_streaming_parameters(sample_rate: int) -> StreamingParameters:
    """Get streaming parameters with current configuration."""
    return StreamingParameters(
        sample_rate=sample_rate,
        speech_model=SPEECH_MODEL,
        encoding=ENCODING,
        end_of_turn_confidence_threshold=END_OF_TURN_CONFIDENCE_THRESHOLD,
        language_detection=LANGUAGE_DETECTION,
        min_end_of_turn_silence_when_confident=MIN_END_OF_TURN_SILENCE_WHEN_CONFIDENT,
        max_turn_silence=MAX_TURN_SILENCE,
        format_turns=True,
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for the dynamic keyterms streaming demo.

    When a file is provided, runs comparison mode:
    1. First without any keyterm boosting (baseline)
    2. Then with LLM-generated keyterm boosting

    Without a file, runs microphone mode with boosting enabled.
    """
    global conversation_state

    parser = argparse.ArgumentParser(
        description="Dynamic keyterms streaming demo for housing/healthcare scheduling"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=None,
        help="Optional: Path to a WAV file to stream instead of using microphone"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        help=f"Sample rate for audio (default: {SAMPLE_RATE})"
    )
    args = parser.parse_args()

    # Microphone mode (no file provided)
    if not args.audio_file:
        conversation_state = ConversationState()
        client = create_streaming_client()
        client.on(StreamingEvents.Begin, on_begin)
        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Termination, on_terminated)
        client.on(StreamingEvents.Error, on_error)
        client.connect(get_streaming_parameters(args.sample_rate))

        print("\nStarting microphone stream...")
        print("Speak about housing or healthcare appointments.")
        print("Keyterms will update automatically every 50 words.\n")
        audio_source = aai.extras.MicrophoneStream(sample_rate=args.sample_rate)

        try:
            client.stream(audio_source)
        except KeyboardInterrupt:
            print("\nStopping stream...")
        finally:
            client.disconnect(terminate=True)
        return

    # Comparison mode (file provided)
    print(f"\n{'#'*60}")
    print(f"# COMPARISON MODE: Running file twice")
    print(f"# 1) Without keyterm boosting (baseline)")
    print(f"# 2) With LLM-generated keyterm boosting")
    print(f"{'#'*60}")

    # SESSION 1: No boosting (baseline)
    print(f"\n\n{'='*60}")
    print("SESSION 1: NO BOOSTING (baseline)")
    print(f"{'='*60}")

    conversation_state = ConversationState()
    client1 = create_streaming_client()
    client1.on(StreamingEvents.Begin, on_begin_no_boost)
    client1.on(StreamingEvents.Turn, on_turn_no_boost)
    client1.on(StreamingEvents.Termination, on_terminated_no_boost)
    client1.on(StreamingEvents.Error, on_error)
    client1.connect(get_streaming_parameters(args.sample_rate))

    print(f"\nStreaming audio file: {args.audio_file}")
    audio_source1 = stream_file(args.audio_file, args.sample_rate)

    try:
        client1.stream(audio_source1)
    finally:
        client1.disconnect(terminate=True)

    session1_turns = conversation_state.final_formatted_turns.copy()

    # SESSION 2: With keyterm boosting
    print(f"\n\n{'='*60}")
    print("SESSION 2: WITH KEYTERM BOOSTING")
    print(f"{'='*60}")

    conversation_state = ConversationState()
    client2 = create_streaming_client()
    client2.on(StreamingEvents.Begin, on_begin)
    client2.on(StreamingEvents.Turn, on_turn)
    client2.on(StreamingEvents.Termination, on_terminated)
    client2.on(StreamingEvents.Error, on_error)
    client2.connect(get_streaming_parameters(args.sample_rate))

    print(f"\nStreaming audio file: {args.audio_file}")
    audio_source2 = stream_file(args.audio_file, args.sample_rate)

    try:
        client2.stream(audio_source2)
    finally:
        client2.disconnect(terminate=True)

    session2_turns = conversation_state.final_formatted_turns.copy()

    # COMPARISON SUMMARY
    print(f"\n\n{'#'*60}")
    print("# FINAL COMPARISON")
    print(f"{'#'*60}")

    if os.path.exists(GROUND_TRUTH_FILE):
        with open(GROUND_TRUTH_FILE, "r") as f:
            ground_truth = f.read().strip()
        print(f"\n{'='*60}")
        print("GROUND TRUTH:")
        print(f"{'='*60}")
        print(f"  {ground_truth}")

    print(f"\n{'='*60}")
    print("SESSION 1 (NO BOOSTING):")
    print(f"{'='*60}")
    for i, turn in enumerate(session1_turns, 1):
        print(f"  Turn {i}: {turn}")

    print(f"\n{'='*60}")
    print("SESSION 2 (WITH BOOSTING):")
    print(f"{'='*60}")
    for i, turn in enumerate(session2_turns, 1):
        print(f"  Turn {i}: {turn}")


if __name__ == "__main__":
    main()
