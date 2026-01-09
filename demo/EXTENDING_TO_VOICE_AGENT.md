# Extending This Demo to a Full Voice Agent

This document explains how to extend the keyterm boosting demo to include a full voice agent using LLM Gateway for conversational AI responses.

## Overview

While the main demo (`dynamic_keyterms_streaming.py`) focuses on improving transcription accuracy through dynamic keyterms, the same LLM Gateway can be used to maintain agent context and provide intelligent responses based on conversation history.

## Tracking Agent Messages

The Chat Completions endpoint (`https://llm-gateway.assemblyai.com/v1/chat/completions`) supports multi-turn conversations through the `messages` array. To track agent context, maintain a conversation history:

```python
conversation_history = [
    {"role": "system", "content": "You are a helpful housing/healthcare scheduling assistant."},
    {"role": "user", "content": "I need to schedule an appointment with Dr. Martinez."},
    {"role": "assistant", "content": "I can help you schedule with Dr. Martinez. What date works best?"},
    {"role": "user", "content": "Next Tuesday afternoon would be ideal."},
    # ... continue adding messages as the conversation progresses
]
```

## Informing the Agent of Conversation Context

To give your agent full context of what has been said, you can:

### Option 1: Include the full transcript in the system message

```python
messages = [
    {
        "role": "system",
        "content": f'''You are a scheduling assistant. Here is the conversation so far:

        TRANSCRIPT:
        {current_transcript}

        Respond helpfully to continue the conversation.'''
    },
    {"role": "user", "content": "What was the patient asking about?"}
]
```

### Option 2: Use the conversation history array for turn-by-turn context

```python
# After each user turn from STT:
conversation_history.append({"role": "user", "content": transcribed_text})

# Get agent response:
response = requests.post(
    "https://llm-gateway.assemblyai.com/v1/chat/completions",
    headers={"Authorization": API_KEY, "Content-Type": "application/json"},
    json={
        "model": "claude-sonnet-4-5-20250929",
        "messages": conversation_history,
        "max_tokens": 500
    }
)

agent_response = response.json()["choices"][0]["message"]["content"]

# Add agent response to history:
conversation_history.append({"role": "assistant", "content": agent_response})
```

### Option 3: Combine with previous conversation history for returning customers

```python
# Load previous conversations
previous_convos = load_previous_conversations()

messages = [
    {
        "role": "system",
        "content": f'''You are a scheduling assistant for housing and healthcare.

        CUSTOMER HISTORY (previous interactions):
        {json.dumps(previous_convos, indent=2)}

        Use this context to provide personalized assistance.'''
    },
    *conversation_history  # Current conversation turns
]
```

## Full Agent Integration Example

Here's how you might integrate this with a voice agent:

```python
def handle_user_speech(transcribed_text: str):
    # Add to conversation history
    conversation_history.append({"role": "user", "content": transcribed_text})

    # Include previous conversation context
    previous_convos = load_previous_conversations()
    context_summary = summarize_history(previous_convos)  # Your summarization logic

    # Call LLM Gateway for agent response
    response = requests.post(
        "https://llm-gateway.assemblyai.com/v1/chat/completions",
        headers={"Authorization": API_KEY, "Content-Type": "application/json"},
        json={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a helpful scheduling assistant. Customer context: {context_summary}"
                },
                *conversation_history
            ],
            "max_tokens": 300
        }
    )

    agent_response = response.json()["choices"][0]["message"]["content"]
    conversation_history.append({"role": "assistant", "content": agent_response})

    # Send to TTS and play to user
    return agent_response
```

This approach ensures:
- The agent has full context of the current conversation
- Previous interaction history informs the agent's responses
- Keyterms can be updated based on both STT output and agent responses
- The entire system maintains coherent context throughout the call

## How to Extend the Demo

The demo is designed to be easily extensible when you're ready to add an agent component. The key principle is **separation of concerns**:

### 1. Keep the keyterm logic as-is for STT accuracy

The dynamic keyterm generation (`load_previous_conversations`, `generate_initial_keyterms`, `refresh_keyterms`) continues to run independently to optimize transcription quality. This ensures your ASR accuracy improvements remain intact.

### 2. Add a separate conversation_history array to track the agent dialogue

Create a new list to track the back-and-forth between user (STT output) and assistant (agent responses):

```python
# Add this to ConversationState or as a separate global
agent_conversation_history = [
    {"role": "system", "content": "You are a housing/healthcare scheduling assistant."}
]
```

### 3. Use that history when calling LLM Gateway for agent responses

When the user finishes speaking (`end_of_turn=True`), append their transcript to the history, call LLM Gateway for a response, then append the agent's reply:

```python
def on_turn_with_agent(client, event: TurnEvent):
    if event.end_of_turn:
        # Existing keyterm refresh logic stays here...

        # NEW: Agent response generation
        agent_conversation_history.append({"role": "user", "content": event.transcript})

        response = requests.post(
            "https://llm-gateway.assemblyai.com/v1/chat/completions",
            headers={"Authorization": API_KEY, "Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-5-20250929",
                "messages": agent_conversation_history,
                "max_tokens": 300
            }
        )

        agent_reply = response.json()["choices"][0]["message"]["content"]
        agent_conversation_history.append({"role": "assistant", "content": agent_reply})

        # Send agent_reply to TTS for playback...
```

This separation allows the keyterm system and agent system to evolve independently while sharing the same underlying STT stream and LLM Gateway infrastructure.
