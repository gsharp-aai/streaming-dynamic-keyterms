# Dynamic Keyterm Boosting for Streaming ASR

This demo shows how to improve speech recognition accuracy by dynamically boosting domain-specific [keyterms](https://www.assemblyai.com/docs/universal-streaming/keyterms-prompting) during real-time transcription. It uses AssemblyAI's [Streaming API](https://www.assemblyai.com/docs/universal-streaming) with [LLM Gateway](https://www.assemblyai.com/docs/llm-gateway/overview)-generated keyterms extracted from customer conversation history.

## What This Demo Does

The script demonstrates a two-phase keyterm boosting approach:

1. **Generic Fallback Keyterms**: The session starts immediately with generic healthcare/housing terms (e.g., "appointment", "prescription", "lease") for low-latency startup.

2. **Contextual LLM-Generated Keyterms**: In the background, an LLM analyzes the customer's conversation history and generates personalized keyterms (names, locations, medications, etc.) that are then sent to the streaming session.

### How Keyterm Generation Works

The LLM extracts proper nouns from conversation history that ASR would typically struggle with:
- **Person names** with less common spellings (e.g., "Oluwatoyin Adéwálé", "Byrne-Donahue")
- **Place names** that are phonetically ambiguous (e.g., "Schuylkill", "Ouachita", "Wilkes-Barre")
- **Medication names** (e.g., "Atorvastatin", "Farxiga")
- **Organization names** specific to the customer's context

### The Conversation History Database

The `previous_conversations.json` file simulates a customer database. In production, this would be replaced with a call to your actual database where customer context is stored (CRM, call logs, appointment history, etc.). The key insight is that you likely already have valuable context about each caller that can dramatically improve transcription accuracy.

## Setup

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install "assemblyai[extras]" python-dotenv requests
```

### 2. Set Your API Key

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then edit `.env` and add your AssemblyAI API key (found on the [API Keys page](https://www.assemblyai.com/dashboard/api-keys)):

```
ASSEMBLYAI_API_KEY=your_api_key_here
```

Note: LLM Gateway is a paid feature, so you'll need to add a card to your account (your free credits will still apply).

## Running the Demo

### ⭐ Option 1: Comparison Mode with Audio File (Recommended)

This is the best way to evaluate the impact of keyterm boosting. Provide an audio file containing words from the `previous_conversations.json` database (names, places, medications). The script runs the same audio twice:
1. First **without** keyterm boosting (baseline)
2. Then **with** LLM-generated keyterm boosting

At the end, you see a side-by-side comparison showing how boosting improves accuracy for difficult terms.

```bash
cd demo
python main.py ../files/test_file.wav
```

**Demo output (with brackets for readability):**

```
============================================================
GROUND TRUTH:
============================================================
  Hi, this is [Kelly Byrne-Donoghue] and I'm calling just to confirm
  my appointment with Dr. [Oluwatoyin Adéwálé] at the [Schuylkill]
  Family Health Center. I also need to reschedule my physical
  therapy with Dr. [Xiomara] at [Ouachita] Rehabilitation Center.
  My sister [Leigh Rhys-Davies] is picking up my [Atorvastatin] and
  [Farxiga] prescriptions from the [Wilkes-Barre] [CVS] pharmacy.

============================================================
SESSION 1 (NO BOOSTING):
============================================================
  Turn 1: Hi, this is [Kelly Byrne Donahue], and I'm calling just to
          confirm my appointment with Dr. [Oluatoyan Adewale] at the
          [Schuylkill] Family Health Center.
  Turn 2: I also need to reschedule my physical therapy with
          Dr. [Ziomara] at the [Wichita] Rehabilitation Center.
  Turn 3: My sister, [Lee Re Davies] is picking up my [autor bastatin]
          and [farzika] prescriptions from the [Wilkes Bear] [CBS] pharmacy.

============================================================
SESSION 2 (WITH BOOSTING):
============================================================
  Turn 1: Hi, this is [Kelly Byrne-Donahue], and I'm calling just to
          confirm my appointment with Dr. [Oluwatoyin Adéwálé] at the
          [Schuylkill] Family Health Center.
  Turn 2: I also need to reschedule my physical therapy with
          Dr. [Xiomara] at the [Ouachita] Rehabilitation Center.
  Turn 3: My sister, [Leigh Rhys-Davies] is picking up my [Atorvastatin]
          and [Farxiga] prescriptions from the [Wilkes-Barre] [CVS] pharmacy.
```

**Audio file requirements:**
- Format: WAV (16-bit PCM)
- Sample rate: 16kHz (or specify with `--sample-rate` and change `SAMPLE_RATE` in `config.py`)
- Channels: Mono

### Option 2: Live Microphone Streaming

Stream directly from your microphone with keyterm boosting enabled:

```bash
cd demo
python main.py
```

Speak naturally and watch how the transcription handles difficult names and terms. After 50 words, keyterms will dynamically refresh based on conversation content. You can customize endpointing behavior and other streaming parameters in `config.py` (see [API reference](https://www.assemblyai.com/docs/api-reference/streaming-api/streaming-api#request.query) and [turn detection configurations](https://www.assemblyai.com/docs/universal-streaming/turn-detection#quick-start-configurations)).

## What This Demo Does NOT Include

This demo focuses solely on **Speech-to-Text with keyterm boosting**. It does not include:
- Voice agent / conversational AI responses
- Text-to-Speech output

**⭐ To extend this to a full voice agent, see [`EXTENDING_TO_VOICE_AGENT.md`](demo/EXTENDING_TO_VOICE_AGENT.md) for detailed instructions on using LLM Gateway to maintain agent response history alongside transcription context.**

## How It Works Under the Hood

1. **Session Start**: Generic keyterms are loaded immediately for low-latency startup
2. **Background LLM Call**: The customer's conversation history is sent to Claude via LLM Gateway
3. **Keyterm Extraction**: The LLM returns up to 100 domain-specific keyterms
4. **Dynamic Update**: Keyterms are pushed to the active streaming session via `set_params()`
5. **Ongoing Refresh**: Every 50 words, keyterms can be refreshed based on new conversation content

## File Structure

```
demo/
├── main.py          # Main entry point - streaming logic and event handlers
├── config.py        # Configuration constants (edit this to customize)
├── keyterms.py      # LLM Gateway integration and keyterm generation
├── previous_conversations.json  # Sample conversation history database
└── EXTENDING_TO_VOICE_AGENT.md  # Guide for adding voice agent capabilities
```

## Customization

- ⭐ **Change configuration**: Edit `config.py` to adjust sample rate, speech model, LLM model, and [streaming parameters](https://www.assemblyai.com/docs/api-reference/streaming-api/streaming-api#request.query)
- **Change the conversation history**: Edit `previous_conversations.json` or replace the `load_previous_conversations()` function in `keyterms.py` with your own database call
- **Modify keyterm extraction**: Adjust the LLM prompt in `keyterms.py` for your domain
- **Adjust refresh frequency**: Change `KEYTERM_REFRESH_THRESHOLD` in `config.py` (default: 50 words)
