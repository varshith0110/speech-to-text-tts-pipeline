"""
Full STT + TTS Pipeline
========================
STT: OpenAI Whisper  → converts microphone audio to text (word-by-word timestamps)
TTS: Edge TTS        → converts text back to natural speech

Install dependencies:
    pip install openai-whisper edge-tts sounddevice soundfile numpy pyaudio

Usage:
    python pipeline.py
"""

import asyncio
import tempfile
import os
import threading
import time
import wave
import numpy as np

# ── STT (Whisper) ──────────────────────────────────────────────────────────────
import whisper

# ── TTS (Edge TTS) ─────────────────────────────────────────────────────────────
import edge_tts

# ── Audio recording ────────────────────────────────────────────────────────────
import sounddevice as sd
import soundfile as sf


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

STT_MODEL       = "base"          # Whisper model: tiny | base | small | medium | large
SAMPLE_RATE     = 16000           # Hz — Whisper works best at 16kHz
RECORD_SECONDS  = 15           # How long to record microphone input3
CHANNELS        = 1               # Mono

# Edge TTS voice — change to your preference:
# Full list: run `edge-tts --list-voices` in terminal
TTS_VOICE       = "en-US-JennyNeural"   # Natural female voice (US English)
# TTS_VOICE     = "en-US-GuyNeural"     # Natural male voice
# TTS_VOICE     = "en-IN-NeerjaNeural"  # Indian English female


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — RECORD AUDIO FROM MICROPHONE
# ══════════════════════════════════════════════════════════════════════════════

def record_audio(duration: int = RECORD_SECONDS, sample_rate: int = SAMPLE_RATE) -> str:
    """
    Records audio from the microphone for `duration` seconds.
    Saves to a temp WAV file and returns its path.
    """
    print(f"\n Recording for {duration} seconds... Speak now!")
    
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32"
    )
    sd.wait()  # Block until recording is done
    print("  Recording complete.\n")

    # Save to a temporary WAV file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_data, sample_rate)
    return tmp.name


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SPEECH TO TEXT (Whisper)
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_audio(audio_path: str, model_name: str = STT_MODEL) -> dict:
    """
    Transcribes the audio file using OpenAI Whisper.
    Returns a dict with:
      - 'text'    : full transcription string
      - 'words'   : list of {word, start, end} dicts (word-level timestamps)
      - 'segments': full Whisper segment data
    """
    print(f" Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    print("  Transcribing audio...")
    result = model.transcribe(
        audio_path,
        word_timestamps=True,   # Enable word-level timestamps ✨
        language="en"
    )

    # Flatten word timestamps from all segments
    words = []
    for segment in result.get("segments", []):
        for word_data in segment.get("words", []):
            words.append({
                "word" : word_data["word"].strip(),
                "start": round(word_data["start"], 2),
                "end"  : round(word_data["end"], 2),
            })

    return {
        "text"    : result["text"].strip(),
        "words"   : words,
        "segments": result["segments"],
    }


def print_word_by_word(words: list):
    """
    Simulates the LinkedIn demo: prints each word as it would be spoken,
    with timing delays to match the original audio pacing.
    """
    if not words:
        return

    print("\n  Word-by-word playback:")
    print("─" * 40)
    
    start_time = time.time()
    
    for i, word_info in enumerate(words):
        # Wait until the word's timestamp
        target_time = word_info["start"]
        elapsed = time.time() - start_time
        wait = target_time - elapsed
        if wait > 0:
            time.sleep(wait)
        
        # Print word with a highlight effect
        print(f"\033[93m{word_info['word']}\033[0m", end=" ", flush=True)
    
    print("\n" + "─" * 40)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TEXT TO SPEECH (Edge TTS)
# ══════════════════════════════════════════════════════════════════════════════

async def text_to_speech(text: str, output_path: str, voice: str = TTS_VOICE):
    """
    Converts text to speech using Edge TTS and saves to output_path.
    """
    print(f"\n Generating speech with voice: {voice}")
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_path)
    print(f"  Audio saved to: {output_path}")


def play_audio(audio_path: str):
    """
    Plays an audio file through the speakers.
    """
    print("▶   Playing audio...")
    data, sample_rate = sf.read(audio_path)
    sd.play(data, sample_rate)
    sd.wait()
    print("  Playback complete.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

async def run_pipeline():
    print("=" * 50)
    print("     STT + TTS PIPELINE")
    print("=" * 50)

    # ── Mode selection ─────────────────────────────────────────────────────────
    print("\nChoose mode:")
    print("  1. Record from microphone → STT → display text")
    print("  2. Type text → TTS → play audio")
    print("  3. Record from microphone → STT → TTS → play back")
    choice = input("\nEnter choice (1/2/3): ").strip()

    # ── MODE 1: Microphone → STT ───────────────────────────────────────────────
    if choice == "1":
        audio_path = record_audio()
        result = transcribe_audio(audio_path)

        print(f"\n  Full Transcription:\n  \"{result['text']}\"")
        print_word_by_word(result["words"])

        os.unlink(audio_path)  # Clean up temp file

    # ── MODE 2: Text → TTS ────────────────────────────────────────────────────
    elif choice == "2":
        text = input("\nEnter text to speak: ").strip()
        if not text:
            print("No text entered.")
            return

        output_path = "tts_output.mp3"
        await text_to_speech(text, output_path)
        play_audio(output_path)

    # ── MODE 3: Full Pipeline (Mic → STT → TTS → Speaker) ─────────────────────
    elif choice == "3":
        # Step A: Record
        audio_path = record_audio()

        # Step B: Transcribe
        result = transcribe_audio(audio_path)
        transcribed_text = result["text"]

        print(f"\n📄  Transcribed Text:\n  \"{transcribed_text}\"")
        print_word_by_word(result["words"])

        # Step C: Convert back to speech
        output_path = "pipeline_output.mp3"
        await text_to_speech(transcribed_text, output_path)

        # Step D: Play the TTS output
        play_audio(output_path)

        os.unlink(audio_path)

    else:
        print("Invalid choice.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    asyncio.run(run_pipeline())