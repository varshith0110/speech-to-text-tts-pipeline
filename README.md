# Speech-to-Text and Text-to-Speech Pipeline

A Python project that combines speech recognition and text-to-speech to convert spoken audio into text and generate spoken audio from text.

## Features

- **Speech-to-Text:** Records audio from a microphone and converts speech into text using OpenAI Whisper.
- **Text-to-Speech:** Converts typed text into spoken audio using Edge TTS.
- **Speech-to-Text-to-Speech:** Transcribes recorded speech and then reads the recognized text aloud.
- **Word Timestamps:** Uses Whisper's word-level timestamp information when available.

## Technologies Used

- Python
- OpenAI Whisper
- Edge TTS
- SoundDevice
- SoundFile
- NumPy

## Requirements

- Python installed on your computer
- A working microphone for speech recording
- An internet connection for downloading the Whisper model and using Edge TTS
- FFmpeg, if required by your Whisper installation

## Installation

Clone this repository:

```bash
git clone https://github.com/varshith0110/speech-to-text-tts-pipeline.git
cd speech-to-text-tts-pipeline
```

Create and activate a virtual environment.

**Windows:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install openai-whisper edge-tts sounddevice soundfile numpy
```

## Usage

Run the Python script:

```bash
python fishaudio.py
```

Follow the prompts in the terminal to use the available speech recognition and text-to-speech modes.

## Project Structure

```text
speech-to-text-tts-pipeline/
├── fishaudio.py
├── .gitignore
└── README.md
```

## Future Improvements

- Add support for multiple languages.
- Improve error handling for microphone and audio playback issues.
- Add a simple graphical user interface.
- Make audio recording duration configurable.

## License

No license has been added yet.
