# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a voice cloning TTS (Text-to-Speech) system that separates speakers from audio files and clones their voices using various TTS backends. The system supports multiple TTS backends including Coqui TTS, pyttsx3, and espeak, with both local CPU-based processing and cloud-based models.

## Architecture

The codebase is organized into three main components:

1. **AudioProcessor** (`audio_processor.py`): Handles audio preprocessing, format conversion, noise reduction, and audio quality enhancement
2. **SpeakerSeparator** (`speaker_separator.py`): Separates speakers from audio using either local spectral clustering or HuggingFace pyannote models
3. **VoiceCloner** (`voice_cloner.py`): Clones voices using various TTS backends (Coqui TTS, pyttsx3, espeak)

The main entry point is `__main__.py` which orchestrates the complete pipeline: audio preprocessing → speaker separation → voice cloning.

## Common Development Commands

### Environment Setup
```bash
# Create virtual environment with Python 3.11
uv venv -p 3.11.7
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install -r ./pyproject.toml --all-extras
```

### Running the Application

#### First-time setup (create voice models):
```bash
# Create voice models from audio file (one-time setup)
python -m voice_clone_tts input_audio.wav --method local --backend coqui --use-cpu --save-models

# This will create voice models in the 'output/voice_models' directory
# Models are saved as: output/voice_models/speaker_0_reference.wav, speaker_0_metadata.json, etc.
```

#### Fast generation (using saved models):
```bash
# Generate speech using saved voice models (very fast)
python -m voice_clone_tts --load-models output/voice_models --text "Your new text here" --backend coqui --use-cpu

# With custom output directory
python -m voice_clone_tts --load-models my_output/voice_models --text "Hello world" --output-dir my_output --backend coqui --use-cpu

# Using custom models directory
python -m voice_clone_tts --load-models /path/to/custom/models --text "Your text" --backend coqui --use-cpu
```

#### Legacy usage (full pipeline each time):
```bash
# Basic usage with CPU processing (slower, processes everything each time)
python -m voice_clone_tts input_audio.wav --method local --backend coqui --use-cpu

# Using HuggingFace models (requires token)
python -m voice_clone_tts input_audio.wav --method huggingface --hf-token YOUR_TOKEN --backend coqui

# Clean previous output files before processing (preserves directory structure)
python -m voice_clone_tts input_audio.wav --method local --backend coqui --use-cpu --clean

# Note: Use WAV files for best results. WebM/MP4 files require ffmpeg for conversion
```

### Development Commands
```bash
# Run tests
pytest

# Code formatting
black .

# Linting
flake8 .

# Install in development mode
pip install -e .
```

## Key Implementation Details

### Backend Selection
- **Coqui TTS**: Supports voice cloning with reference audio, falls back to standard TTS
- **pyttsx3**: Local TTS without voice cloning, cycles through available system voices
- **espeak**: Command-line based TTS
- **auto**: Automatically selects best available backend

### Speaker Separation Methods
- **local**: Uses spectral clustering on MFCC features, spectral centroid, rolloff, and zero-crossing rate
- **huggingface**: Uses pyannote.audio models for more accurate speaker diarization

### Optional Dependencies
The project uses optional dependencies for different backends:
- `[coqui]`: For Coqui TTS support
- `[huggingface]`: For pyannote.audio models
- `[local]`: For pyttsx3 support
- `[full]`: All backends

### Error Handling
All backends have fallback mechanisms that generate silent audio files when TTS operations fail, ensuring the pipeline continues to completion.

## File Structure Notes

- `pyproject.toml`: Main configuration with optional dependencies
- `setup.py`: Legacy setup file (pyproject.toml is preferred)
- `output/`: Default directory for generated audio files and voice models
- Audio files are processed in WAV format at 16kHz sample rate
- Generated files follow naming convention: `speaker_N_synthesis.wav`

## Development Notes

- The system is optimized for CPU usage to ensure compatibility across different environments
- Audio processing uses librosa for feature extraction and scipy for filtering
- Voice cloning quality depends on the backend used and quality of reference audio
- Local speaker separation works best with clear, distinct speakers and adequate audio duration