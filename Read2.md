# Voice Clone TTS

A Python package for separating speakers from audio and generating synthetic speech using their voices.

## Installation

### Basic Installation (CPU only, local processing)
```bash
uv add voice-clone-tts
```

### With Coqui TTS (better voice quality)
```bash
uv add voice-clone-tts[coqui]
```

### With HuggingFace models (best speaker separation)
```bash
uv add voice-clone-tts[huggingface]
```

### Full installation (all features)
```bash
uv add voice-clone-tts[full]
```

## Audio Format Support

The package supports various audio formats including:
- WAV (native support)
- WebM (requires ffmpeg)
- MP4/M4A (requires ffmpeg)
- MP3 (requires ffmpeg)
- FLAC, OGG, WMA (requires ffmpeg)

### FFmpeg Installation

**Windows:**
```bash
# Using chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**Linux:**
```bash
sudo apt install ffmpeg  # Ubuntu/Debian
sudo yum install ffmpeg  # CentOS/RHEL
```

## Usage

### Command Line (WebM support)

```bash
# Basic usage with WebM file
voice-clone-tts input_audio.webm --output-dir output --method local

# With Coqui TTS
voice-clone-tts input_audio.webm --output-dir output --method local --backend coqui

# With HuggingFace models
voice-clone-tts input_audio.webm --output-dir output --method huggingface --hf-token YOUR_TOKEN
```

### Python API

```python
from voice_clone_tts import SpeakerSeparator, VoiceCloner, AudioProcessor

# Convert WebM to WAV first
wav_file = AudioProcessor.convert_to_wav("input.webm")

# Or use auto-conversion in preprocessing
preprocessed = AudioProcessor.preprocess_audio("input.webm", auto_convert=True)

# Local CPU-based speaker separation (no HuggingFace token needed)
separator = SpeakerSeparator(method="local")
speaker_audio = separator.separate_speakers(preprocessed, num_speakers=2)

# Voice cloning with CPU-optimized backend
cloner = VoiceCloner(backend="auto", use_cpu=True)
output_file = cloner.clone_voice("Your text here", "reference.wav", "output.wav")
```

## CPU-Optimized Usage

For best CPU performance without cloud dependencies:

```python
# Local processing only
separator = SpeakerSeparator(method="local")
cloner = VoiceCloner(backend="pyttsx3", use_cpu=True)  # Or "coqui" with CPU models

# Process WebM file
wav_file = AudioProcessor.convert_to_wav("input.webm")
speaker_audio = separator.separate_speakers(wav_file, num_speakers=2)
```

## Requirements

- Python 3.8+
- PyTorch
- Coqui TTS
- pyannote.audio
- librosa
- HuggingFace token (for speaker diarization)

## Notes

- Replace SAMPLE_TEXT with your desired text content
- Ensure you have proper rights to use the audio content
- GPU acceleration recommended for faster processing
