from setuptools import setup, find_packages

setup(
    name="voice-clone-tts",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "coqui-tts>=0.22.0",
        "librosa>=0.10.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "torch>=1.13.0",
        "torchaudio>=0.13.0",
        "pyannote.audio>=3.1.0",
        "speechbrain>=0.5.0",
        "soundfile>=0.12.0",
        "matplotlib>=3.5.0",
        "pydub>=0.25.0",
    ],
    entry_points={
        "console_scripts": [
            "voice-clone-tts=voice_clone_tts.main:main",
        ],
    },
)
