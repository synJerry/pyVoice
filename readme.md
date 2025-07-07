Windows Setup (assumes user installed Python)

```powershell
pip install uv
# From this local dir
#uv python install 3.12
#uv python install 3.11
uv venv -p 3.11.7
#uv init
.venv\Scripts\activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# TTS Tries failed on windows
#uv pip install TTS
#uv pip install --only-binary=all TTS
# Use a fork with Windows binary wheels
uv pip install coqui-tts
uv pip install -r .\pyproject.toml --all-extras
```

Run
```powershell
python -m voice-clone-tts Syn.webm --method local --backend coqui --use-cpu
```
