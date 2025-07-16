## Windows Setup (assumes user installed Python)

### Windows Powershell Setup
```powershell
pip install uv
# From this local dir
#uv python install 3.11
uv venv -p 3.11.7
#uv init
.venv\Scripts\activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Use a TTS fork with Windows binary wheels
#uv pip install coqui-tts
uv pip install -r .\pyproject.toml --all-extras
```

#### Optional WSL Setup
Claude Code currently only supports Linux so we can set it up in WSL to access the same Windows Folder. This is pretty ugly and leads to redundant packages and ML models being installed. Better current option (until Claude Code officially supports Windows) is probably to run the whole repo and commands in a Linux container and use Vscode Containers to access.

Launch WSL `wsl -d Ubuntu-24.04` then run Linux commands
```bash
# Fix Windows PATH showing up in Linux
echo $PATH | tr ':' '\n' | grep -v "/mnt/c" | tr '\n' ':' | sed 's/:$//'
# Create python symlink if needed
sudo ln -s /usr/bin/python3 /usr/bin/python
sudo apt install -y python3-pip
# Tried pathing python/pip to install uv but easier (much uglier) was to just install uv directly
curl -LsSf https://astral.sh/uv/install.sh | sh
# Reload your shell or add to PATH
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc
# Create our WSL .venv with a different directory name
uv venv .venv-linux -p 3.11.7
source .venv-linux/bin/activate # May need to run this on subsequent WSL runs before running 'claude'
# This might get big because we have similar libraries as Windows (TODO: See 
#    if the ML libraries can be linked to keep size down)
uv pip install -r ./pyproject.toml --all-extras
```

Run
```powershell
#python -m voice_clone_tts Syn.webm --method local --backend coqui --use-cpu
python -m voice_clone_tts "Syn.wav" --method local --backend coqui --use-cpu
python -m voice_clone_tts "Syn.wav" --method local --backend coqui --use-cpu --show-speaker-info --clean
python -m voice_clone_tts "Single.wav" --num-speakers 1 --method local --backend coqui --use-cpu --show-speaker-info --clean
# If using AWS Transcribe output
python -m voice_clone_tts "Syn.wav" --aws-transcribe "transcribe.json" --backend coqui --use-cpu --clean
python -m voice_clone_tts "Syn.wav" --aws-transcribe "transcribe.json" --backend coqui --use-cpu --use-transcript --clean
python -m voice_clone_tts "Syn.wav" --aws-transcribe "transcribe.json" --backend coqui --use-cpu --use-transcript --output-format mp3 --clean
python -m voice_clone_tts "Syn.wav" --aws-transcribe "transcribe.json" --backend coqui --use-cpu --use-transcript --output-format mp3 --use-16khz --clean
#--save-models
```

## Docker attempt
From https://docs.coqui.ai/en/latest/docker_images.html
```bash
# Use maintained fork "github.com/idiap/coqui-ai-TTS" that has several fixes
#docker pull ghcr.io/coqui-ai/tts-cpu --platform linux/amd64
#docker run --rm -it -p 5002:5002 --entrypoint /bin/bash ghcr.io/coqui-ai/tts-cpu
docker pull ghcr.io/idiap/coqui-tts-cpu --platform linux/amd64
docker run --rm -it -p 5002:5002 --entrypoint /bin/bash ghcr.io/idiap/coqui-tts-cpu
# From within docker container
tts --list_models #To get the list of available models
# Start a server
python3 TTS/server/server.py --model_name tts_models/en/vctk/vits
# 148MB Model downloads, but why is ths container so big (12GB) if the model doesn't already exist locally?
```
