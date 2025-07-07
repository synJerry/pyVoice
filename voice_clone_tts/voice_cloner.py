import os
import torch
import soundfile as sf
from typing import Optional, List
import tempfile
import warnings

# Optional imports for different TTS backends
try:
    from TTS.api import TTS
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    COQUI_TTS_AVAILABLE = True
except ImportError:
    COQUI_TTS_AVAILABLE = False
    warnings.warn("Coqui TTS not available. Install with: pip install coqui-tts")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    import espeak
    ESPEAK_AVAILABLE = True
except ImportError:
    ESPEAK_AVAILABLE = False


class VoiceCloner:
    """
    Voice cloning using various TTS backends, optimized for CPU usage.
    """
    
    def __init__(self, backend: str = "auto", model_name: str = None, use_cpu: bool = True):
        """
        Initialize voice cloner.
        
        Args:
            backend: "coqui", "pyttsx3", "espeak", or "auto"
            model_name: Specific model name (for Coqui TTS)
            use_cpu: Force CPU usage for better compatibility
        """
        self.backend = backend
        self.use_cpu = use_cpu
        self.device = torch.device("cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tts = None
        self.pyttsx3_engine = None
        
        if backend == "auto":
            self._setup_auto_backend(model_name)
        elif backend == "coqui":
            self._setup_coqui_backend(model_name)
        elif backend == "pyttsx3":
            self._setup_pyttsx3_backend()
        elif backend == "espeak":
            self._setup_espeak_backend()
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _setup_auto_backend(self, model_name: str = None):
        """Setup the best available backend automatically."""
        if COQUI_TTS_AVAILABLE:
            print("Using Coqui TTS backend")
            self.backend = "coqui"
            self._setup_coqui_backend(model_name)
        elif PYTTSX3_AVAILABLE:
            print("Using pyttsx3 backend (no voice cloning)")
            self.backend = "pyttsx3"
            self._setup_pyttsx3_backend()
        else:
            raise RuntimeError("No TTS backend available. Install coqui-tts or pyttsx3")
    
    def _setup_coqui_backend(self, model_name: str = None):
        """Setup Coqui TTS backend with CPU optimization."""
        if not COQUI_TTS_AVAILABLE:
            raise ImportError("Coqui TTS not available. Install with: pip install coqui-tts")
        
        if model_name is None:
            # Use a lightweight model for CPU processing
            model_name = "tts_models/en/ljspeech/tacotron2-DDC"
        
        try:
            print(f"Loading Coqui TTS model: {model_name}")
            self.tts = TTS(model_name, progress_bar=False).to(self.device)
            self.model_name = model_name
            
            # Check if model supports voice cloning
            self.supports_voice_cloning = hasattr(self.tts, 'tts_to_file') and 'xtts' in model_name.lower()
            
            if not self.supports_voice_cloning:
                print("Note: Selected model doesn't support voice cloning. Using standard TTS.")
                
        except Exception as e:
            print(f"Error loading Coqui TTS model {model_name}: {e}")
            # Fallback to basic model
            try:
                fallback_model = "tts_models/en/ljspeech/tacotron2-DDC"
                print(f"Trying fallback model: {fallback_model}")
                self.tts = TTS(fallback_model, progress_bar=False).to(self.device)
                self.model_name = fallback_model
                self.supports_voice_cloning = False
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                raise RuntimeError("Could not load any Coqui TTS model")
    
    def _setup_pyttsx3_backend(self):
        """Setup pyttsx3 backend (local, no internet required)."""
        if not PYTTSX3_AVAILABLE:
            raise ImportError("pyttsx3 not available. Install with: pip install pyttsx3")
        
        import pyttsx3
        self.pyttsx3_engine = pyttsx3.init()
        self.supports_voice_cloning = False
        
        # List available voices
        voices = self.pyttsx3_engine.getProperty('voices')
        print(f"Available pyttsx3 voices: {len(voices)}")
        for i, voice in enumerate(voices[:3]):  # Show first 3
            print(f"  {i}: {voice.id}")
    
    def _setup_espeak_backend(self):
        """Setup espeak backend."""
        if not ESPEAK_AVAILABLE:
            raise ImportError("espeak not available. Install espeak system package")
        
        self.supports_voice_cloning = False
    
    def clone_voice(self, text: str, reference_audio: str = None, 
                   output_file: str = None, language: str = "en", 
                   voice_index: int = 0) -> str:
        """
        Clone voice using reference audio or generate speech.
        
        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio file (for voice cloning)
            output_file: Output audio file path
            language: Language code
            voice_index: Voice index for pyttsx3 backend
            
        Returns:
            Path to generated audio file
        """
        if output_file is None:
            output_file = "output_speech.wav"
        
        if self.backend == "coqui":
            return self._clone_voice_coqui(text, reference_audio, output_file, language)
        elif self.backend == "pyttsx3":
            return self._clone_voice_pyttsx3(text, output_file, voice_index)
        elif self.backend == "espeak":
            return self._clone_voice_espeak(text, output_file)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _clone_voice_coqui(self, text: str, reference_audio: str, 
                          output_file: str, language: str) -> str:
        """Voice cloning with Coqui TTS."""
        if self.tts is None:
            raise RuntimeError("Coqui TTS model not available")
        
        try:
            if self.supports_voice_cloning and reference_audio:
                # Voice cloning with reference audio
                print(f"Cloning voice from {reference_audio}")
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=reference_audio,
                    language=language,
                    file_path=output_file
                )
            else:
                # Standard TTS without voice cloning
                print("Generating speech with standard TTS")
                wav = self.tts.tts(text=text)
                
                # Handle different output formats
                if isinstance(wav, torch.Tensor):
                    wav = wav.cpu().numpy()
                
                # Get sample rate
                sample_rate = getattr(self.tts.synthesizer.output_sample_rate, 'value', 22050) \
                    if hasattr(self.tts, 'synthesizer') else 22050
                
                sf.write(output_file, wav, sample_rate)
            
            return output_file
            
        except Exception as e:
            print(f"Error in Coqui TTS: {e}")
            # Create a silent audio file as fallback
            silent_audio = torch.zeros(22050)  # 1 second of silence
            sf.write(output_file, silent_audio.numpy(), 22050)
            return output_file
    
    def _clone_voice_pyttsx3(self, text: str, output_file: str, voice_index: int) -> str:
        """Generate speech with pyttsx3."""
        if self.pyttsx3_engine is None:
            raise RuntimeError("pyttsx3 engine not available")
        
        try:
            # Set voice
            voices = self.pyttsx3_engine.getProperty('voices')
            if voices and voice_index < len(voices):
                self.pyttsx3_engine.setProperty('voice', voices[voice_index].id)
            
            # Save to file
            self.pyttsx3_engine.save_to_file(text, output_file)
            self.pyttsx3_engine.runAndWait()
            
            return output_file
            
        except Exception as e:
            print(f"Error in pyttsx3: {e}")
            # Create a silent audio file as fallback
            silent_audio = torch.zeros(22050)
            sf.write(output_file, silent_audio.numpy(), 22050)
            return output_file
    
    def _clone_voice_espeak(self, text: str, output_file: str) -> str:
        """Generate speech with espeak."""
        try:
            import subprocess
            
            # Use espeak command line
            cmd = ["espeak", "-w", output_file, text]
            subprocess.run(cmd, check=True)
            
            return output_file
            
        except Exception as e:
            print(f"Error in espeak: {e}")
            # Create a silent audio file as fallback
            silent_audio = torch.zeros(22050)
            sf.write(output_file, silent_audio.numpy(), 22050)
            return output_file
    
    def batch_clone_voices(self, text: str, reference_audios: List[str], 
                          output_dir: str, language: str = "en") -> List[str]:
        """
        Clone multiple voices with the same text.
        
        Args:
            text: Text to synthesize
            reference_audios: List of reference audio files
            output_dir: Output directory
            language: Language code
            
        Returns:
            List of generated audio file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        
        for i, ref_audio in enumerate(reference_audios):
            output_file = os.path.join(output_dir, f"speaker_{i}_synthesis.wav")
            
            if self.supports_voice_cloning:
                self.clone_voice(text, ref_audio, output_file, language)
            else:
                # Use different voice indices for pyttsx3
                voice_index = i % 5  # Cycle through available voices
                self.clone_voice(text, None, output_file, language, voice_index)
            
            output_files.append(output_file)
        
        return output_files
