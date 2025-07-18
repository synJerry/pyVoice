import os
import torch
import soundfile as sf
from typing import Optional, List, Dict
import tempfile
import warnings
import json
import shutil
from .filesystem_manager import FileSystemManager

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
    
    def __init__(self, backend: str = "auto", model_name: str = None, use_cpu: bool = True, 
                 filesystem_manager: FileSystemManager = None):
        """
        Initialize voice cloner.
        
        Args:
            backend: "coqui", "pyttsx3", "espeak", or "auto"
            model_name: Specific model name (for Coqui TTS)
            use_cpu: Force CPU usage for better compatibility
            filesystem_manager: FileSystemManager instance for optimized I/O
        """
        self.backend = backend
        self.use_cpu = use_cpu
        self.device = torch.device("cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tts = None
        self.fs_manager = filesystem_manager or FileSystemManager()
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
            # Use XTTS model for voice cloning capability
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        
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
            # Fallback to basic XTTS model
            try:
                fallback_model = "tts_models/multilingual/multi-dataset/xtts_v2"
                print(f"Trying fallback model: {fallback_model}")
                self.tts = TTS(fallback_model, progress_bar=False).to(self.device)
                self.model_name = fallback_model
                self.supports_voice_cloning = True
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
            if self.supports_voice_cloning and reference_audio and self.fs_manager.file_exists(reference_audio):
                # Voice cloning with reference audio
                print(f"Cloning voice from {reference_audio}")
                
                # Check if reference audio has sufficient duration
                import librosa
                ref_audio, _ = librosa.load(reference_audio, sr=None)
                duration = len(ref_audio) / 16000  # Assuming 16kHz
                
                if duration < 3.0:  # Need at least 3 seconds for good voice cloning
                    print(f"Warning: Reference audio is only {duration:.1f}s. Voice cloning may be poor quality.")
                
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=reference_audio,
                    language=language,
                    file_path=output_file
                )
            else:
                # Standard TTS without voice cloning
                if reference_audio and not self.fs_manager.file_exists(reference_audio):
                    print(f"Warning: Reference audio file not found: {reference_audio}")
                if not self.supports_voice_cloning:
                    print("Warning: Current model doesn't support voice cloning")
                print("Generating speech with standard TTS")
                wav = self.tts.tts(text=text)
                
                # Handle different output formats
                if isinstance(wav, torch.Tensor):
                    wav = wav.cpu().numpy()
                
                # Get sample rate
                sample_rate = getattr(self.tts.synthesizer.output_sample_rate, 'value', 22050) \
                    if hasattr(self.tts, 'synthesizer') else 22050
                
                sf.write(output_file, wav, sample_rate, subtype='PCM_16')
            
            return output_file
            
        except Exception as e:
            print(f"Error in Coqui TTS: {e}")
            # Create a silent audio file as fallback
            silent_audio = torch.zeros(22050)  # 1 second of silence
            sf.write(output_file, silent_audio.numpy(), 22050, subtype='PCM_16')
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
            sf.write(output_file, silent_audio.numpy(), 22050, subtype='PCM_16')
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
            sf.write(output_file, silent_audio.numpy(), 22050, subtype='PCM_16')
            return output_file
    
    def _chunk_text(self, text: str, max_chars: int = 200) -> List[str]:
        """
        Split text into chunks that respect sentence boundaries and character limits.
        
        Args:
            text: Text to split
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chars:
            return [text]
        
        # Split by sentences first
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed limit, start new chunk
            if len(current_chunk) + len(sentence) + 1 > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence is too long, split by words
                    words = sentence.split()
                    for word in words:
                        if len(current_chunk) + len(word) + 1 > max_chars:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = word
                            else:
                                # Single word is too long, force split
                                chunks.append(word[:max_chars])
                                current_chunk = word[max_chars:]
                        else:
                            current_chunk += " " + word if current_chunk else word
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _concatenate_audio_files(self, audio_files: List[str], output_file: str) -> str:
        """
        Concatenate multiple audio files into one.
        
        Args:
            audio_files: List of audio file paths
            output_file: Output file path
            
        Returns:
            Path to concatenated audio file
        """
        import soundfile as sf
        
        combined_audio = []
        sample_rate = None
        
        for audio_file in audio_files:
            if self.fs_manager.file_exists(audio_file):
                audio, sr = sf.read(audio_file)
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    # Resample if needed
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                
                combined_audio.append(audio)
                # Add small pause between chunks
                pause = np.zeros(int(0.2 * sample_rate))  # 200ms pause
                combined_audio.append(pause)
        
        if combined_audio:
            final_audio = np.concatenate(combined_audio)
            sf.write(output_file, final_audio, sample_rate, subtype='PCM_16')
            
            # Clean up temporary files
            for temp_file in audio_files:
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
        
        return output_file

    def batch_clone_voices(self, text: str, reference_audios: List[str], 
                          output_dir: str, language: str = "en", output_format: str = "wav") -> List[str]:
        """
        Clone multiple voices with the same text, handling long text by chunking.
        
        Args:
            text: Text to synthesize
            reference_audios: List of reference audio files
            output_dir: Output directory
            language: Language code
            output_format: Output audio format ("wav" or "mp3")
            
        Returns:
            List of generated audio file paths
        """
        self.fs_manager.ensure_directory(output_dir)
        output_files = []
        
        # Check if text needs chunking (for Coqui TTS character limit)
        max_chars = 200 if self.backend == "coqui" else 1000
        text_chunks = self._chunk_text(text, max_chars)
        
        if len(text_chunks) > 1:
            print(f"Text is {len(text)} characters, splitting into {len(text_chunks)} chunks")
        
        for i, ref_audio in enumerate(reference_audios):
            # Always generate WAV first, then convert if needed
            wav_output_file = os.path.join(output_dir, f"speaker_{i}_synthesis.wav")
            final_output_file = os.path.join(output_dir, f"speaker_{i}_synthesis.{output_format}")
            
            if len(text_chunks) == 1:
                # Single chunk - process normally
                if self.supports_voice_cloning:
                    self.clone_voice(text, ref_audio, wav_output_file, language)
                else:
                    voice_index = i % 5
                    self.clone_voice(text, None, wav_output_file, language, voice_index)
            else:
                # Multiple chunks - process each and concatenate
                chunk_files = []
                temp_dir = os.path.join(output_dir, f"temp_speaker_{i}")
                self.fs_manager.ensure_directory(temp_dir)
                
                for j, chunk in enumerate(text_chunks):
                    chunk_file = os.path.join(temp_dir, f"chunk_{j}.wav")
                    
                    if self.supports_voice_cloning:
                        self.clone_voice(chunk, ref_audio, chunk_file, language)
                    else:
                        voice_index = i % 5
                        self.clone_voice(chunk, None, chunk_file, language, voice_index)
                    
                    chunk_files.append(chunk_file)
                
                # Concatenate chunks
                self._concatenate_audio_files(chunk_files, wav_output_file)
                
                # Clean up temporary directory
                try:
                    shutil.rmtree(temp_dir)
                except OSError:
                    pass
            
            # Convert to MP3 if requested
            if output_format == "mp3":
                from .audio_processor import AudioProcessor
                AudioProcessor.convert_to_mp3(wav_output_file, final_output_file)
                # Remove the temporary WAV file
                self.fs_manager.cleanup_temp_files([wav_output_file])
            else:
                final_output_file = wav_output_file
            
            output_files.append(final_output_file)
        
        return output_files
    
    def save_voice_models(self, reference_audios: List[str], model_dir: str) -> Dict[str, str]:
        """
        Save voice models for reuse.
        
        Args:
            reference_audios: List of reference audio files
            model_dir: Directory to save voice models
            
        Returns:
            Dictionary mapping speaker IDs to model paths
        """
        self.fs_manager.ensure_directory(model_dir)
        
        voice_models = {}
        
        # Batch validate reference files
        file_exists_map = self.fs_manager.batch_file_exists(reference_audios)
        
        for i, ref_audio in enumerate(reference_audios):
            if not file_exists_map.get(ref_audio, False):
                print(f"Warning: Reference audio not found: {ref_audio}")
                continue
            
            # Extract speaker_id from filename (e.g., "speaker_0.wav" -> "speaker_0")
            base_name = os.path.splitext(os.path.basename(ref_audio))[0]
            speaker_id = base_name if base_name.startswith("speaker_") else f"speaker_{i}"
            
            # Copy reference audio to model directory with consistent naming
            model_audio_path = os.path.join(model_dir, f"{speaker_id}_reference.wav")
            shutil.copy2(ref_audio, model_audio_path)
            
            # Save model metadata
            metadata = {
                "speaker_id": speaker_id,
                "reference_audio": f"{speaker_id}_reference.wav",
                "backend": self.backend,
                "model_name": getattr(self, 'model_name', 'unknown'),
                "created_at": __import__('datetime').datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(model_dir, f"{speaker_id}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            voice_models[speaker_id] = model_audio_path
            print(f"Saved voice model for {speaker_id} at {model_audio_path}")
        
        return voice_models
    
    def load_voice_models(self, model_dir: str) -> Dict[str, str]:
        """
        Load saved voice models.
        
        Args:
            model_dir: Directory containing voice models
            
        Returns:
            Dictionary mapping speaker IDs to reference audio paths
        """
        voice_models = {}
        
        if not self.fs_manager.directory_exists(model_dir):
            print(f"Voice model directory not found: {model_dir}")
            return voice_models
        
        voice_models = self.fs_manager.get_model_files(model_dir)
        
        if not voice_models:
            print(f"No valid voice models found in {model_dir}")
        else:
            print(f"Loaded {len(voice_models)} voice models from {model_dir}")
            for speaker_id in voice_models:
                print(f"  - {speaker_id}")
        
        return voice_models
    
    def generate_from_models(self, text: str, voice_models: Dict[str, str], 
                           output_dir: str, language: str = "en", output_format: str = "wav") -> List[str]:
        """
        Generate speech using saved voice models, handling long text by chunking.
        
        Args:
            text: Text to synthesize
            voice_models: Dictionary mapping speaker IDs to reference audio paths
            output_dir: Output directory
            language: Language code
            output_format: Output audio format ("wav" or "mp3")
            
        Returns:
            List of generated audio file paths
        """
        self.fs_manager.ensure_directory(output_dir)
        output_files = []
        
        # Check if text needs chunking
        max_chars = 200 if self.backend == "coqui" else 1000
        text_chunks = self._chunk_text(text, max_chars)
        
        if len(text_chunks) > 1:
            print(f"Text is {len(text)} characters, splitting into {len(text_chunks)} chunks")
        
        for speaker_id, ref_audio_path in voice_models.items():
            # Always generate WAV first, then convert if needed
            wav_output_file = os.path.join(output_dir, f"{speaker_id}_synthesis.wav")
            final_output_file = os.path.join(output_dir, f"{speaker_id}_synthesis.{output_format}")
            
            print(f"Generating speech for {speaker_id}...")
            
            if len(text_chunks) == 1:
                # Single chunk - process normally
                self.clone_voice(text, ref_audio_path, wav_output_file, language)
            else:
                # Multiple chunks - process each and concatenate
                chunk_files = []
                temp_dir = os.path.join(output_dir, f"temp_{speaker_id}")
                self.fs_manager.ensure_directory(temp_dir)
                
                for j, chunk in enumerate(text_chunks):
                    chunk_file = os.path.join(temp_dir, f"chunk_{j}.wav")
                    self.clone_voice(chunk, ref_audio_path, chunk_file, language)
                    chunk_files.append(chunk_file)
                
                # Concatenate chunks
                self._concatenate_audio_files(chunk_files, wav_output_file)
                
                # Clean up temporary directory
                try:
                    shutil.rmtree(temp_dir)
                except OSError:
                    pass
            
            # Convert to MP3 if requested
            if output_format == "mp3":
                from .audio_processor import AudioProcessor
                AudioProcessor.convert_to_mp3(wav_output_file, final_output_file)
                # Remove the temporary WAV file
                self.fs_manager.cleanup_temp_files([wav_output_file])
            else:
                final_output_file = wav_output_file
            
            output_files.append(final_output_file)
        
        return output_files
