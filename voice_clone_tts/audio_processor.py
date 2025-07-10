import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
from typing import Tuple, Optional
import os
import tempfile
import ffmpeg

class AudioProcessor:
    """
    Audio processing utilities for voice cloning pipeline.
    """
    
    @staticmethod
    def preprocess_audio(audio_file: str, target_sr: int = 16000, 
                        output_file: Optional[str] = None, auto_convert: bool = True) -> str:
        """
        Preprocess audio file for speaker separation.
        
        Args:
            audio_file: Input audio file
            target_sr: Target sample rate
            output_file: Output file path (optional)
            auto_convert: Whether to automatically convert from other formats
            
        Returns:
            Path to preprocessed audio file
        """
        # Check if we need to convert the audio format
        input_format = AudioProcessor.detect_audio_format(audio_file)
        
        if auto_convert and input_format.lower() not in ['wav', 'mp3', 'flac', 'ogg']:
            # Check if ffmpeg is available
            if not AudioProcessor.check_ffmpeg_available():
                raise RuntimeError(f"ffmpeg is required to convert {input_format} files. Please install ffmpeg or convert the file to WAV/MP3 format manually.")
            
            # Convert to temporary WAV file first
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav.close()
            
            try:
                print(f"Converting {input_format} to WAV format...")
                # Use ffmpeg to convert to WAV
                stream = ffmpeg.input(audio_file)
                stream = ffmpeg.output(stream, temp_wav.name, acodec='pcm_s16le', ar=target_sr)
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                
                # Load the converted audio
                audio, sr = librosa.load(temp_wav.name, sr=target_sr)
                
                # Clean up temporary file
                os.unlink(temp_wav.name)
                
            except Exception as e:
                # Clean up temporary file on error
                try:
                    os.unlink(temp_wav.name)
                except:
                    pass
                raise RuntimeError(f"Failed to convert audio format {input_format}: {e}")
        else:
            # Load audio directly
            audio, sr = librosa.load(audio_file, sr=target_sr)
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        # Apply noise reduction (simple high-pass filter)
        audio = AudioProcessor._apply_highpass_filter(audio, sr)
        
        # Save preprocessed audio
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_file = f"{base_name}_preprocessed.wav"
        
        sf.write(output_file, audio, target_sr)
        return output_file
    
    @staticmethod
    def _apply_highpass_filter(audio: np.ndarray, sr: int, 
                              cutoff: float = 80.0) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency noise.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            cutoff: Cutoff frequency in Hz
            
        Returns:
            Filtered audio signal
        """
        nyquist = sr / 2
        normalized_cutoff = cutoff / nyquist
        b, a = butter(4, normalized_cutoff, btype='high')
        filtered_audio = filtfilt(b, a, audio)
        return filtered_audio
    
    @staticmethod
    def enhance_audio_quality(audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Enhance audio quality for better voice cloning.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Enhanced audio signal
        """
        # Normalize
        audio = librosa.util.normalize(audio)
        
        # Apply spectral centroid-based enhancement
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        mean_centroid = np.mean(spectral_centroids)
        
        # Simple spectral enhancement based on centroid
        if mean_centroid < 1500:  # If audio is too muffled
            # Apply slight high-frequency boost
            audio = AudioProcessor._apply_highpass_filter(audio, sr, cutoff=100)
        
        return audio
    
    @staticmethod
    def detect_audio_format(audio_file: str) -> str:
        """
        Detect the format of an audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            File format extension (e.g., 'wav', 'mp3', 'webm')
        """
        return os.path.splitext(audio_file)[1].lower().lstrip('.')
    
    @staticmethod
    def check_ffmpeg_available() -> bool:
        """
        Check if ffmpeg is available on the system.
        
        Returns:
            True if ffmpeg is available, False otherwise
        """
        try:
            ffmpeg.probe('')
        except ffmpeg.Error:
            return True  # ffmpeg is available but got an error (expected)
        except FileNotFoundError:
            return False  # ffmpeg not found
        return True
