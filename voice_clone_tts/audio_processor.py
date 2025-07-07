import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
from typing import Tuple, Optional
import os


class AudioProcessor:
    """
    Audio processing utilities for voice cloning pipeline.
    """
    
    @staticmethod
    def preprocess_audio(audio_file: str, target_sr: int = 16000, 
                        output_file: Optional[str] = None) -> str:
        """
        Preprocess audio file for speaker separation.
        
        Args:
            audio_file: Input audio file
            target_sr: Target sample rate
            output_file: Output file path (optional)
            
        Returns:
            Path to preprocessed audio file
        """
        # Load audio
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
