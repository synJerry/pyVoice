import os
import librosa
import numpy as np
import torch
import torchaudio
import soundfile as sf
from typing import List, Tuple, Dict, Optional
import tempfile
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
import warnings

# Optional imports for HuggingFace models
try:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    warnings.warn("pyannote.audio not available. Using local processing only.")


class SpeakerSeparator:
    """
    Separates speakers from audio using local processing or HuggingFace models.
    """
    
    def __init__(self, method: str = "local", huggingface_token: str = None):
        """
        Initialize the speaker separator.
        
        Args:
            method: "local" for CPU-based processing or "huggingface" for cloud models
            huggingface_token: HuggingFace token (only needed for huggingface method)
        """
        self.method = method
        self.huggingface_token = huggingface_token
        self.pipeline = None
        self.embedding_model = None
        
        if method == "huggingface":
            self._setup_hf_models()
        elif method == "local":
            print("Using local CPU-based speaker separation")
    
    def _setup_hf_models(self):
        """Setup pyannote models for speaker diarization."""
        if not PYANNOTE_AVAILABLE:
            raise ImportError("pyannote.audio not available. Install with: pip install pyannote.audio")
        
        try:
            # Initialize speaker diarization pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.huggingface_token
            )
            
            # Initialize speaker embedding model
            self.embedding_model = PretrainedSpeakerEmbedding(
                "speechbrain/spkrec-ecapa-voxceleb",
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        except Exception as e:
            print(f"Warning: Could not load pyannote models: {e}")
            print("You may need to accept user agreement at https://huggingface.co/pyannote/speaker-diarization-3.1")
    
    def separate_speakers(self, audio_file: str, num_speakers: int = 2) -> Dict[str, np.ndarray]:
        """
        Separate speakers from audio file.
        
        Args:
            audio_file: Path to audio file
            num_speakers: Expected number of speakers
            
        Returns:
            Dictionary mapping speaker IDs to audio arrays
        """
        if self.method == "huggingface":
            return self._separate_speakers_hf(audio_file, num_speakers)
        else:
            return self._separate_speakers_local(audio_file, num_speakers)
    
    def _separate_speakers_hf(self, audio_file: str, num_speakers: int) -> Dict[str, np.ndarray]:
        """HuggingFace-based speaker separation."""
        if self.pipeline is None:
            raise RuntimeError("Speaker diarization pipeline not available")
        
        # Perform speaker diarization
        diarization = self.pipeline(audio_file, num_speakers=num_speakers)
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=16000)
        
        # Extract speaker segments
        speaker_audio = {}
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_audio:
                speaker_audio[speaker] = []
            
            # Extract audio segment
            start_sample = int(turn.start * sr)
            end_sample = int(turn.end * sr)
            segment = audio[start_sample:end_sample]
            
            speaker_audio[speaker].append(segment)
        
        # Concatenate segments for each speaker
        for speaker in speaker_audio:
            speaker_audio[speaker] = np.concatenate(speaker_audio[speaker])
        
        return speaker_audio
    
    def _separate_speakers_local(self, audio_file: str, num_speakers: int) -> Dict[str, np.ndarray]:
        """
        Local CPU-based speaker separation using spectral clustering.
        Works well when you have decent amounts of speaker audio.
        """
        print("Using local speaker separation - this works best with clear, distinct speakers")
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=16000)
        
        # Parameters for windowing
        window_size = int(2.0 * sr)  # 2 second windows
        hop_size = int(0.5 * sr)     # 0.5 second hop
        
        # Extract features for each window
        features = []
        timestamps = []
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            
            # Extract multiple features
            mfccs = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=window, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=window, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(window)
            
            # Combine features (ensure all are 1D arrays)
            combined_features = np.concatenate([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.atleast_1d(np.mean(spectral_centroid)),
                np.atleast_1d(np.mean(spectral_rolloff)),
                np.atleast_1d(np.mean(zero_crossing_rate))
            ])
            
            features.append(combined_features)
            timestamps.append(i / sr)
        
        # Convert to numpy array
        features = np.array(features)
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Cluster into speakers
        kmeans = KMeans(n_clusters=num_speakers, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_normalized)
        
        # Group audio segments by speaker
        speaker_audio = {}
        
        for i, label in enumerate(labels):
            speaker_id = f"speaker_{label}"
            if speaker_id not in speaker_audio:
                speaker_audio[speaker_id] = []
            
            # Extract audio segment
            start_sample = i * hop_size
            end_sample = min(start_sample + window_size, len(audio))
            segment = audio[start_sample:end_sample]
            
            speaker_audio[speaker_id].append(segment)
        
        # Concatenate segments for each speaker
        for speaker in speaker_audio:
            speaker_audio[speaker] = np.concatenate(speaker_audio[speaker])
        
        # Post-process: merge very short segments and clean up
        speaker_audio = self._post_process_speakers(speaker_audio, min_duration=5.0, sr=sr)
        
        return speaker_audio
    
    def _post_process_speakers(self, speaker_audio: Dict[str, np.ndarray], 
                              min_duration: float, sr: int) -> Dict[str, np.ndarray]:
        """
        Post-process speaker audio to remove very short segments and merge similar speakers.
        """
        # Remove speakers with too little audio
        filtered_speakers = {}
        
        for speaker_id, audio in speaker_audio.items():
            duration = len(audio) / sr
            if duration >= min_duration:
                filtered_speakers[speaker_id] = audio
            else:
                print(f"Removing {speaker_id} - too short ({duration:.1f}s)")
        
        # If we have more than expected speakers, merge the shortest ones
        if len(filtered_speakers) > 2:
            # Sort by duration
            speaker_durations = [(k, len(v)/sr) for k, v in filtered_speakers.items()]
            speaker_durations.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only the two longest speakers
            kept_speakers = {}
            for i, (speaker_id, duration) in enumerate(speaker_durations[:2]):
                kept_speakers[f"speaker_{i}"] = filtered_speakers[speaker_id]
            
            return kept_speakers
        
        return filtered_speakers
    
    def save_speaker_audio(self, speaker_audio: Dict[str, np.ndarray], 
                          output_dir: str, sr: int = 16000):
        """
        Save separated speaker audio to files.
        
        Args:
            speaker_audio: Dictionary of speaker audio arrays
            output_dir: Output directory
            sr: Sample rate
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for speaker_id, audio in speaker_audio.items():
            output_path = os.path.join(output_dir, f"{speaker_id}.wav")
            sf.write(output_path, audio, sr)
            print(f"Saved {speaker_id} audio to {output_path}")
