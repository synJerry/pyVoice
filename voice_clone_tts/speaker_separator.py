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
import glob
import shutil
import json

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
    Separates speakers from audio using local processing, HuggingFace models, or AWS Transcribe diarization.
    """
    
    def __init__(self, method: str = "local", huggingface_token: str = None, aws_transcribe_file: str = None):
        """
        Initialize the speaker separator.
        
        Args:
            method: "local", "huggingface", or "aws" for separation method
            huggingface_token: HuggingFace token (only needed for huggingface method)
            aws_transcribe_file: Path to AWS Transcribe JSON file (only needed for aws method)
        """
        self.method = method
        self.huggingface_token = huggingface_token
        self.aws_transcribe_file = aws_transcribe_file
        self.pipeline = None
        self.embedding_model = None
        
        if method == "huggingface":
            self._setup_hf_models()
        elif method == "local":
            print("Using local CPU-based speaker separation")
        elif method == "aws":
            if not aws_transcribe_file:
                raise ValueError("AWS Transcribe file path required for 'aws' method")
            print(f"Using AWS Transcribe diarization from {aws_transcribe_file}")
    
    def _setup_hf_models(self):
        """Setup pyannote models for speaker diarization."""
        if not PYANNOTE_AVAILABLE:
            raise ImportError("pyannote.audio not available. Install with: pip install pyannote.audio")
        
        try:
            # Clear SpeechBrain cache to avoid compatibility issues
            self._clear_speechbrain_cache()
            
            # Suppress SpeechBrain deprecation warnings during initialization
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*deprecated.*")
                warnings.filterwarnings("ignore", message=".*speechbrain.inference.*")
                
                # Initialize speaker diarization pipeline - try latest model first
                try:
                    self.pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization@2.1",
                        use_auth_token=self.huggingface_token
                    )
                    print("Using pyannote/speaker-diarization@2.1")
                except:
                    # Fallback to older model if latest not available
                    self.pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=self.huggingface_token
                    )
                    print("Using pyannote/speaker-diarization-3.1")
                
                # Initialize speaker embedding model
                self.embedding_model = PretrainedSpeakerEmbedding(
                    "speechbrain/spkrec-ecapa-voxceleb",
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
        except Exception as e:
            print(f"Warning: Could not load pyannote models: {e}")
            print("You may need to accept user agreement at https://huggingface.co/pyannote/speaker-diarization-3.1")
    
    def separate_speakers(self, audio_file: str, num_speakers: int = 2, target_sr: int = 16000) -> Dict[str, np.ndarray]:
        """
        Separate speakers from audio file.
        
        Args:
            audio_file: Path to audio file
            num_speakers: Expected number of speakers
            target_sr: Target sample rate for processing
            
        Returns:
            Dictionary mapping speaker IDs to audio arrays
        """
        if self.method == "huggingface":
            speaker_audio = self._separate_speakers_hf(audio_file, num_speakers, target_sr)
        elif self.method == "aws":
            speaker_audio = self._separate_speakers_aws(audio_file, target_sr)
        else:
            speaker_audio = self._separate_speakers_local(audio_file, num_speakers, target_sr)
        
        # Report detected speakers with detailed information
        detected_count = len(speaker_audio)
        print(f"🎤 Detected {detected_count} speaker(s) in audio:")
        
        # Get detailed speaker information
        speaker_info = self.get_speaker_info(speaker_audio, sr=target_sr)
        
        for speaker_id, info in speaker_info.items():
            duration = info["duration_seconds"]
            activity = info["relative_activity"]
            activity_desc = "Very Active" if activity > 0.8 else "Active" if activity > 0.5 else "Moderate" if activity > 0.3 else "Quiet"
            print(f"  - {speaker_id}: {duration:.1f}s audio, {activity_desc} ({activity:.1%} relative activity)")
        
        return speaker_audio
    
    def _separate_speakers_hf(self, audio_file: str, num_speakers: int, target_sr: int = 16000) -> Dict[str, np.ndarray]:
        """HuggingFace-based speaker separation."""
        if self.pipeline is None:
            raise RuntimeError("Speaker diarization pipeline not available")
        
        # Perform speaker diarization
        diarization = self.pipeline(audio_file, num_speakers=num_speakers)
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=target_sr)
        
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
        
        print(f"HuggingFace separation: Found {len(speaker_audio)} speakers")
        
        return speaker_audio
    
    def _separate_speakers_aws(self, audio_file: str, target_sr: int = 16000) -> Dict[str, np.ndarray]:
        """
        AWS Transcribe-based speaker separation using diarization timestamps.
        """
        print("Using AWS Transcribe diarization for precise speaker separation")
        
        try:
            # Load AWS Transcribe JSON
            with open(self.aws_transcribe_file, 'r', encoding='utf-8') as f:
                aws_data = json.load(f)
            
            # Extract speaker labels
            if "results" not in aws_data or "speaker_labels" not in aws_data["results"]:
                raise ValueError("Invalid AWS Transcribe JSON: missing speaker_labels")
            
            speaker_labels = aws_data["results"]["speaker_labels"]
            segments = speaker_labels.get("segments", [])
            
            if not segments:
                raise ValueError("No speaker segments found in AWS Transcribe JSON")
            
            # Load audio
            audio, sr = librosa.load(audio_file, sr=target_sr)
            
            # Group segments by speaker
            speaker_segments = {}
            for segment in segments:
                speaker_label = segment["speaker_label"]
                start_time = float(segment["start_time"])
                end_time = float(segment["end_time"])
                
                # Convert time to sample indices
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                # Ensure indices are within audio bounds
                start_sample = max(0, start_sample)
                end_sample = min(len(audio), end_sample)
                
                if end_sample > start_sample:  # Valid segment
                    if speaker_label not in speaker_segments:
                        speaker_segments[speaker_label] = []
                    
                    speaker_segments[speaker_label].append((start_sample, end_sample))
            
            # Extract and concatenate audio for each speaker
            speaker_audio = {}
            for speaker_label, segments in speaker_segments.items():
                # Sort segments by start time
                segments.sort(key=lambda x: x[0])
                
                # Extract audio segments
                audio_segments = []
                for start_sample, end_sample in segments:
                    segment_audio = audio[start_sample:end_sample]
                    if len(segment_audio) > 0:
                        audio_segments.append(segment_audio)
                
                if audio_segments:
                    # Convert AWS speaker label to internal format
                    internal_speaker_id = speaker_label.replace("spk_", "speaker_")
                    speaker_audio[internal_speaker_id] = np.concatenate(audio_segments)
            
            # Report detected speakers count
            detected_speakers = speaker_labels.get("speakers", len(speaker_audio))
            print(f"AWS Transcribe detected {detected_speakers} speakers")
            
            return speaker_audio
            
        except FileNotFoundError:
            raise ValueError(f"AWS Transcribe file not found: {self.aws_transcribe_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in AWS Transcribe file: {self.aws_transcribe_file}")
        except Exception as e:
            raise ValueError(f"Error processing AWS Transcribe file: {e}")
    
    def _separate_speakers_local(self, audio_file: str, num_speakers: int, target_sr: int = 16000) -> Dict[str, np.ndarray]:
        """
        Improved local CPU-based speaker separation using enhanced feature extraction.
        """
        print("Using enhanced local speaker separation with improved features")
        
        # Load audio with better preprocessing
        audio, sr = librosa.load(audio_file, sr=target_sr)
        
        # Apply noise reduction and normalization
        audio = librosa.effects.preemphasis(audio)
        audio = librosa.util.normalize(audio)
        
        # Parameters for windowing - smaller windows for better resolution
        window_size = int(1.0 * sr)  # 1 second windows
        hop_size = int(0.25 * sr)    # 0.25 second hop (75% overlap)
        
        # Extract enhanced features for each window
        features = []
        timestamps = []
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            
            # Skip very quiet segments
            if np.mean(np.abs(window)) < 0.01:
                continue
            
            # Skip segments that are too short for meaningful analysis
            if len(window) < 1024:  # Minimum ~64ms at 16kHz
                continue
            
            # Extract comprehensive features with safe FFT sizing
            try:
                feature_result = self._safe_feature_extraction(window, sr)
                if feature_result is None:
                    continue
                
                mfccs, chroma, tonnetz, spectral_centroid, spectral_rolloff, spectral_bandwidth, zero_crossing_rate = feature_result
            except Exception as e:
                print(f"Warning: Feature extraction failed for segment at {i/sr:.2f}s: {e}")
                continue
            
            # Combine enhanced features
            combined_features = np.concatenate([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.mean(chroma, axis=1),
                np.std(chroma, axis=1),
                np.mean(tonnetz, axis=1),
                np.atleast_1d(np.mean(spectral_centroid)),
                np.atleast_1d(np.std(spectral_centroid)),
                np.atleast_1d(np.mean(spectral_rolloff)),
                np.atleast_1d(np.std(spectral_rolloff)),
                np.atleast_1d(np.mean(spectral_bandwidth)),
                np.atleast_1d(np.std(spectral_bandwidth)),
                np.atleast_1d(np.mean(zero_crossing_rate)),
                np.atleast_1d(np.std(zero_crossing_rate))
            ])
            
            features.append(combined_features)
            timestamps.append(i / sr)
        
        # Convert to numpy array
        features = np.array(features)
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Apply PCA for dimensionality reduction if needed
        if features_normalized.shape[1] > 50:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50, random_state=42)
            features_normalized = pca.fit_transform(features_normalized)
        
        # Use improved clustering with multiple attempts
        best_kmeans = None
        best_inertia = float('inf')
        
        for _ in range(5):  # Multiple random initializations
            kmeans = KMeans(n_clusters=num_speakers, random_state=None, n_init=10, max_iter=300)
            kmeans.fit(features_normalized)
            
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_kmeans = kmeans
        
        labels = best_kmeans.labels_
        print(f"Clustering completed with inertia: {best_inertia:.2f}")
        
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
        
        print(f"Local separation: Found {len(speaker_audio)} speakers after post-processing")
        
        return speaker_audio
    
    def _clear_speechbrain_cache(self):
        """Clear SpeechBrain cache to avoid compatibility issues."""
        try:
            # Try to import pyannote cache directory
            from pyannote.audio.core.model import CACHE_DIR
            speechbrain_cache_pattern = os.path.join(CACHE_DIR, "speechbrain", "*")
            
            # Remove all SpeechBrain cache files
            for cache_file in glob.glob(speechbrain_cache_pattern):
                try:
                    if os.path.isfile(cache_file):
                        os.remove(cache_file)
                    elif os.path.isdir(cache_file):
                        shutil.rmtree(cache_file)
                except OSError:
                    # Ignore errors if files are in use or permission denied
                    pass
        except ImportError:
            # If pyannote is not available or CACHE_DIR is not accessible, skip cache clearing
            pass
        except Exception:
            # Ignore any other cache clearing errors
            pass
    
    def _safe_feature_extraction(self, signal, sr=16000):
        """
        Safely extract features from audio signal with dynamic FFT sizing.
        
        Args:
            signal: Audio signal array
            sr: Sample rate
            
        Returns:
            Tuple of extracted features or None if extraction fails
        """
        if len(signal) == 0:
            return None
        
        # Calculate appropriate FFT size based on signal length
        # Use power of 2 for optimal FFT performance
        max_fft_size = len(signal) // 2  # FFT size should be less than half signal length
        n_fft = min(2048, max_fft_size)
        
        # Ensure minimum FFT size
        if n_fft < 32:
            # Pad signal to minimum size
            signal = np.pad(signal, (0, 64 - len(signal)), 'constant')
            n_fft = 64
        
        # Ensure n_fft is power of 2
        n_fft = 2 ** int(np.log2(n_fft))
        
        # Calculate hop length as 1/4 of FFT size
        hop_length = max(1, n_fft // 4)
        
        try:
            # Extract STFT-based features with n_fft parameter
            mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
            spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(signal, hop_length=hop_length)
            chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
            
            # Extract tonnetz using pre-computed chroma (no n_fft parameter)
            try:
                tonnetz = librosa.feature.tonnetz(chroma=chroma)
            except Exception as e:
                print(f"Warning: Tonnetz extraction failed, using zeros: {e}")
                # Create dummy tonnetz features with correct dimensions
                tonnetz = np.zeros((6, chroma.shape[1]))
            
            return mfccs, chroma, tonnetz, spectral_centroid, spectral_rolloff, spectral_bandwidth, zero_crossing_rate
            
        except Exception as e:
            print(f"Feature extraction failed with n_fft={n_fft}, signal_length={len(signal)}: {e}")
            return None
    
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
    
    def get_speaker_info(self, speaker_audio: Dict[str, np.ndarray], sr: int = 16000) -> Dict[str, Dict]:
        """
        Get detailed information about detected speakers.
        
        Args:
            speaker_audio: Dictionary mapping speaker IDs to audio arrays
            sr: Sample rate
            
        Returns:
            Dictionary with speaker information
        """
        speaker_info = {}
        
        for speaker_id, audio in speaker_audio.items():
            duration = len(audio) / sr
            # Calculate RMS energy as a proxy for voice activity
            rms_energy = np.sqrt(np.mean(audio ** 2))
            
            speaker_info[speaker_id] = {
                "duration_seconds": duration,
                "samples": len(audio),
                "rms_energy": rms_energy,
                "relative_activity": rms_energy / max([np.sqrt(np.mean(a ** 2)) for a in speaker_audio.values()])
            }
        
        return speaker_info
