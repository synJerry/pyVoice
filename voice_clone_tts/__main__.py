import argparse
import os
from .speaker_separator import SpeakerSeparator
from .voice_cloner import VoiceCloner
from .audio_processor import AudioProcessor
from .filesystem_manager import FileSystemManager


# Sample text - replace with actual lyrics
SAMPLE_TEXT = """
Oh say can you see, by the dawn's early light,
What so proudly we hailed at the twilight's last gleaming...
"""  # Replace with full lyrics


def main():
    """Main function to run the voice cloning pipeline."""
    parser = argparse.ArgumentParser(description="Voice Clone TTS Pipeline")
    parser.add_argument("audio_file", nargs='?', help="Input audio file (WebM, MP4, MP3, WAV, etc.)")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--method", choices=["local", "huggingface"], default="local", 
                       help="Speaker separation method")
    parser.add_argument("--backend", choices=["auto", "coqui", "pyttsx3", "espeak"], 
                       default="auto", help="TTS backend")
    parser.add_argument("--hf-token", help="HuggingFace token (for huggingface method)")
    parser.add_argument("--text", default=SAMPLE_TEXT, help="Text to synthesize")
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--save-models", action="store_true", 
                       help="Save voice models for reuse")
    parser.add_argument("--load-models", type=str, 
                       help="Load saved voice models from directory (skip speaker separation)")
    parser.add_argument("--models-dir", default=None, 
                       help="Directory to save/load voice models (defaults to output_dir/voice_models)")
    parser.add_argument("--clean", action="store_true",
                       help="Delete existing files in output directory before processing (preserves subdirectories)")
    parser.add_argument("--suppress-warnings", action="store_true",
                       help="Suppress SpeechBrain deprecation and other non-critical warnings")
    parser.add_argument("--show-speaker-info", action="store_true",
                       help="Show detailed information about detected speakers")
    parser.add_argument("--num-speakers", type=int, default=2,
                       help="Expected number of speakers in the audio (default: 2)")
    parser.add_argument("--use-transcript", action="store_true",
                       help="Read text from transcript.txt file (overrides --text)")
    parser.add_argument("--aws-transcribe", type=str, metavar="JSON_FILE",
                       help="Use AWS Transcribe JSON output for speaker diarization (overrides --method)")
    parser.add_argument("--output-format", choices=["wav", "mp3"], default="wav",
                       help="Output audio format (default: wav)")
    parser.add_argument("--use-16khz", action="store_true",
                       help="Force 16kHz sample rate (default: use input file's sample rate)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.load_models and not args.audio_file:
        parser.error("audio_file is required unless --load-models is specified")
    
    # AWS Transcribe validation is handled automatically
    
    # Validate number of speakers (unless using AWS Transcribe which detects automatically)
    if not args.aws_transcribe:
        if args.num_speakers < 1:
            parser.error("--num-speakers must be at least 1")
        elif args.num_speakers > 10:
            print(f"Warning: {args.num_speakers} speakers is quite high. Speaker separation accuracy may decrease with many speakers.")
        elif args.num_speakers == 1:
            print("Note: With 1 speaker, the entire audio will be treated as a single speaker.")
    
    # Handle transcript file loading
    if args.use_transcript:
        try:
            with open("transcript.txt", "r", encoding="utf-8") as f:
                transcript_text = f.read().strip()
            
            if transcript_text:
                args.text = transcript_text
                print("✅ Loaded text from transcript.txt")
                print(f"📄 Text length: {len(transcript_text)} characters")
            else:
                print("⚠️  Warning: transcript.txt is empty, using default text")
                
        except FileNotFoundError:
            print("⚠️  Warning: transcript.txt not found, using default text")
        except UnicodeDecodeError:
            print("⚠️  Warning: Could not decode transcript.txt (try UTF-8 encoding), using default text")
        except Exception as e:
            print(f"⚠️  Warning: Could not read transcript.txt: {e}, using default text")
    
    # Handle warning suppression
    if args.suppress_warnings:
        import warnings
        warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*deprecated.*")
        warnings.filterwarnings("ignore", message=".*speechbrain.inference.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
    
    # Initialize filesystem manager
    fs_manager = FileSystemManager()
    
    # Handle clean flag  
    if args.clean and fs_manager.directory_exists(args.output_dir):
        print(f"Cleaning files in output directory: {args.output_dir}")
        import shutil
        import glob
        
        # Delete all files in the directory and subdirectories, but preserve directory structure
        for root, dirs, files in os.walk(args.output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except OSError as e:
                    print(f"Warning: Could not delete {file_path}: {e}")
        
        # Invalidate cache after cleaning
        fs_manager.invalidate_cache()
    
    # Create output directory
    fs_manager.ensure_directory(args.output_dir)
    
    # Set default models directory within output directory
    if args.models_dir is None:
        args.models_dir = os.path.join(args.output_dir, "voice_models")
    
    # Initialize voice cloner with filesystem manager
    cloner = VoiceCloner(backend=args.backend, use_cpu=args.use_cpu, filesystem_manager=fs_manager)
    
    # Check if loading existing models
    if args.load_models:
        print(f"Loading voice models from {args.load_models}...")
        voice_models = cloner.load_voice_models(args.load_models)
        
        if not voice_models:
            print("No voice models found. Exiting.")
            return
        
        # Generate speech using loaded models
        synthesis_dir = os.path.join(args.output_dir, "synthesis")
        
        # For loaded models, we don't have original sample rate, so use 16kHz unless user wants higher
        if args.use_16khz:
            target_output_sr = None  # Stay at 16kHz
            print("Using 16kHz for loaded models")
        else:
            # Default to 22050 Hz for better quality when using loaded models
            target_output_sr = 22050
            print("Will upsample loaded models output to 22050Hz for better quality")
        
        output_files = cloner.generate_from_models(
            args.text, 
            voice_models, 
            synthesis_dir,
            output_format=args.output_format,
            target_output_sr=target_output_sr
        )
    else:
        # Full pipeline: preprocess, separate, clone
        # Step 1: Get original sample rate for final output quality
        import librosa
        _, orig_sr = librosa.load(args.audio_file, sr=None, duration=0.1)  # Just load a small chunk to get SR
        
        print(f"Detected original sample rate: {orig_sr}Hz")
        
        # Use 16kHz for processing (TTS models expect this) but remember original SR
        processing_sr = 16000  # Always use 16kHz for processing
        output_sr = 16000 if args.use_16khz else orig_sr  # Use original SR for final output unless forced
        
        print(f"Processing at {processing_sr}Hz, output at {output_sr}Hz")
        
        # Only resample if output SR is different from processing SR
        target_output_sr = output_sr if output_sr != processing_sr else None
        if target_output_sr:
            print(f"Will resample final output to {target_output_sr}Hz")
        else:
            print("No resampling needed - output will remain at 16kHz")
        
        # Step 1: Preprocess audio (handles format conversion)
        print("Preprocessing audio...")
        preprocessed_audio = AudioProcessor.preprocess_audio(
            args.audio_file, 
            target_sr=processing_sr,
            output_file=os.path.join(args.output_dir, "preprocessed.wav"),
            auto_convert=True,
            filesystem_manager=fs_manager
        )
        
        # Step 2: Separate speakers
        print("Separating speakers...")
        
        # Determine separation method
        if args.aws_transcribe:
            # Use AWS Transcribe diarization
            separator = SpeakerSeparator(method="aws", aws_transcribe_file=args.aws_transcribe)
            print(f"🎯 Using AWS Transcribe diarization from {args.aws_transcribe}")
            speaker_audio = separator.separate_speakers(preprocessed_audio)
        else:
            # Use traditional methods
            separator = SpeakerSeparator(
                method=args.method,
                huggingface_token=args.hf_token
            )
            
            # Recommend better method if using local
            if args.method == "local":
                print("NOTE: For better speaker separation quality, consider using:")
                print("  --method huggingface --hf-token YOUR_TOKEN")
                print("  This uses state-of-the-art neural models instead of basic clustering")
            
            print(f"Attempting to separate into {args.num_speakers} speaker(s)...")
            speaker_audio = separator.separate_speakers(preprocessed_audio, num_speakers=args.num_speakers, target_sr=processing_sr)
        
        # Show detailed speaker information if requested
        if args.show_speaker_info:
            print("\n📊 Detailed Speaker Analysis:")
            speaker_info = separator.get_speaker_info(speaker_audio, sr=processing_sr)
            for speaker_id, info in speaker_info.items():
                print(f"\n{speaker_id}:")
                print(f"  Duration: {info['duration_seconds']:.2f} seconds")
                print(f"  Samples: {info['samples']:,}")
                print(f"  RMS Energy: {info['rms_energy']:.4f}")
                print(f"  Relative Activity: {info['relative_activity']:.1%}")
        
        # Save separated audio
        speaker_dir = os.path.join(args.output_dir, "speakers")
        separator.save_speaker_audio(speaker_audio, speaker_dir, sr=processing_sr)
        
        # Get reference audio files
        reference_files = []
        for speaker_id in speaker_audio.keys():
            ref_file = os.path.join(speaker_dir, f"{speaker_id}.wav")
            reference_files.append(ref_file)
        
        # Save voice models if requested or if no model files exist
        model_files_exist = fs_manager.has_model_files(args.models_dir)
        
        if args.save_models or not model_files_exist:
            print(f"Saving voice models to {args.models_dir}...")
            cloner.save_voice_models(reference_files, args.models_dir)
        
        # Generate synthetic speech
        print("Cloning voices...")
        synthesis_dir = os.path.join(args.output_dir, "synthesis")
        output_files = cloner.batch_clone_voices(
            args.text, 
            reference_files, 
            synthesis_dir,
            output_format=args.output_format,
            target_output_sr=target_output_sr
        )
    
    print(f"Voice cloning complete! Generated files:")
    for file in output_files:
        print(f"  - {file}")
    
    # Show format info
    input_format = AudioProcessor.detect_audio_format(args.audio_file)
    print(f"Input format: {input_format}")
    if input_format != 'wav':
        print(f"Converted from {input_format} to WAV for processing")


if __name__ == "__main__":
    main()
