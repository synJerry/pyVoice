import argparse
import os
from .speaker_separator import SpeakerSeparator
from .voice_cloner import VoiceCloner
from .audio_processor import AudioProcessor


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
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.load_models and not args.audio_file:
        parser.error("audio_file is required unless --load-models is specified")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set default models directory within output directory
    if args.models_dir is None:
        args.models_dir = os.path.join(args.output_dir, "voice_models")
    
    # Initialize voice cloner
    cloner = VoiceCloner(backend=args.backend, use_cpu=args.use_cpu)
    
    # Check if loading existing models
    if args.load_models:
        print(f"Loading voice models from {args.load_models}...")
        voice_models = cloner.load_voice_models(args.load_models)
        
        if not voice_models:
            print("No voice models found. Exiting.")
            return
        
        # Generate speech using loaded models
        synthesis_dir = os.path.join(args.output_dir, "synthesis")
        output_files = cloner.generate_from_models(
            args.text, 
            voice_models, 
            synthesis_dir
        )
    else:
        # Full pipeline: preprocess, separate, clone
        # Step 1: Preprocess audio (handles format conversion)
        print("Preprocessing audio...")
        preprocessed_audio = AudioProcessor.preprocess_audio(
            args.audio_file, 
            output_file=os.path.join(args.output_dir, "preprocessed.wav"),
            auto_convert=True
        )
        
        # Step 2: Separate speakers
        print("Separating speakers...")
        separator = SpeakerSeparator(
            method=args.method,
            huggingface_token=args.hf_token
        )
        speaker_audio = separator.separate_speakers(preprocessed_audio, num_speakers=2)
        
        # Save separated audio
        speaker_dir = os.path.join(args.output_dir, "speakers")
        separator.save_speaker_audio(speaker_audio, speaker_dir)
        
        # Get reference audio files
        reference_files = []
        for speaker_id in speaker_audio.keys():
            ref_file = os.path.join(speaker_dir, f"{speaker_id}.wav")
            reference_files.append(ref_file)
        
        # Save voice models if requested or if they don't exist
        if args.save_models or not os.path.exists(args.models_dir):
            print(f"Saving voice models to {args.models_dir}...")
            cloner.save_voice_models(reference_files, args.models_dir)
        
        # Generate synthetic speech
        print("Cloning voices...")
        synthesis_dir = os.path.join(args.output_dir, "synthesis")
        output_files = cloner.batch_clone_voices(
            args.text, 
            reference_files, 
            synthesis_dir
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
