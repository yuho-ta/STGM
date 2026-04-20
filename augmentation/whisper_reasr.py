"""
Script to re-ASR TTS-generated audio with Whisper
Get "realistic text" corresponding to the TTS audio
"""
import os
from pathlib import Path
from typing import List, Dict
import whisper
import json
import librosa

def load_whisper_model(model_size: str = "base"):
    """Load the Whisper model"""
    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)
    return model

def reasr_augmented_audios(
    metadata_file: Path,
    whisper_model,
    output_file: Path = None,
    language: str = "en"
):
    """
    Re-ASR augmented audios with Whisper
    
    Args:
        metadata_file: Augmentation metadata JSON file
        whisper_model: Whisper model
        output_file: Output file path
        language: Audio language ("en" for English, "ja" for Japanese)
    """
    # Load metadata
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata_raw = json.load(f)
    
    # Backward compatibility:
    # - If it's a dict, use entries
    # - If it's a list, use it as-is
    if isinstance(metadata_raw, dict):
        metadata = metadata_raw.get("entries", [])
    else:
        metadata = metadata_raw
    
    print(f"Processing {len(metadata)} augmented audios...")
    
    # Get the data directory (to resolve relative paths)
    # data/train/aug{num}/{class}_augmented/metadata.json -> search upward for the data directory
    data_dir = metadata_file.parent
    while data_dir.name != "data" and data_dir.parent != data_dir:
        data_dir = data_dir.parent
    
    # Re-ASR each augmented audio
    results = []
    for i, item in enumerate(metadata):
        augmented_audio_path = data_dir / item["augmented_audio"]
        
        if not augmented_audio_path.exists():
            print(f"Warning: {augmented_audio_path} does not exist, skipping...")
            continue
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(metadata)}")
        
        # Re-ASR with Whisper
        try:
            # Preload with librosa to avoid FFmpeg issues
            audio_array, sr = librosa.load(str(augmented_audio_path), sr=16000, mono=True)
            result = whisper_model.transcribe(audio_array, language=language)
            reasr_text = result["text"].strip()
            
            # Save the result
            item["reasr_text"] = reasr_text
            results.append(item)
            
        except Exception as e:
            print(f"Error transcribing {augmented_audio_path}: {e}")
            continue
    
    # Save results
    if output_file is None:
        output_file = metadata_file.parent / "metadata_with_reasr.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nRe-ASR completed!")
    print(f"Processed {len(results)}/{len(metadata)} audios")
    print(f"Results saved to {output_file}")
    
    # Display sample results
    print("\nSample results:")
    for i, item in enumerate(results[:3]):
        print(f"\n{i+1}. {Path(item['augmented_audio']).name}")
        print(f"   Shuffled text:  {item['shuffled_text'][:50]}...")
        print(f"   Re-ASR text:    {item['reasr_text'][:50]}...")


def process_split_reasr(
    data_dir: str = "data",
    split: str = "train",
    class_name: str = "ad",
    whisper_model_size: str = "base",
    language: str = "en",
    num_augmentations: int = 1
):
    """
    Re-ASR augmented audios for the specified split/class
    
    Args:
        data_dir: Data directory
        split: train or val
        class_name: ad or cn
        whisper_model_size: Whisper model size
        language: Audio language ("en" for English, "ja" for Japanese)
    """
    data_path = Path(data_dir)
    metadata_file = data_path / split / f"aug{num_augmentations}" / f"{class_name}_augmented" / "metadata.json"
    
    if not metadata_file.exists():
        print(f"Error: {metadata_file} does not exist. Run fish_tts_generate.py first.")
        return
    
    # Load the Whisper model
    model = load_whisper_model(whisper_model_size)
    
    # Run re-ASR (function updated to pass the language parameter)
    reasr_augmented_audios(metadata_file, model, language=language)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Re-ASR augmented audios with Whisper")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Split to process")
    parser.add_argument("--class_name", type=str, default="ad", choices=["ad", "cn"], help="Class to process")
    parser.add_argument("--whisper_model", type=str, default="base", help="Whisper model size")
    parser.add_argument("--language", type=str, default="en", help="Audio language (en/ja)")
    parser.add_argument("--num_augmentations", type=int, default=1, help="Number of augmentations per audio (for selecting aug folder)")
    
    args = parser.parse_args()
    
    process_split_reasr(
        data_dir=args.data_dir,
        split=args.split,
        class_name=args.class_name,
        whisper_model_size=args.whisper_model,
        language=args.language,
        num_augmentations=args.num_augmentations
    )

