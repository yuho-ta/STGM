"""
Script to prepare reference audio
Condition: select audio with more disfluency content, within 30s–2min
"""
import os
from pathlib import Path
import librosa
import numpy as np
from typing import List, Tuple

def get_audio_duration(audio_path: Path) -> float:
    """Get the duration of an audio file (seconds)"""
    try:
        y, sr = librosa.load(str(audio_path), sr=None)
        duration = len(y) / sr
        return duration
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return 0.0

def select_reference_audios(
    audio_dir: Path,
    min_duration: float = 30.0,
    max_duration: float = 120.0,
    num_references: int = 5
) -> List[Path]:
    """
    Select reference audios
    
    Args:
        audio_dir: Directory containing audio files
        min_duration: Minimum duration (seconds)
        max_duration: Maximum duration (seconds)
        num_references: Number of reference audios to select
    
    Returns:
        List of paths to the selected reference audios
    """
    # If using a preprocessed directory, use only *_subject.wav
    # Otherwise, use all wav/mp3 files
    if "_preprocessed" in str(audio_dir):
        audio_files = list(audio_dir.glob("*_subject.wav"))
    else:
        audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))
    
    # Filter by duration
    valid_audios = []
    for audio_file in audio_files:
        duration = get_audio_duration(audio_file)
        if min_duration <= duration <= max_duration:
            valid_audios.append((audio_file, duration))
    
    # Sort by length (descending)
    valid_audios.sort(key=lambda x: x[1], reverse=True)
    
    # Select the top N audios (longer audio is more likely to contain disfluency)
    selected = [audio for audio, _ in valid_audios[:num_references]]
    
    print(f"Found {len(valid_audios)} valid audios (30s-2min)")
    print(f"Selected {len(selected)} reference audios:")
    for audio in selected:
        duration = get_audio_duration(audio)
        print(f"  {audio.name}: {duration:.2f}s")
    
    return selected

def prepare_references_for_all_classes(
    data_dir: str = "data",
    split: str = "train",
    min_duration: float = 30.0,
    max_duration: float = 120.0,
    num_references: int = 5,
    use_preprocessed: bool = True
):
    """
    Prepare reference audios for all classes
    
    Args:
        data_dir: Data directory
        split: train or val
        min_duration: Minimum duration (seconds)
        max_duration: Maximum duration (seconds)
        num_references: Number of reference audios per class
        use_preprocessed: Whether to use preprocessed audio
    """
    data_path = Path(data_dir)
    ref_output_dir = data_path / split / "references"
    ref_output_dir.mkdir(parents=True, exist_ok=True)
    
    for class_name in ["ad", "cn"]:
        # Use preprocessed audio if available
        if use_preprocessed:
            preprocessed_dir = data_path / split / f"{class_name}_preprocessed"
            if preprocessed_dir.exists() and list(preprocessed_dir.glob("*.wav")):
                class_dir = preprocessed_dir
            else:
                class_dir = data_path / split / class_name
        else:
            class_dir = data_path / split / class_name
        
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist, skipping...")
            continue
        
        print(f"\n=== Processing {class_name.upper()} ===")
        references = select_reference_audios(
            class_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            num_references=num_references
        )
        
        # Copy reference audios to a dedicated folder
        class_ref_dir = ref_output_dir / class_name
        class_ref_dir.mkdir(parents=True, exist_ok=True)
        
        for ref_audio in references:
            import shutil
            shutil.copy2(ref_audio, class_ref_dir / ref_audio.name)
        
        print(f"Saved {len(references)} reference audios to {class_ref_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare reference audios for TTS")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Split to process")
    parser.add_argument("--min_duration", type=float, default=30.0, help="Minimum duration (seconds)")
    parser.add_argument("--max_duration", type=float, default=120.0, help="Maximum duration (seconds)")
    parser.add_argument("--num_references", type=int, default=5, help="Number of reference audios per class")
    
    args = parser.parse_args()
    
    prepare_references_for_all_classes(
        data_dir=args.data_dir,
        split=args.split,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        num_references=args.num_references
    )

