"""
Data augmentation module
- Text shuffle
- Prepare reference audio
- TTS generation
- Re-ASR
"""
from pathlib import Path

def extract_subject_id(audio_path: str) -> str:
    """
    Extract the subject ID from an audio file path (up to the first "-").
    
    Examples:
        "train/ad/001-0.mp3" -> "001"
        "train/ad/001-1.mp3" -> "001"
        "001-0.mp3" -> "001"
        "001-0.wav" -> "001"
        "001.mp3" -> "001" (if there is no "-", use the whole filename)
    
    Args:
        audio_path: Path to the audio file (full path or filename)
    
    Returns:
        Subject ID (the part before "-", with the extension removed)
    """
    # Remove the extension from the filename
    name_without_ext = Path(audio_path).stem
    # Split by "-" and take the first part as the subject ID
    if '-' in name_without_ext:
        return name_without_ext.split('-')[0]
    else:
        # If "-" is not present, use the whole filename as the subject ID
        return name_without_ext

