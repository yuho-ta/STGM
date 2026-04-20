"""
Generate TTS using Fish Speech.
Create new audio from reference audio and shuffled text.
"""
import os
import sys
import random
import re
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
import csv
import io
import subprocess
import json

def safe_print(*args, **kwargs):
    """
    Safe print function for Windows console Unicode output.
    If UnicodeEncodeError happens, ignore the error and continue.
    """
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # If strings contain Unicode characters, encode safely and print
        try:
            # Replace characters that cannot be encoded
            safe_args = []
            for arg in args:
                if isinstance(arg, str):
                    # Replace unencodable characters with '?'
                    safe_args.append(arg.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8', errors='replace'))
                else:
                    safe_args.append(arg)
            print(*safe_args, **kwargs)
        except Exception:
            # If it still fails, try printing a string representation
            try:
                safe_str = ' '.join(str(arg).encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8', errors='replace') for arg in args)
                print(safe_str, **kwargs)
            except Exception:
                # Last resort: ignore the error and continue
                pass

try:
    import soundfile as sf
    import numpy as np
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    safe_print("Warning: soundfile and numpy are required for audio concatenation")
    safe_print("  Install them with: pip install soundfile numpy")

def get_python_executable():
    """Get venv Python if available; otherwise use system Python."""
    venv_python = Path("venv") / "Scripts" / "python.exe" if os.name == 'nt' else Path("venv") / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable

# Import extract_subject_id (relative or absolute import)
try:
    from . import extract_subject_id
except ImportError:
    # Fallback for direct execution
    # Add parent directory to sys.path to import the augmentation module
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from augmentation import extract_subject_id

def load_shuffled_texts(csv_path: Path) -> List[Tuple[str, str, str]]:
    """
    Load shuffled texts.
    
    Returns:
        [(audio_path, original_text, shuffled_text), ...]
    """
    texts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append((
                row["audio_path"],
                row["original_text"],
                row["shuffled_text"]
            ))
    return texts


def get_reference_audios(ref_dir: Path) -> List[Path]:
    """Get reference audio file list (mp3 and wav)."""
    return list(ref_dir.glob("*.mp3")) + list(ref_dir.glob("*.wav"))


def load_reference_wav_bytes_for_api(reference_audio: Path) -> bytes:
    """
    Load reference audio as WAV bytes for the Fish Speech API.
    If the reference is too long, truncate from the beginning
    (exceeding RoPE fixed length may crash the GPU).

    Environment variables:
        FISH_MAX_REFERENCE_SECONDS: Max length in seconds (default: 180).
    If soundfile cannot decode the format, read the file bytes as-is (previous behavior).
    """
    max_sec = float(os.environ.get("FISH_MAX_REFERENCE_SECONDS", "180"))
    max_samples_env = int(os.environ.get("FISH_MAX_REFERENCE_SAMPLES", "0"))

    if HAS_AUDIO_LIBS:
        try:
            data, sr = sf.read(str(reference_audio), always_2d=True)
            n = data.shape[0]
            cap = n
            if max_sec > 0:
                cap = min(cap, int(max_sec * sr))
            if max_samples_env > 0:
                cap = min(cap, max_samples_env)
            if n > cap:
                safe_print(
                    f"Warning: Reference is {n / sr:.1f}s; truncating to {cap / sr:.1f}s "
                    f"({reference_audio.name})"
                )
                data = data[:cap, :]
            buf = io.BytesIO()
            sf.write(buf, data, sr, format="WAV")
            return buf.getvalue()
        except Exception as e:
            safe_print(
                f"Warning: Could not load/trim reference with soundfile ({e}); "
                f"sending raw file bytes ({reference_audio.name})"
            )

    with open(reference_audio, "rb") as f:
        return f.read()


# Same as text_shuffle.py: one period token = 0.5s silence
PERIOD_DURATION = 0.5


def split_text_by_pause_markers(
    text: str,
    min_silence_duration: float = 0.5
) -> List[Tuple[str, float]]:
    """
    Split a shuffled text by pause markers (". . .") and compute segment silence lengths.
    Since text_shuffle inserts periods every 0.5 seconds, timestamps are not required.

    Args:
        text: shuffled_text that contains pause periods (".", ". . .", ...)
        min_silence_duration: Split only when silence duration is at least this many seconds (default: 0.5s)

    Returns:
        [(segment_text, silence_after_seconds), ...]
        (silence_after for the last segment is 0.0)
    """
    if not text or not text.strip():
        return []
    # Pattern: treat repetitions of "space + dot" as separators
    # text_shuffle output (e.g., " . . . ") is treated as a sequence of space+dot tokens
    parts = re.split(r'((?: \.)+)', text)
    if len(parts) < 2:
        return [(text.strip(), 0.0)]
    # parts = [seg1, sep1, seg2, sep2, ...]; odd indices are separator strings
    segments_with_silence = []
    i = 0
    while i < len(parts):
        seg = parts[i].strip() if i < len(parts) else ""
        if not seg:
            i += 1
            continue
        silence_after = 0.0
        if i + 1 < len(parts):
            sep = parts[i + 1]
            n_dots = sep.count('.')
            silence_after = n_dots * PERIOD_DURATION
        segments_with_silence.append((seg, silence_after))
        i += 2
    if not segments_with_silence:
        return [(text.strip(), 0.0)]
    # Merge separators with silence shorter than min_silence (split only for long pauses)
    merged = []
    cur_text = segments_with_silence[0][0]
    cur_silence_after = segments_with_silence[0][1]
    for i in range(1, len(segments_with_silence)):
        next_text, next_silence = segments_with_silence[i]
        if cur_silence_after >= min_silence_duration:
            merged.append((cur_text.strip(), cur_silence_after))
            cur_text = next_text
            cur_silence_after = next_silence
        else:
            cur_text = cur_text + " " + next_text
            cur_silence_after = next_silence
    merged.append((cur_text.strip(), cur_silence_after))
    return merged


def concatenate_audio_files_with_pauses(audio_files: List[Path], output_path: Path, pause_durations: List[float]) -> bool:
    """
    Concatenate multiple audio files, inserting different silence durations between them.
    
    Args:
        audio_files: List of audio files to concatenate
        output_path: Output file path
        pause_durations: Silence durations to insert after each file (last file excluded)
    
    Returns:
        Whether concatenation succeeded
    """
    if not HAS_AUDIO_LIBS:
        safe_print("Error: soundfile and numpy are required for audio concatenation")
        return False
    
    if not audio_files:
        return False
    
    if len(pause_durations) != len(audio_files) - 1:
        safe_print(f"Warning: pause_durations length ({len(pause_durations)}) != audio_files length - 1 ({len(audio_files) - 1})")
        # If pause_durations are missing, pad with 0.0
        pause_durations = pause_durations + [0.0] * (len(audio_files) - 1 - len(pause_durations))
    
    try:
        audio_data_list = []
        sample_rate = None
        
        for i, audio_file in enumerate(audio_files):
            if not audio_file.exists():
                safe_print(f"Warning: Audio file not found: {audio_file}")
                continue
            
            data, sr = sf.read(str(audio_file))
            if sample_rate is None:
                sample_rate = sr
            elif sample_rate != sr:
                # If the sample rate differs, resample (simple implementation)
                safe_print(f"Warning: Sample rate mismatch ({sr} vs {sample_rate}), skipping resampling")
            
            audio_data_list.append(data)
            
            # Add silence after each file except the last (only when pause_duration > 0)
            if i < len(audio_files) - 1 and pause_durations[i] > 0:
                silence_samples = int(pause_durations[i] * sample_rate)
                silence = np.zeros(silence_samples, dtype=data.dtype)
                audio_data_list.append(silence)
        
        if not audio_data_list:
            return False
        
        # Concatenate
        concatenated = np.concatenate(audio_data_list)
        
        # Save
        sf.write(str(output_path), concatenated, sample_rate)
        return True
    
    except Exception as e:
        safe_print(f"Error concatenating audio files: {e}")
        return False

def concatenate_audio_files(audio_files: List[Path], output_path: Path, silence_duration: float = 0.0) -> bool:
    """
    Concatenate multiple audio files.
    
    Args:
        audio_files: List of audio files to concatenate
        output_path: Output file path
        silence_duration: Silence duration to insert between files (seconds)
    
    Returns:
        Whether concatenation succeeded
    """
    if not HAS_AUDIO_LIBS:
        safe_print("Error: soundfile and numpy are required for audio concatenation")
        return False
    
    if not audio_files:
        return False
    
    try:
        audio_data_list = []
        sample_rate = None
        
        for audio_file in audio_files:
            if not audio_file.exists():
                safe_print(f"Warning: Audio file not found: {audio_file}")
                continue
            
            data, sr = sf.read(str(audio_file))
            if sample_rate is None:
                sample_rate = sr
            elif sample_rate != sr:
                # If sample rate differs, resample (simple implementation)
                safe_print(f"Warning: Sample rate mismatch ({sr} vs {sample_rate}), skipping resampling")
            
            audio_data_list.append(data)
            
            # Add silence between files (not after the last file)
            if silence_duration > 0 and audio_file != audio_files[-1]:
                silence_samples = int(silence_duration * sample_rate)
                silence = np.zeros(silence_samples, dtype=data.dtype)
                audio_data_list.append(silence)
        
        if not audio_data_list:
            return False
        
        # Concatenate
        concatenated = np.concatenate(audio_data_list)
        
        # Save
        sf.write(str(output_path), concatenated, sample_rate)
        return True
    
    except Exception as e:
        safe_print(f"Error concatenating audio files: {e}")
        return False

def generate_tts_with_fish_speech(
    reference_audio: Path,
    text: str,
    output_path: Path,
    fish_speech_path: str = None,
    reference_text: str = None,
    timeout: int = None,
    segments_with_silence: Optional[List[Tuple[str, float]]] = None,
) -> bool:
    """
    Generate TTS with Fish Speech.
    
    Args:
        reference_audio: Reference audio path (trim to a max length before sending; uses env var FISH_MAX_REFERENCE_SECONDS, etc.)
        text: Text to generate (shuffled_text)
        output_path: Output audio file path
        fish_speech_path: Fish Speech path (None = read from environment)
        reference_text: Text corresponding to the reference_audio (original_text). If None, use an empty string.
        timeout: API request timeout (seconds). If None, use FISH_SPEECH_TIMEOUT or default 600s.
        segments_with_silence: When provided, split by [(segment_text, silence_after), ...],
            generate per segment, insert silence, and concatenate (typically produced by split_text_by_pause_markers).
    
    Returns:
        Whether generation succeeded
    
    Note:
        Adjust the implementation to match the actual Fish Speech API you are using.
        Examples:
        - When using the Fish Speech Python API:
          from fish_speech import TTS
          tts = TTS()
          tts.generate(reference_audio, text, output_path)
        
        - When using the command line:
          fish-speech infer --reference <ref> --text <text> --output <out>
        
        - When using the REST API:
          requests.post(url, json={"reference": ref, "text": text})
    """
    # If splitting by pause markers (". . .") into segments + silence lengths
    if segments_with_silence and len(segments_with_silence) > 1:
        temp_dir = Path(tempfile.mkdtemp())
        generated_files = []
        pause_durations = [silence_after for _, silence_after in segments_with_silence[:-1]]

        try:
            for i, (segment_text, _) in enumerate(segments_with_silence):
                temp_output = temp_dir / f"seg_{i:03d}.wav"
                success = generate_tts_with_fish_speech(
                    reference_audio=reference_audio,
                    text=segment_text,
                    output_path=temp_output,
                    fish_speech_path=fish_speech_path,
                    reference_text=reference_text,
                    timeout=timeout,
                    segments_with_silence=None,
                )
                if success and temp_output.exists():
                    generated_files.append(temp_output)
                else:
                    safe_print(f"Warning: Failed to generate segment {i+1}/{len(segments_with_silence)}: {segment_text[:50]}...")
            if generated_files:
                success = concatenate_audio_files_with_pauses(
                    generated_files, output_path, pause_durations
                )
                return success
            return False
        finally:
            try:
                for temp_file in temp_dir.glob("*"):
                    temp_file.unlink()
                temp_dir.rmdir()
            except Exception as e:
                safe_print(f"Warning: Failed to clean up temp files: {e}")
        return False

    # Normal generation (send the full text at once)
    # Get Fish Speech path
    if fish_speech_path is None:
        fish_speech_path = os.environ.get("FISH_SPEECH_PATH", "fish-speech")
    
    # Use the Fish Speech API server
    # Default: localhost:8080 (server default port)
    # On Windows, 127.0.0.1 is recommended (more stable than 0.0.0.0)
    fish_speech_api_url = os.environ.get("FISH_SPEECH_API_URL", "http://127.0.0.1:8080")
    
    try:
        import requests  # type: ignore
        import base64
        import time
        
        # Reference audio (trim if too long; to avoid max sequence length issues in API/codecs)
        audio_data = load_reference_wav_bytes_for_api(reference_audio)
        
        # Send API request
        # Fish Speech API endpoint: /v1/tts (defined in api_server.py)
        api_url = f"{fish_speech_api_url.rstrip('/')}/v1/tts"
        
        # Retry settings (for network instability on Windows)
        max_retries = 5  # Increase retry count
        base_retry_delay = 2  # Base delay (seconds)
        # Timeout setting: argument > env var > default (10 minutes)
        max_timeout = timeout if timeout is not None else int(os.environ.get("FISH_SPEECH_TIMEOUT", "600"))
        
        def is_windows_connection_error(e: Exception) -> bool:
            """Detect Windows-specific connection errors (e.g., WinError 64)."""
            if isinstance(e, OSError):
                # WinError 64: "The network name cannot be found/used"
                # Error 10054: "Connection reset by peer"
                if hasattr(e, 'winerror'):
                    return e.winerror in (64, 10054)
                # Check errno as well
                if hasattr(e, 'errno'):
                    return e.errno in (64, 10054)
            return False
        
        for attempt in range(max_retries):
            try:
                # Exponential backoff: gradually increase the retry delay.
                retry_delay = base_retry_delay * (2 ** attempt)
                
                response = requests.post(
                    api_url,
                    json={
                        "text": text,  # shuffled_text: text to generate
                        "references": [{
                            "audio": base64.b64encode(audio_data).decode("utf-8"),
                            "text": reference_text if reference_text else ""  # original_text: reference-audio text
                        }],
                        "format": "wav",
                        "normalize": True,  # Explicitly enable text normalization
                        "chunk_length": 200,
                        "max_new_tokens": 1024,
                        "top_p": 0.8,
                        "repetition_penalty": 1.1,
                        "temperature": 0.8
                    },
                    timeout=max_timeout  # Use a long timeout (default 10 minutes)
                )
                
                if response.status_code == 200:
                    # Save audio data
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    return True
                elif response.status_code == 500:
                    # Retry on server error
                    if attempt < max_retries - 1:
                        safe_print(f"Server error (status {response.status_code}), retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        safe_print(f"Error generating TTS: API returned status {response.status_code} after {max_retries} attempts")
                        safe_print(f"  URL: {api_url}")
                        safe_print(f"  Response: {response.text}")
                        return False
                else:
                    # Do not retry other error codes
                    safe_print(f"Error generating TTS: API returned status {response.status_code}")
                    safe_print(f"  URL: {api_url}")
                    safe_print(f"  Response: {response.text}")
                    if response.status_code == 404:
                        safe_print(f"  Note: Make sure Fish Speech API server is running on {fish_speech_api_url}")
                        safe_print(f"        Start it with: cd fish-speech && python -m tools.api_server --listen 127.0.0.1:8080 ...")
                    return False
                    
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
                # Retry on network errors.
                is_windows_error = is_windows_connection_error(e)
                error_type = "Windows connection error" if is_windows_error else "Network error"
                
                if attempt < max_retries - 1:
                    safe_print(f"{error_type}: {e}, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    if is_windows_error:
                        safe_print(f"  This is a Windows-specific network error (WinError 64/10054).")
                        safe_print(f"  It may be caused by firewall/antivirus interference or connection timeout.")
                    time.sleep(retry_delay)
                    continue
                else:
                    safe_print(f"Error: Failed to connect to Fish Speech API server after {max_retries} attempts")
                    safe_print(f"  URL: {api_url}")
                    safe_print(f"  Error: {e}")
                    if is_windows_error:
                        safe_print(f"  This is a Windows-specific network error. Try:")
                        safe_print(f"  1. Check firewall/antivirus settings")
                        safe_print(f"  2. Use 127.0.0.1 instead of 0.0.0.0 for the API server")
                        safe_print(f"  3. Increase timeout: set FISH_SPEECH_TIMEOUT=1200 (20 minutes)")
                    safe_print(f"  Make sure the API server is running:")
                    safe_print(f"  1. cd fish-speech")
                    safe_print(f"  2. python -m tools.api_server --listen 127.0.0.1:8080 --llama-checkpoint-path checkpoints/openaudio-s1-mini --decoder-checkpoint-path checkpoints/openaudio-s1-mini/codec.pth --decoder-config-name modded_dac_vq")
                    safe_print(f"  3. Or set FISH_SPEECH_API_URL to the correct URL")
                    return False
            except OSError as e:
                # Handle Windows-specific OSError (e.g., WinError 64).
                if is_windows_connection_error(e):
                    if attempt < max_retries - 1:
                        safe_print(f"Windows connection error (WinError {getattr(e, 'winerror', getattr(e, 'errno', 'unknown'))}): {e}")
                        safe_print(f"  Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        safe_print(f"  This error often occurs due to connection interruption during handshake.")
                        time.sleep(retry_delay)
                        continue
                    else:
                        safe_print(f"Error: Windows connection error after {max_retries} attempts")
                        safe_print(f"  WinError: {getattr(e, 'winerror', getattr(e, 'errno', 'unknown'))}")
                        safe_print(f"  Error: {e}")
                        safe_print(f"  Suggestions:")
                        safe_print(f"  1. Use 127.0.0.1 instead of 0.0.0.0: --listen 127.0.0.1:8080")
                        safe_print(f"  2. Check firewall/antivirus settings")
                        safe_print(f"  3. Try a different port: --listen 127.0.0.1:8081")
                        safe_print(f"  4. Increase timeout: set FISH_SPEECH_TIMEOUT=1200")
                        return False
                else:
                    # Re-raise other OSErrors.
                    raise
        
        return False
        
    except ImportError:
        safe_print("Error: 'requests' library is required for Fish Speech API usage")
        safe_print("  Install it with: pip install requests")
        return False
    except Exception as e:
        safe_print(f"Error generating TTS via API: {e}")
        return False

def augment_class_with_fish_speech(
    data_dir: str = "data",
    split: str = "train",
    class_name: str = "ad",
    num_augmentations: int = 3,
    fish_speech_path: str = None,
    random_seed: int = 42,
    add_silence_on_concat: bool = False,
    timeout: int = None,
    min_silence_to_split: float = 0.5,
):
    """
    Augment audio within a class using Fish Speech.

    Args:
        data_dir: Data directory
        split: train or val
        class_name: ad or cn
        num_augmentations: Number of augmentations to generate per audio file
        fish_speech_path: Path to Fish Speech
        random_seed: Random seed
        min_silence_to_split: Split at pauses longer than this many seconds
            (when using text pause markers; default 0.5)
    """
    random.seed(random_seed)
    
    data_path = Path(data_dir)
    
    # Load shuffled text pairs
    # Use the CSV located under a subdirectory split by num_augmentations
    shuffled_csv = data_path / split / f"aug{num_augmentations}" / "shuffle" / f"{class_name}_shuffled_texts.csv"
    if not shuffled_csv.exists():
        safe_print(f"Error: {shuffled_csv} does not exist. Run text_shuffle.py first.")
        return
    
    shuffled_texts = load_shuffled_texts(shuffled_csv)
    safe_print(f"Loaded {len(shuffled_texts)} shuffled text pairs")
    
    # Create output directory (subdirectory split by num_augmentations)
    output_dir = data_path / split / f"aug{num_augmentations}" / f"{class_name}_augmented"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = []
    
    # Track augmentation index for each original audio file
    audio_aug_count = {}  # {audio_path: current_aug_idx}
    
    # Generate TTS for each shuffled text
    # The shuffled_texts.csv already includes num_augmentations shuffled texts
    # Generate TTS once per shuffled text
    for i, (audio_path, original_text, shuffled_text) in enumerate(shuffled_texts):
        # Get the original audio path
        # audio_path is relative to data_dir, so join it
        audio_path_obj = data_path / audio_path
        
        # Always use the original audio file as the reference audio
        ref_audio = audio_path_obj
        
        # Check that the reference audio exists
        if not ref_audio.exists():
            safe_print(f"Warning: {ref_audio} does not exist, skipping...")
            continue
        
        # Manage augmentation index per audio file
        if audio_path not in audio_aug_count:
            audio_aug_count[audio_path] = 0
        else:
            audio_aug_count[audio_path] += 1
        
        aug_idx = audio_aug_count[audio_path]
        output_filename = f"{Path(audio_path).stem}_aug_{aug_idx}.wav"
        output_path = output_dir / output_filename

        if output_path.exists():
            safe_print(f"Warning: {output_path} already exists, skipping...")
            metadata.append({
                "original_audio": str(audio_path),
                "augmented_audio": str(output_path.relative_to(data_path)),
                "reference_audio": str(ref_audio.relative_to(data_path)),
                "original_text": original_text,
                "shuffled_text": shuffled_text
            })
            continue
        
        safe_print(f"\n[{i+1}/{len(shuffled_texts)}] Generating augmentation {aug_idx+1}/{num_augmentations} for {Path(audio_path).name}")
        safe_print(f"  Reference: {ref_audio.name}")
        safe_print(f"  Text: {shuffled_text[:50]}...")
        
        # If add_silence_on_concat is enabled, split by text pause markers
        # (". . .", where each dot represents 0.5s) and insert silence.
        segments_with_silence = None
        if add_silence_on_concat:
            segments_with_silence = split_text_by_pause_markers(
                shuffled_text, min_silence_duration=min_silence_to_split
            )
            if segments_with_silence and len(segments_with_silence) <= 1:
                segments_with_silence = None
        
        success = generate_tts_with_fish_speech(
            reference_audio=ref_audio,
            text=shuffled_text,
            reference_text=original_text,
            output_path=output_path,
            fish_speech_path=fish_speech_path,
            timeout=timeout,
            segments_with_silence=segments_with_silence,
        )
        
        if success:
            metadata.append({
                "original_audio": str(audio_path),
                "augmented_audio": str(output_path.relative_to(data_path)),
                "reference_audio": str(ref_audio.relative_to(data_path)),
                "original_text": original_text,
                "shuffled_text": shuffled_text
            })
    
    # Save metadata (including num_augmentations info)
    metadata_file = output_dir / "metadata.json"
    # For backward compatibility, save in dict format (some existing code may expect list format).
    metadata_with_info = {
        "num_augmentations": num_augmentations,
        "entries": metadata
    }
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_with_info, f, ensure_ascii=False, indent=2)
    
    safe_print(f"\nAugmentation completed!")
    safe_print(f"Generated {len(metadata)} augmented audios")
    safe_print(f"Metadata saved to {metadata_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate TTS with Fish Speech")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Split to process")
    parser.add_argument("--class_name", type=str, default="ad", choices=["ad", "cn"], help="Class to process")
    parser.add_argument("--num_augmentations", type=int, default=3, help="Number of augmentations per audio")
    parser.add_argument("--fish_speech_path", type=str, default=None, help="Path to Fish Speech")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--add-silence-on-concat", action="store_true", default=True, help="Split by text pause markers ( . . . ) and add silence between segments (0.5s per dot)")
    parser.add_argument("--timeout", type=int, default=None, help="API request timeout in seconds (default: 600 seconds / 10 minutes). Increase this if you encounter Windows connection errors.")
    parser.add_argument("--min-silence-to-split", type=float, default=1.00, help="Minimum silence (seconds) to split at when using text pause markers (default: 0.5)")
    
    args = parser.parse_args()
    
    augment_class_with_fish_speech(
        data_dir=args.data_dir,
        split=args.split,
        class_name=args.class_name,
        num_augmentations=args.num_augmentations,
        fish_speech_path=args.fish_speech_path,
        random_seed=args.seed,
        add_silence_on_concat=args.add_silence_on_concat,
        timeout=args.timeout,
        min_silence_to_split=args.min_silence_to_split,
    )

