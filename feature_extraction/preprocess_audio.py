"""
Audio preprocessing script.
- Speaker diarization + Whisper transcription with timestamps
- Optionally extract the main speaker (longest utterance)
- Volume normalization
- Noise removal
"""
import os
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import csv
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr

# Suppress unnecessary warnings (does not affect processing)
warnings.filterwarnings("ignore", message="Module 'speechbrain.pretrained' was deprecated")
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0")
warnings.filterwarnings("ignore", message="Failed to launch Triton kernels")

# PyTorch 2.6 compatibility patch: set torch.load default to weights_only=False
# pyannote.audio model files require weights_only=False
try:
    import torch
    original_torch_load = torch.load
    
    def patched_torch_load(*args, **kwargs):
        # Force weights_only=False (use at your own risk)
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)
    
    torch.load = patched_torch_load
except ImportError:
    pass

# huggingface_hub compatibility patch: map use_auth_token -> token
try:
    from huggingface_hub import hf_hub_download as original_hf_hub_download
    
    def patched_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            token_value = kwargs.pop("use_auth_token")
            if token_value and "token" not in kwargs:
                kwargs["token"] = token_value
        return original_hf_hub_download(*args, **kwargs)
    
    import huggingface_hub
    huggingface_hub.hf_hub_download = patched_hf_hub_download
except ImportError:
    pass

from pyannote.audio import Pipeline
import whisper
import json

def normalize_audio_volume_in_memory(
    y: np.ndarray,
    sr: int,
    target_db: float = -20.0
) -> Tuple[np.ndarray, int]:
    """
    Normalize and adjust audio volume (in-memory).
    
    Args:
        y: Audio data (numpy array)
        sr: Sampling rate
        target_db: Target volume in dB (default: -20dB, typical audio level)
    
    Returns:
        (normalized audio data, sampling rate)
    """
    # Compute current RMS level (for logging)
    rms_before = librosa.feature.rms(y=y)[0]
    current_rms_db = librosa.power_to_db(np.mean(rms_before**2))
    
    # RMS-based normalization
    target_rms_db = target_db
    gain_db = target_rms_db - current_rms_db
    
    # Apply gain
    gain_linear = librosa.db_to_amplitude(gain_db)
    y_normalized = y * gain_linear
    
    # Prevent clipping by limiting the maximum amplitude
    max_amplitude = 0.95  # Limit to 95% to avoid clipping
    if np.max(np.abs(y_normalized)) > max_amplitude:
        y_normalized = y_normalized / np.max(np.abs(y_normalized)) * max_amplitude
    
    # Check adjusted level
    rms_after = librosa.feature.rms(y=y_normalized)[0]
    rms_db_after = librosa.power_to_db(np.mean(rms_after**2))
    print(f"    Volume adjusted: {current_rms_db:.2f}dB -> {rms_db_after:.2f}dB (target: {target_db}dB)")
    
    return y_normalized, sr

def remove_noise_in_memory(
    y: np.ndarray,
    sr: int,
    stationary: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Perform noise removal (in-memory processing).
    
    Args:
        y: Audio data (numpy array)
        sr: Sampling rate
        stationary: Whether the noise is stationary
    
    Returns:
        (noise-reduced audio data, sampling rate)
    """
    # Spectral gating (noisereduce library)
    reduced_noise = nr.reduce_noise(
        y=y,
        sr=sr,
        stationary=stationary,
        prop_decrease=0.8  # Noise reduction strength
    )
    
    return reduced_noise, sr

def load_speaker_diarization_pipeline(model_path: str = None, token: str = None):
    """
    Load the speaker diarization pipeline.
    
    Args:
        model_path: Custom model path (None = use default model)
        token: Hugging Face token (if None, try to read from environment variables)
    
    Returns:
        Loaded pipeline, or None on failure.
    """
    try:
        # Get token (try environment variables)
        if token is None:
            # Try to read from environment variables
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        # Set env var (huggingface_hub reads it automatically)
        original_token = os.environ.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
        
        try:
            if model_path:
                # Custom model path
                pipeline = Pipeline.from_pretrained(model_path)
            else:
                # Default model (pyannote/speaker-diarization-3.1)
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            
            # Move to GPU when available
            import torch
            if torch.cuda.is_available():
                pipeline = pipeline.to(torch.device("cuda"))
                print(f"Successfully loaded speaker diarization pipeline (GPU: {torch.cuda.get_device_name(0)})")
            else:
                print(f"Successfully loaded speaker diarization pipeline (CPU)")
            return pipeline
        finally:
            # Restore env var
            if original_token is None:
                os.environ.pop("HF_TOKEN", None)
            else:
                os.environ["HF_TOKEN"] = original_token
                
    except Exception as e:
        print(f"Warning: Could not load speaker diarization pipeline: {e}")
        print(f"  Note: Some models may require a Hugging Face token.")
        print(f"  Set HF_TOKEN or HUGGINGFACE_TOKEN environment variable, or use --hf_token argument.")
        return None

def load_whisper_model(model_size: str = "base"):
    """Load the Whisper model."""
    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)
    # Move to GPU when available
    import torch
    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"  Whisper model moved to GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  Whisper model using CPU")
    return model

def perform_speaker_diarization_with_whisper(
    audio_path: Path,
    output_dir: Path,
    diarization_pipeline,
    whisper_model,
    language: str = "en",
    min_speaker_duration: float = 1.0,
    audio_data: np.ndarray = None,
    sample_rate: int = None
) -> Dict:
    """
    Run speaker diarization and transcribe with Whisper (with timestamps).
    
    Returns:
        {
            "segments": [(start_time, end_time, speaker_id, text), ...],
            "speaker_durations": {speaker_id: total_duration, ...},
            "main_speaker": speaker_id
        }
    """
    if diarization_pipeline is None:
        print(f"Warning: Speaker diarization pipeline not available")
        return None
    
    try:
        # Preload audio to avoid FFmpeg issues
        import librosa
        if audio_data is not None and sample_rate is not None:
            # Use in-memory data
            waveform = audio_data
            sample_rate = sample_rate
        else:
            # Load from file
            print(f"  Loading audio file...")
            waveform, sample_rate = librosa.load(str(audio_path), sr=None, mono=False)
        
        # Convert to mono when needed
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)
        
        # Step 1: Speaker diarization
        print(f"  Performing speaker diarization...")
        # Avoid torchcodec issues by preloading audio and passing as a dict
        try:
            import torch
            # Expected format: {'waveform': torch.Tensor, 'sample_rate': int}
            audio_dict = {
                'waveform': torch.from_numpy(waveform).unsqueeze(0),  # (1, samples)
                'sample_rate': sample_rate
            }
            diarization = diarization_pipeline(audio_dict)
        except Exception as audio_error:
            # If dict-based loading fails, try using the file path
            print(f"  Warning: Failed to load audio as dict, trying file path: {audio_error}")
            diarization = diarization_pipeline(str(audio_path))
        
        # Extract segments
        diarization_segments = []
        # Handle both DiarizeOutput (new API) and Annotation (legacy API)
        if hasattr(diarization, 'speaker_diarization'):
            # New API: DiarizeOutput object
            annotation = diarization.speaker_diarization
        else:
            # Legacy API: direct Annotation object
            annotation = diarization
        
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            duration = turn.end - turn.start
            if duration >= min_speaker_duration:
                diarization_segments.append((turn.start, turn.end, speaker))
        
        if not diarization_segments:
            print(f"  No speaker segments found")
            return None
        
        # Step 2: Transcribe with Whisper (timestamps)
        # Avoid FFmpeg issues by passing the numpy array directly
        print(f"  Transcribing with Whisper...")
        # Whisper expects 16kHz; resample if needed
        whisper_sample_rate = 16000
        if sample_rate != whisper_sample_rate:
            waveform_for_whisper = librosa.resample(waveform, orig_sr=sample_rate, target_sr=whisper_sample_rate)
        else:
            waveform_for_whisper = waveform
        
        whisper_result = whisper_model.transcribe(
            waveform_for_whisper,
            language=language,
            word_timestamps=True
        )
        
        # Step 3: Match diarization segments with Whisper segments
        segments_with_text = []
        whisper_segments = whisper_result.get("segments", [])
        
        for diar_start, diar_end, speaker_id in diarization_segments:
            # Find Whisper segments inside this speaker segment
            segment_texts = []
            for wseg in whisper_segments:
                wseg_start = wseg.get("start", 0)
                wseg_end = wseg.get("end", 0)
                wseg_text = wseg.get("text", "").strip()
                
                # Check overlap in time ranges
                if not (wseg_end < diar_start or wseg_start > diar_end):
                    # Add text for overlapping parts
                    segment_texts.append({
                        "start": max(wseg_start, diar_start),
                        "end": min(wseg_end, diar_end),
                        "text": wseg_text
                    })
            
            # Combine text
            combined_text = " ".join([s["text"] for s in segment_texts])
            segments_with_text.append((diar_start, diar_end, speaker_id, combined_text))
        
        # Step 4: Compute total speaking time per speaker
        speaker_durations = {}
        for start, end, speaker_id, _ in segments_with_text:
            duration = end - start
            speaker_durations[speaker_id] = speaker_durations.get(speaker_id, 0) + duration
        
        # Step 5: Identify the main speaker (longest duration)
        main_speaker = max(speaker_durations.items(), key=lambda x: x[1])[0]
        
        # Save results
        result = {
            "segments": segments_with_text,
            "speaker_durations": speaker_durations,
            "main_speaker": main_speaker,
            "whisper_segments": whisper_segments
        }
        
        # Save JSON
        result_file = output_dir / f"{audio_path.stem}_diarization.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Save a readable text format
        segments_file = output_dir / f"{audio_path.stem}_segments.txt"
        with open(segments_file, "w", encoding="utf-8") as f:
            f.write("start_time\tend_time\tspeaker_id\ttext\n")
            for start, end, speaker_id, text in segments_with_text:
                f.write(f"{start:.3f}\t{end:.3f}\t{speaker_id}\t{text}\n")
        
        print(f"  Found {len(segments_with_text)} segments")
        print(f"  Main speaker (subject): {main_speaker} ({speaker_durations[main_speaker]:.2f}s)")
        
        return result
        
    except Exception as e:
        print(f"Error in speaker diarization with Whisper: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_main_speaker_audio_from_memory(
    y: np.ndarray,
    sr: int,
    diarization_result: Dict,
    output_path: Path
) -> bool:
    """
    Extract and combine only the main speaker's audio (longest speaker), using in-memory audio.
    
    Args:
        y: Audio data (numpy array)
        sr: Sampling rate
        diarization_result: Speaker diarization result
        output_path: Output audio file path
    
    Returns:
        Whether extraction succeeded
    """
    try:
        if diarization_result is None:
            return False
        
        main_speaker = diarization_result["main_speaker"]
        segments = diarization_result["segments"]
        
        # Extract main speaker segments only
        main_speaker_segments = [
            (start, end, text) 
            for start, end, speaker_id, text in segments 
            if speaker_id == main_speaker
        ]
        
        if not main_speaker_segments:
            print(f"  No segments found for main speaker {main_speaker}")
            return False
        
        # Combine segments
        combined_audio = []
        for start, end, text in main_speaker_segments:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment = y[start_sample:end_sample]
            combined_audio.append(segment)
        
        # Combine audio, inserting short silence between segments
        silence_duration = 0.1  # 0.1s silence
        silence_samples = int(silence_duration * sr)
        silence = np.zeros(silence_samples)
        
        final_audio = []
        for i, segment in enumerate(combined_audio):
            final_audio.append(segment)
            if i < len(combined_audio) - 1:  # Not the last segment
                final_audio.append(silence)
        
        combined_y = np.concatenate(final_audio)
        
        # Save (only final output)
        sf.write(str(output_path), combined_y, sr)
        
        print(f"  Extracted main speaker audio: {len(main_speaker_segments)} segments, {len(combined_y)/sr:.2f}s")
        return True
        
    except Exception as e:
        print(f"Error extracting main speaker audio: {e}")
        import traceback
        traceback.print_exc()
        return False

def preprocess_audio_file(
    audio_path: Path,
    output_dir: Path,
    diarization_pipeline=None,
    whisper_model=None,
    perform_volume_normalization: bool = True,
    perform_noise_removal: bool = True,
    perform_diarization: bool = True,
    extract_main_speaker: bool = True,
    volume_target_db: float = -20.0,
    language: str = "en",
    split: str = "train",
    class_name: str = "ad"
) -> Optional[Path]:
    """
    Preprocess a single audio file.
    
    Processing steps:
    1. Volume normalization (in-memory, RMS)
    2. Noise removal (in-memory, spectral gating)
    3. Crop and concatenate only the "PAR" speaking intervals from the segmentation CSV
    4. Run speaker diarization + Whisper on the preprocessed audio (main speaker estimation)
    5. Extract only the main speaker audio from the diarization results
    6. Save the processed audio
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load audio (full audio)
    print(f"  Loading audio file...")
    y, sr = librosa.load(str(audio_path), sr=None)
    
    # Step 1: Volume normalization (in-memory)
    if perform_volume_normalization:
        print(f"  Normalizing volume...")
        y, sr = normalize_audio_volume_in_memory(y, sr, target_db=volume_target_db)
    
    # Step 2: Noise removal (in-memory)
    if perform_noise_removal:
        print(f"  Removing noise...")
        y, sr = remove_noise_in_memory(y, sr)
    
    # Step 3: Extract only "PAR" intervals from the segmentation CSV (train/val/test)
    # Estimate project root (one level above feature_extraction directory)
    project_root = Path(__file__).resolve().parent.parent
    par_segments: List[Tuple[float, float]] = []
    seg_csv_path: Optional[Path] = None
    
    try:
        if split in ("train", "val"):
            # Example: segmentation/ad/adrso024.csv
            seg_csv_path = project_root / "segmentation" / class_name / f"{audio_path.stem}.csv"
        elif split == "test":
            # Example: test-dist/segmentation/adrsdt1.csv
            seg_csv_path = project_root / "test-dist" / "segmentation" / f"{audio_path.stem}.csv"
        
        if seg_csv_path is not None and seg_csv_path.exists():
            print(f"  Using segmentation file: {seg_csv_path}")
            with open(seg_csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("speaker") == "PAR":
                        try:
                            # ADReSS/ADReSSo format: begin/end are in milliseconds
                            begin_ms = float(row.get("begin", "0") or 0)
                            end_ms = float(row.get("end", "0") or 0)
                            if end_ms > begin_ms:
                                par_segments.append((begin_ms / 1000.0, end_ms / 1000.0))
                        except ValueError:
                            continue
    except Exception as e:
        print(f"  Warning: Failed to read segmentation CSV for {audio_path.name}: {e}")
    
    # Create an audio track by concatenating only the PAR segments (when present)
    if par_segments:
        print(f"  Found {len(par_segments)} PAR segments in segmentation. Cropping audio...")
        segments_audio = []
        # Insert a short silence between segments to make the result more natural
        silence_duration = 0.1  # 0.1s silence
        silence_samples = int(silence_duration * sr)
        silence = np.zeros(silence_samples, dtype=y.dtype)
        
        for idx, (start_sec, end_sec) in enumerate(par_segments):
            start_sample = max(0, int(start_sec * sr))
            end_sample = min(len(y), int(end_sec * sr))
            if end_sample > start_sample:
                segments_audio.append(y[start_sample:end_sample])
                if idx < len(par_segments) - 1:
                    segments_audio.append(silence)
        
        if segments_audio:
            y = np.concatenate(segments_audio)
            print(f"  Cropped audio to PAR segments: {len(y)/sr:.2f}s")
        else:
            print(f"  Warning: No valid PAR segments after time conversion. Using full audio.")
    else:
        # If there is no segmentation file or no PAR intervals, use full audio
        print(f"  No segmentation file found or no PAR segments. Using full audio.")
    
    # Step 4: Speaker diarization + Whisper (on the preprocessed audio)
    diarization_result = None
    if perform_diarization and diarization_pipeline is not None and whisper_model is not None:
        # Create a temporary file for diarization (not deleted to avoid Windows file-lock issues)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            sf.write(str(tmp_path), y, sr)
        
        # Close the file handle before running diarization
        diarization_result = perform_speaker_diarization_with_whisper(
            tmp_path,
            output_dir,
            diarization_pipeline,
            whisper_model,
            language=language,
            audio_data=y,
            sample_rate=sr
        )
        # Keep the temporary file (avoid Windows errors when the file is still in use)
    
    # Step 5: Extract main speaker audio (always run when diarization results exist)
    if diarization_result is not None:
        # If extract_main_speaker is False, skip main speaker extraction
        if not extract_main_speaker:
            print("  Skipping main speaker extraction (extract_main_speaker=False)")
        else:
            print("  Removing other speakers' speech segments...")
            main_speaker = diarization_result["main_speaker"]
            segments = diarization_result["segments"]

            # Remove segments that are not spoken by the main speaker (keep the rest)
            other_speaker_intervals = [
                (float(start), float(end))
                for start, end, speaker_id, _text in segments
                if speaker_id != main_speaker
            ]

            if other_speaker_intervals:
                # Sort by time and merge overlaps
                other_speaker_intervals.sort(key=lambda x: x[0])
                merged_intervals: List[List[float]] = []
                for start, end in other_speaker_intervals:
                    start = max(0.0, start)
                    end = min(len(y) / sr, end)
                    if end <= start:
                        continue
                    if not merged_intervals or start > merged_intervals[-1][1]:
                        merged_intervals.append([start, end])
                    else:
                        merged_intervals[-1][1] = max(merged_intervals[-1][1], end)

                # Concatenate only the complement of removed intervals (i.e., remove other-speaker speech)
                kept_audio = []
                prev_sample = 0
                total_samples = len(y)
                for start_sec, end_sec in merged_intervals:
                    start_sample = int(start_sec * sr)
                    end_sample = int(end_sec * sr)
                    start_sample = max(0, min(start_sample, total_samples))
                    end_sample = max(0, min(end_sample, total_samples))
                    if end_sample <= start_sample:
                        continue

                    if start_sample > prev_sample:
                        kept_audio.append(y[prev_sample:start_sample])
                    prev_sample = end_sample

                if prev_sample < total_samples:
                    kept_audio.append(y[prev_sample:])

                y = np.concatenate(kept_audio) if kept_audio else np.zeros((0,), dtype=y.dtype)
                removed_s = sum((e - s) for s, e in merged_intervals)
                print(
                    f"  Removed other speech: {len(merged_intervals)} intervals, removed ~{removed_s:.2f}s, output {len(y)/sr:.2f}s"
                )
            else:
                print("  No other-speaker segments found; keeping full audio.")
    
    # Step 6: Save processed audio
    final_output_path = output_dir / f"{audio_path.stem}_processed.wav"
    sf.write(str(final_output_path), y, sr)
    return final_output_path

def preprocess_class_audios(
    data_dir: str = "data",
    split: str = "train",
    class_name: str = "ad",
    perform_volume_normalization: bool = True,
    perform_noise_removal: bool = True,
    perform_diarization: bool = True,
    extract_main_speaker: bool = True,
    volume_target_db: float = -20.0,
    language: str = "en",
    diarization_model: str = None,
    whisper_model_size: str = "base",
    hf_token: str = None,
    max_files: int = None
):
    """Preprocess all audios within a class.
    
    Args:
        perform_volume_normalization: Whether to perform volume normalization
        volume_target_db: Target volume in dB
        max_files: Maximum number of files to process (testing). If None, process all files
    """
    data_path = Path(data_dir)
    input_dir = data_path / split / class_name
    
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        return
    
    # Output directory
    output_dir = data_path / split / f"{class_name}_preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    diarization_pipeline = None
    if perform_diarization:
        print(f"Loading speaker diarization pipeline...")
        diarization_pipeline = load_speaker_diarization_pipeline(diarization_model, token=hf_token)
        if diarization_pipeline is None:
            print("Warning: Speaker diarization will be skipped")
            perform_diarization = False
    
    whisper_model = None
    if perform_diarization:
        print(f"Loading Whisper model...")
        whisper_model = load_whisper_model(whisper_model_size)
    
    # Process audio files
    audio_files = list(input_dir.glob("*.mp3")) + list(input_dir.glob("*.wav"))
    
    # Test mode: limit number of files
    if max_files is not None and max_files > 0:
        audio_files = audio_files[:max_files]
        print(f"\n[TEST MODE] Processing {len(audio_files)} audio files (limited from total) for {class_name.upper()}...")
    else:
        print(f"\nProcessing {len(audio_files)} audio files for {class_name.upper()}...")
    
    processed_files = []
    for i, audio_file in enumerate(audio_files):
        print(f"\n[{i+1}/{len(audio_files)}] Processing {audio_file.name}")
        
        output_path = preprocess_audio_file(
            audio_file,
            output_dir,
            diarization_pipeline=diarization_pipeline,
            whisper_model=whisper_model,
            perform_volume_normalization=perform_volume_normalization,
            perform_noise_removal=perform_noise_removal,
            perform_diarization=perform_diarization,
            extract_main_speaker=extract_main_speaker,
            volume_target_db=volume_target_db,
            language=language,
            split=split,
            class_name=class_name
        )
        
        if output_path and output_path.exists():
            processed_files.append(output_path)
    
    print(f"\nPreprocessing completed!")
    print(f"Processed {len(processed_files)}/{len(audio_files)} files")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess audio files")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Split to process")
    parser.add_argument("--class_name", type=str, default="ad", choices=["ad", "cn"], help="Class to process")
    parser.add_argument("--no_volume_normalization", action="store_true", help="Skip volume normalization")
    parser.add_argument("--no_noise_removal", action="store_true", help="Skip noise removal")
    parser.add_argument("--no_diarization", action="store_true", help="Skip speaker diarization")
    parser.add_argument("--no_extract_main_speaker", action="store_true", help="Skip main speaker extraction")
    parser.add_argument("--volume_target_db", type=float, default=-20.0, 
                       help="Target volume level in dB for normalization (default: -20.0)")
    parser.add_argument("--language", type=str, default="en", help="Audio language")
    parser.add_argument("--diarization_model", type=str, default=None, help="Path to diarization model")
    parser.add_argument("--whisper_model", type=str, default="base", help="Whisper model size")
    parser.add_argument("--hf_token", type=str, default=None, 
                       help="Hugging Face token (or set HF_TOKEN/HUGGINGFACE_TOKEN environment variable)")
    parser.add_argument("--max_files", type=int, default=None, 
                       help="Maximum number of files to process (for testing). If not specified, process all files.")
    
    args = parser.parse_args()
    
    preprocess_class_audios(
        data_dir=args.data_dir,
        split=args.split,
        class_name=args.class_name,
        perform_volume_normalization=not args.no_volume_normalization,
        perform_noise_removal=not args.no_noise_removal,
        perform_diarization=not args.no_diarization,
        extract_main_speaker=not args.no_extract_main_speaker,
        volume_target_db=args.volume_target_db,
        language=args.language,
        diarization_model=args.diarization_model,
        whisper_model_size=args.whisper_model,
        hf_token=args.hf_token,
        max_files=args.max_files
    )
