"""
Shuffle texts within the same class.
Break the dependency between audio and text so the classifier focuses on class signal.
Shuffle across different subjects (do not reuse text from the same subject).
"""
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import whisper
import librosa

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

def transcribe_audio(audio_path: Path, model, language: str = "en", word_timestamps: bool = False) -> tuple:
    """
    Transcribe audio with Whisper.
    
    Returns:
        word_timestamps=False: (text, None)
        word_timestamps=True: (text, word_timestamps_list)
    """
    try:
        # Preload with librosa to avoid FFmpeg issues
        audio_array, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        result = model.transcribe(audio_array, language=language, word_timestamps=word_timestamps)
        text = result["text"].strip()
        
        if word_timestamps:
            # Extract word-level timestamps
            word_timestamps_list = []
            for segment in result.get("segments", []):
                if "words" in segment:
                    for word_info in segment.get("words", []):
                        word = word_info.get("word", "").strip()
                        start = word_info.get("start", 0)
                        end = word_info.get("end", 0)
                        if word:
                            word_timestamps_list.append((word, start, end))
            return text, word_timestamps_list
        else:
            return text, None
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        if word_timestamps:
            return "", None
        else:
            return "", None

def extract_subject_id(filename: str) -> str:
    """
    Extract subject ID from filename (up to the first "-").
    Example: "001-0.mp3" -> "001", "002-1.wav" -> "002"
    """
    # Remove extension from filename
    name_without_ext = Path(filename).stem
    # Split by "-" and take the first part as subject ID
    if '-' in name_without_ext:
        return name_without_ext.split('-')[0]
    else:
        # If "-" is not present, use the whole filename as subject ID
        return name_without_ext

def get_texts_for_class(
    audio_dir: Path,
    whisper_model,
    class_name: str,
    language: str = "en",
    include_word_timestamps: bool = True,
    data_dir: Path = None,
    split: str = None
) -> List[Tuple[Path, str, list, str]]:
    """
    Transcribe all audios in the class (including subject_id).
    Reuse existing text/timestamps if available.
    
    Args:
        audio_dir: Directory containing audio files
        whisper_model: Whisper model (used only when existing text is not available)
        class_name: Class name
        language: Language
        include_word_timestamps: Whether to include word timestamps
        data_dir: Data directory (used to search existing text)
        split: Split name (used to search existing text)
    
    Returns:
        A list of [(audio_path, text, word_timestamps, subject_id), ...]
        word_timestamps: [(word, start, end), ...] or None
        subject_id: Subject ID (up to the first "-")
    """
    # If using a preprocessed directory, use *_processed.wav or *_subject.wav.
    # Otherwise, use all wav/mp3 files.
    if "_preprocessed" in str(audio_dir):
        # Look for *_processed.wav or *_subject.wav in the preprocessed directory
        audio_files = list(audio_dir.glob("*_processed.wav")) + list(audio_dir.glob("*_subject.wav"))
    else:
        audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))
    texts = []
    
    # Search existing text/timestamps (from diarization results)
    existing_texts = {}
    if data_dir and split:
        # Search diarization result JSON files
        diarization_dir = data_dir / split / f"{class_name}_preprocessed"
        if diarization_dir.exists():
            for json_file in diarization_dir.glob("*_diarization.json"):
                try:
                    import json
                    with open(json_file, "r", encoding="utf-8") as f:
                        diar_data = json.load(f)
                    # Find the entry matching the audio filename
                    audio_stem = json_file.stem.replace("_diarization", "")
                    # Extract text and timestamps from whisper_segments
                    whisper_segments = diar_data.get("whisper_segments", [])
                    if whisper_segments:
                        full_text = " ".join([seg.get("text", "").strip() for seg in whisper_segments])
                        word_timestamps_list = []
                        for seg in whisper_segments:
                            if "words" in seg:
                                for word_info in seg.get("words", []):
                                    word = word_info.get("word", "").strip()
                                    start = word_info.get("start", 0)
                                    end = word_info.get("end", 0)
                                    if word:
                                        word_timestamps_list.append((word, start, end))
                        if full_text:
                            existing_texts[audio_stem] = (full_text, word_timestamps_list if word_timestamps_list else None)
                except Exception as e:
                    pass  # Ignore if reading fails
    
    print(f"Transcribing {len(audio_files)} audios for {class_name.upper()}...")
    if existing_texts:
        print(f"  Found {len(existing_texts)} existing transcriptions, will reuse them")
    
    for i, audio_file in enumerate(audio_files):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(audio_files)}")
        
        # Extract subject ID
        subject_id = extract_subject_id(audio_file.name)
        
        # Check existing text/timestamps
        audio_stem = audio_file.stem
        if audio_stem in existing_texts:
            text, word_timestamps = existing_texts[audio_stem]
            if text:
                if include_word_timestamps and word_timestamps:
                    texts.append((audio_file, text, word_timestamps, subject_id))
                elif not include_word_timestamps:
                    texts.append((audio_file, text, None, subject_id))
                continue  # Skip since we used existing text
        
        # If there is no existing text, transcribe with Whisper
        if include_word_timestamps:
            text, word_timestamps = transcribe_audio(
                audio_file, whisper_model, language=language, word_timestamps=True
            )
            if text:
                texts.append((audio_file, text, word_timestamps, subject_id))
        else:
            text, _ = transcribe_audio(
                audio_file, whisper_model, language=language, word_timestamps=False
            )
            if text:
                texts.append((audio_file, text, None, subject_id))
    
    print(f"Successfully transcribed {len(texts)}/{len(audio_files)} audios")
    return texts

def add_pause_markers_to_text(text: str, word_timestamps: list, pause_threshold_short: float = 0.15, pause_threshold_long: float = 0.5) -> str:
    """
    Use word timestamps to insert pause markers into silence gaps.
    
    Args:
        text: Original text
        word_timestamps: List of [(word, start, end), ...]
        pause_threshold_short: Short pause threshold (seconds, default: 0.15)
        pause_threshold_long: Long pause threshold (seconds, default: 0.5)
    
    Returns:
        Text with pause markers inserted.
    
    Note:
        Add '.' tokens based on silence duration / PERIOD_DURATION.
        Example: 0.6s silence -> 2 periods (. .)
    """
    if not word_timestamps or len(word_timestamps) < 2:
        return text
    
    # Silence duration per '.' token (seconds)
    PERIOD_DURATION = 0.5
    
    # Compute silence gaps from word_timestamps and add pause markers
    result_parts = []
    
    for i in range(len(word_timestamps)):
        word, start, end = word_timestamps[i]
        
        # Append the word (keep punctuation)
        result_parts.append(word)
        
        # Check silence gap to the next word
        if i < len(word_timestamps) - 1:
            next_word, next_start, _ = word_timestamps[i + 1]
            silence_duration = next_start - end
            
            # Add periods based on silence_duration / PERIOD_DURATION
            if silence_duration >= pause_threshold_short:
                num_periods = max(1, int(silence_duration / PERIOD_DURATION))
                # Add the period+space pattern (with leading/trailing spaces: " . . . ")
                result_parts.append(" " + " ".join(["."] * num_periods) + " ")
            else:
                # Very short gap: add a space only
                result_parts.append(" ")
    
    return "".join(result_parts)


def split_text_into_sentences(text: str, word_timestamps: list) -> List[Tuple[str, list, float]]:
    """
    Split text into sentence units and also split corresponding word_timestamps.
    Each sentence includes the silence duration until the next sentence.
    
    Returns:
        [(sentence_text, sentence_word_timestamps, silence_duration_after), ...]
        silence_duration_after: Silence duration (seconds) after this sentence until the next one.
    """
    import re
    
    if not word_timestamps:
        # If word_timestamps are missing, split by punctuation (periods etc.)
        sentences = re.split(r'([.!?]+)', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        # Re-attach the sentence-ending punctuation marks
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                result.append((sentences[i] + sentences[i + 1], [], 0.5))  # Default 0.5 seconds
            else:
                result.append((sentences[i], [], 0.0))
        return result
    
    # Split into sentences using word_timestamps
    sentences = []
    current_sentence_words = []
    current_sentence_timestamps = []
    
    for idx, (word, start, end) in enumerate(word_timestamps):
        current_sentence_words.append(word)
        current_sentence_timestamps.append((word, start, end))
        
        # Split the sentence when the token ends a sentence (. ! ?)
        word_clean = word.rstrip('.,!?;:')
        if word != word_clean or word.endswith('.') or word.endswith('!') or word.endswith('?'):
            sentence_text = " ".join(current_sentence_words)
            if sentence_text.strip():
                # Compute silence gap until the next sentence
                silence_duration = 0.0
                if idx < len(word_timestamps) - 1:
                    # Get the start time of the next word
                    next_word, next_start, _ = word_timestamps[idx + 1]
                    silence_duration = next_start - end
                
                sentences.append((sentence_text, current_sentence_timestamps.copy(), silence_duration))
            current_sentence_words = []
            current_sentence_timestamps = []
    
    # Add remaining words
    if current_sentence_words:
        sentence_text = " ".join(current_sentence_words)
        if sentence_text.strip():
            # Last sentence: silence duration is 0
            sentences.append((sentence_text, current_sentence_timestamps, 0.0))
    
    return sentences

def combine_sentences_from_texts(
    texts: List[Tuple[Path, str, list, str]],
    random_seed: int = 42,
    exclude_original: bool = True,
    add_pause_markers: bool = True,
        pause_threshold_short: float = 0.15,
        pause_threshold_long: float = 0.5
) -> List[Tuple[Path, str, str, list]]:
    """
    Combine texts at sentence granularity within the same class (preserve silence info).
    Shuffle across different subjects (do not use text from the same subject).
    
    Args:
        texts: [(audio_path, original_text, word_timestamps, subject_id), ...]
        random_seed: Random seed
        exclude_original: Whether to exclude the original text
            (if True, the same text will not be assigned)
        add_pause_markers: Whether to add pause markers
        pause_threshold_short: Short pause threshold (seconds)
        pause_threshold_long: Long pause threshold (seconds)
    
    Returns:
        [(audio_path, original_text, combined_text, original_word_timestamps, combined_word_timestamps), ...]
        Note: Since sentences are shuffled/combined, `combined_word_timestamps` is built by
        concatenating timestamps from different audios, which becomes discontinuous across
        sentences. Do not use for alignment; return an empty list instead.
    """
    random.seed(random_seed)
    
    # Split each text into sentence units
    all_sentences = []
    for audio_path, text, word_timestamps, subject_id in texts:
        sentences = split_text_into_sentences(text, word_timestamps)
        all_sentences.append((audio_path, text, word_timestamps, subject_id, sentences))
    
    # Generate text combined with the original audio
    result = []
    for i, (audio_path, original_text, original_word_timestamps, current_subject_id, original_sentences) in enumerate(all_sentences):
        if exclude_original and len(all_sentences) > 1:
            # Select sentences from a list excluding the current subject
            available_sentences = [
                sentences for j, (_, _, _, subject_id, sentences) in enumerate(all_sentences) 
                if subject_id != current_subject_id
            ]
            # Flatten
            flat_sentences = []
            for sentences_list in available_sentences:
                flat_sentences.extend(sentences_list)
        else:
            # Use all sentences (including the same subject)
            flat_sentences = []
            for _, _, _, _, sentences in all_sentences:
                flat_sentences.extend(sentences)
        
        # Shuffle and combine sentences
        shuffled_flat = flat_sentences.copy()
        random.shuffle(shuffled_flat)
        
        # Select the same number as the original sentences
        num_sentences = len(original_sentences)
        selected_sentences = shuffled_flat[:num_sentences] if len(shuffled_flat) >= num_sentences else shuffled_flat
        
        # Combine sentences (add pause markers between and within sentences)
        PERIOD_DURATION = 0.5  # same as add_pause_markers_to_text
        combined_text_parts = []
        for idx, (sentence_text, sentence_timestamps, silence_duration_after) in enumerate(selected_sentences):
            if add_pause_markers and sentence_timestamps:
                sentence_text_to_append = add_pause_markers_to_text(
                    sentence_text,
                    sentence_timestamps,
                    pause_threshold_short=pause_threshold_short,
                    pause_threshold_long=pause_threshold_long
                )
            else:
                sentence_text_to_append = sentence_text
            combined_text_parts.append(sentence_text_to_append)
            if idx < len(selected_sentences) - 1:
                # Between-sentence pause: mark the silence after this utterance
                if add_pause_markers and silence_duration_after >= pause_threshold_short:
                    num_periods = max(1, int(silence_duration_after / PERIOD_DURATION))
                    combined_text_parts.append(" " + " ".join(["."] * num_periods) + " ")
                else:
                    combined_text_parts.append(" ")
        combined_text_with_pauses = "".join(combined_text_parts).strip()
        # Shuffled combination yields discontinuous timestamps across sentences; don't return for alignment
        result.append((audio_path, original_text, combined_text_with_pauses, original_word_timestamps, []))
    
    return result

def shuffle_texts_within_class(
    texts: List[Tuple[Path, str, list, str]],
    random_seed: int = 42,
    exclude_original: bool = True,
    add_pause_markers: bool = True,
        pause_threshold_short: float = 0.15,
        pause_threshold_long: float = 0.5,
    use_sentence_combination: bool = True
) -> List[Tuple[Path, str, str, list]]:
    """
    Shuffle or combine texts within the same class.
    Shuffle across different subjects (do not use text from the same subject).
    
    Args:
        texts: [(audio_path, original_text, word_timestamps, subject_id), ...]
        random_seed: Random seed
        exclude_original: Whether to exclude the original text
            (if True, the same text will not be assigned)
        add_pause_markers: Whether to add pause markers
        pause_threshold_short: Short pause threshold (seconds)
        pause_threshold_long: Long pause threshold (seconds)
        use_sentence_combination: If True, combine at sentence granularity (preserve silence info)
    
    Returns:
        [(audio_path, original_text, shuffled_text, original_word_timestamps, shuffled_word_timestamps), ...]
    """
    if use_sentence_combination:
        return combine_sentences_from_texts(
            texts,
            random_seed=random_seed,
            exclude_original=exclude_original,
            add_pause_markers=add_pause_markers,
            pause_threshold_short=pause_threshold_short,
            pause_threshold_long=pause_threshold_long
        )
    else:
        # Legacy full shuffle (exclude by subject_id)
        random.seed(random_seed)
        
        # Keep (text, word_timestamps) pairs along with subject_id
        text_data_list = [(text, word_ts, subject_id) for _, text, word_ts, subject_id in texts]
        
        # Combine original audio with shuffled text
        result = []
        for i, (audio_path, original_text, word_timestamps, current_subject_id) in enumerate(texts):
            if exclude_original and len(text_data_list) > 1:
                # Select from a list excluding the current subject's texts
                available_data = [(t, ts, sid) for j, (t, ts, sid) in enumerate(text_data_list) 
                                 if sid != current_subject_id]
                if len(available_data) > 0:
                    shuffled_text, shuffled_word_timestamps, _ = random.choice(available_data)
                else:
                    # If no other subjects exist, use original text (warn)
                    print(f"Warning: No other subjects found for {current_subject_id}, using original text")
                    shuffled_text, shuffled_word_timestamps = original_text, word_timestamps
            else:
                # Simple shuffle (may include the original text)
                shuffled_data = text_data_list.copy()
                random.shuffle(shuffled_data)
                shuffled_text, shuffled_word_timestamps, _ = shuffled_data[i]
            
            # Add pause markers using word_timestamps from the shuffled text
            if add_pause_markers and shuffled_word_timestamps:
                shuffled_text_with_pauses = add_pause_markers_to_text(
                    shuffled_text, 
                    shuffled_word_timestamps,
                    pause_threshold_short=pause_threshold_short,
                    pause_threshold_long=pause_threshold_long
                )
            else:
                shuffled_text_with_pauses = shuffled_text

            # Save both original audio word_timestamps and shuffled text word_timestamps
            result.append((audio_path, original_text, shuffled_text_with_pauses, word_timestamps, shuffled_word_timestamps))
        
        return result

def process_class_shuffle(
    data_dir: str = "data",
    split: str = "train",
    class_name: str = "ad",
    whisper_model_size: str = "base",
    language: str = "en",
    random_seed: int = 42,
    output_file: str = None,
    add_pause_markers: bool = True,
        pause_threshold_short: float = 0.15,
        pause_threshold_long: float = 0.5,
    use_sentence_combination: bool = True,
    num_augmentations: int = 1
):
    """
    Execute text shuffle within a class.
    
    Args:
        data_dir: Data directory
        split: train or val
        class_name: ad or cn
        whisper_model_size: Whisper model size
        language: Audio language ("en" for English, "ja" for Japanese)
        random_seed: Random seed
        output_file: Output file path (CSV)
        num_augmentations: Number of shuffled texts to generate per audio (default: 1)
    """
    data_path = Path(data_dir)
    # Use preprocessed audios if available
    preprocessed_dir = data_path / split / f"{class_name}_preprocessed"
    if preprocessed_dir.exists() and list(preprocessed_dir.glob("*.wav")):
        class_dir = preprocessed_dir
        print(f"Using preprocessed audio directory: {class_dir}")
    else:
        class_dir = data_path / split / class_name
    
    if not class_dir.exists():
        print(f"Error: {class_dir} does not exist")
        return
    
    # Load Whisper model (used only when there is no existing text)
    model = load_whisper_model(whisper_model_size)
    
    # Transcribe (also extract word timestamps and subject_id).
    # Prefer existing text/timestamps if available.
    texts = get_texts_for_class(
        class_dir, 
        model, 
        class_name, 
        language=language, 
        include_word_timestamps=True,
        data_dir=data_path,
        split=split
    )
    
    if len(texts) == 0:
        print(f"No texts found for {class_name}")
        return
    
    # Show subject statistics
    subject_ids = [subject_id for _, _, _, subject_id in texts]
    unique_subjects = set(subject_ids)
    print(f"Found {len(unique_subjects)} unique subjects: {sorted(unique_subjects)}")
    print(f"Shuffling texts across different subjects (excluding same subject)...")
    print(f"Generating {num_augmentations} shuffled text(s) per audio...")
    
    # Generate num_augmentations different shuffled texts per original audio.
    # Use different random seeds to increase diversity.
    all_shuffled = []
    for aug_idx in range(num_augmentations):
        # Use a different random_seed for each augmentation
        aug_seed = random_seed + aug_idx * 1000  # Shift seed to produce different results
        shuffled = shuffle_texts_within_class(
            texts, 
            random_seed=aug_seed, 
            exclude_original=True,
            add_pause_markers=add_pause_markers,
            pause_threshold_short=pause_threshold_short,
            pause_threshold_long=pause_threshold_long,
            use_sentence_combination=use_sentence_combination  # combine at sentence granularity (preserve silence info)
        )
        all_shuffled.extend(shuffled)
    
    # Save results (CSV and JSON)
    if output_file is None:
        # Split into a subdirectory per num_augmentations
        aug_dir = data_path / split / f"aug{num_augmentations}" / "shuffle"
        aug_dir.mkdir(parents=True, exist_ok=True)
        output_file = aug_dir / f"{class_name}_shuffled_texts.csv"
    
    import csv
    import json
    
    # CSV file (text only)
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "original_text", "shuffled_text"])
        for audio_path, original_text, shuffled_text, _, _ in all_shuffled:
            writer.writerow([
                str(audio_path.relative_to(data_path)),
                original_text,
                shuffled_text
            ])
    
    # JSON file (also stores word timestamps)
    timestamps_file = output_file.with_suffix(".json")
    timestamps_data = {}
    for audio_path, original_text, shuffled_text, original_word_timestamps, shuffled_word_timestamps in all_shuffled:
        rel_path = str(audio_path.relative_to(data_path))
        # If multiple shuffled texts exist for the same audio file, store them as a list
        if rel_path not in timestamps_data:
            timestamps_data[rel_path] = {
                "original_text": original_text,
                "shuffled_texts": [],
                "original_word_timestamps": original_word_timestamps if original_word_timestamps else [],
                "shuffled_word_timestamps_list": []
            }
        timestamps_data[rel_path]["shuffled_texts"].append(shuffled_text)
        timestamps_data[rel_path]["shuffled_word_timestamps_list"].append(
            shuffled_word_timestamps if shuffled_word_timestamps else []
        )
    
    with open(timestamps_file, "w", encoding="utf-8") as f:
        json.dump(timestamps_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved shuffled texts to {output_file}")
    print(f"Saved word timestamps to {timestamps_file}")
    print(f"Total: {len(all_shuffled)} pairs ({len(texts)} audios × {num_augmentations} augmentations)")
    
    # Show sample pairs
    print("\nSample pairs:")
    for i, (audio_path, orig, shuffled, _, _) in enumerate(shuffled[:3]):
        print(f"\n{i+1}. {audio_path.name}")
        print(f"   Original:  {orig[:50]}...")
        print(f"   Shuffled:  {shuffled[:50]}...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Shuffle texts within class")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Split to process")
    parser.add_argument("--class_name", type=str, default="ad", choices=["ad", "cn"], help="Class to process")
    parser.add_argument("--whisper_model", type=str, default="base", help="Whisper model size")
    parser.add_argument("--language", type=str, default="en", help="Audio language (en/ja)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path")
    parser.add_argument("--add_pause_markers", action="store_true", default=True, help="Add pause markers based on silence duration")
    parser.add_argument("--no_pause_markers", dest="add_pause_markers", action="store_false", help="Disable pause markers")
    parser.add_argument("--pause_threshold_short", type=float, default=0.15, help="Short pause threshold in seconds (default: 0.15, lower = more pauses)")
    parser.add_argument("--pause_threshold_long", type=float, default=0.5, help="Long pause threshold in seconds (default: 0.5, lower = more pauses)")
    parser.add_argument("--use_full_shuffle", action="store_true", default=False, help="Use full text shuffle instead of sentence combination (default: sentence combination to preserve silence info)")
    parser.add_argument("--num_augmentations", type=int, default=1, help="Number of shuffled texts to generate per audio (default: 1)")
    
    args = parser.parse_args()
    
    process_class_shuffle(
        data_dir=args.data_dir,
        split=args.split,
        class_name=args.class_name,
        whisper_model_size=args.whisper_model,
        language=args.language,
        random_seed=args.seed,
        output_file=args.output,
        add_pause_markers=args.add_pause_markers,
        pause_threshold_short=args.pause_threshold_short,
        pause_threshold_long=args.pause_threshold_long,
        use_sentence_combination=not args.use_full_shuffle,
        num_augmentations=args.num_augmentations
    )

