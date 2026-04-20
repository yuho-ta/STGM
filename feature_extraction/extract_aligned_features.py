"""
Extract aligned audio and text features.
Align audio segments with text at the token level.
"""
import os
import sys
# Ensure UTF-8 for logs on Windows terminals to avoid UnicodeEncodeError.
if sys.platform == "win32":
    import io

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import torch
import torchaudio
import librosa
import numpy as np
import math
import unicodedata
import logging
from datetime import datetime
from transformers import (
    Wav2Vec2Processor, Wav2Vec2Model,
    WhisperProcessor, WhisperModel, WhisperTokenizer,
    BertModel, BertTokenizer
)
import whisper

# Setup logger.
def setup_logger(name: str = "extract_aligned_features", log_dir: Path = None, log_level: int = logging.DEBUG):
    """Set up the logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Logger itself is set to INFO (handlers can capture DEBUG).
    
    # Clear existing handlers.
    logger.handlers.clear()
    
    # Set formatter.
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (filters by the specified log level).
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (only if log_dir is provided).
    # Do not create a log file for extract_aligned_features itself.
    if log_dir and name != "extract_aligned_features":
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)  # Record all logs to the file.
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Log file: {log_file.absolute()}")
    
    return logger

# Initialize default logger (can be overridden later; INFO level).
logger = setup_logger(log_level=logging.INFO)


class FeatureExtractor:
    """Feature extractor class."""
    
    # Quantize silence to this step, then group into coarse buckets.
    # Positive values are quantized with a minimum of 1 step.
    SILENCE_DURATION_QUANT_STEP_SEC = 1.0
    
    def __init__(
        self,
        audio_model: str = "wav2vec2",
        whisper_model_size: str = "base",
        device: str = None,
        language: str = "en",
        use_silence_token: bool = False
    ):
        """
        Args:
            audio_model: wav2vec2 (fixed)
            whisper_model_size: Whisper model size (base, small, medium, large)
            device: cuda or cpu
            language: Audio language
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.audio_model = audio_model
        self.language = language
        
        # Load Whisper model (for text features and timestamps).
        logger.info(f"Loading Whisper model: {whisper_model_size}")
        self.whisper_model = whisper.load_model(whisper_model_size)
        
        # Load WhisperProcessor and WhisperModel (transformers version) for timestamp extraction.
        whisper_model_name = f"openai/whisper-{whisper_model_size}"
        logger.info(f"Loading Whisper from transformers: {whisper_model_name}")
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
        self.whisper_tokenizer = WhisperTokenizer.from_pretrained(whisper_model_name)
        self.whisper_text_model = WhisperModel.from_pretrained(whisper_model_name).to(self.device)
        self.whisper_text_model.eval()
        
        # Load BERT model (for text feature extraction; English data).
        logger.info("Loading BERT model for text features: bert-base-uncased")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        self.bert_model.eval()
        # Silence cache: store BERT embeddings for coarse buckets (e.g., "<SILENCE3_5>").
        self._silence_duration_embedding_cache: Dict[str, torch.Tensor] = {}
        # Silence cache for audio: store wav2vec2 features (mean) for representative "0 waveform".
        self._silence_audio_feature_cache: Dict[str, torch.Tensor] = {}
        self._use_silence_token = bool(use_silence_token)
        
        # Load audio model (wav2vec2 fixed).
        if audio_model != "wav2vec2":
            raise ValueError(f"Only wav2vec2 is supported. Got: {audio_model}")
        
        self._load_audio_model()
        
        # Segment length setting (wav2vec2: 50 segments/second).
        self.segment_length = 50
    
    
    def _load_audio_model(self):
        """Load the audio model (wav2vec2 fixed)."""
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        self.audio_model.eval()
        logger.info(f"Loaded audio model: wav2vec2")
    
    def extract_text_features(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Extract text features with BERT (variable length; padding=longest; truncate to max_length)."""
        # Normalize the text.
        text = unicodedata.normalize("NFC", str(text))
        
        # Tokenize with BERT (do not pad to a fixed length; only to required length; max 512).
        inputs = self.bert_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Extract BERT features.
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # last_hidden_state: (batch_size, seq_len, hidden_dim)
            features = outputs.last_hidden_state.squeeze(0).cpu()  # (seq_len, hidden_dim)
        
        # NaN check.
        if torch.isnan(features).any():
            features = torch.nan_to_num(features, nan=0.0)
        
        return features
    
    @classmethod
    def _quantize_silence_duration_sec(cls, duration_sec: float) -> float:
        """Quantize a silence duration to the nearest SILENCE_DURATION_QUANT_STEP_SEC step."""
        step = cls.SILENCE_DURATION_QUANT_STEP_SEC
        d = float(duration_sec)
        if d <= 0:
            return 0.0
        q = math.floor(d / step + 0.5) * step
        if q < step:
            q = step
        return float(q)
    
    @classmethod
    def _coarse_silence_bucket_key(cls, quantized_sec: float) -> str:
        """Map a quantized duration (in seconds) to a coarse bucket key."""
        q = int(round(float(quantized_sec)))
        if q <= 0:
            return "0"
        if q <= 2:
            return "1_2"
        if q <= 4:
            return "3_5"
        if q <= 7:
            return "5_8"
        return "GE8"
    
    @classmethod
    def _silence_tag_for_bucket_key(cls, bucket_key: str) -> str:
        """Return the fixed BERT tag for a given silence bucket key."""
        if bucket_key == "0":
            return "<SILENCE0>"
        if bucket_key == "1_2":
            return "<SILENCE1_2>"
        if bucket_key == "3_5":
            return "<SILENCE3_5>"
        if bucket_key == "5_8":
            return "<SILENCE5_8>"
        if bucket_key == "GE8":
            return "<SILENCE_GE8>"
        return f"<SILENCE_{bucket_key}>"
    
    @classmethod
    def _silence_tag_for_duration(cls, duration_sec: float) -> str:
        """Create a silence tag from a raw silence duration."""
        d = cls._quantize_silence_duration_sec(duration_sec)
        if d <= 0:
            return "<SILENCE0>"
        bk = cls._coarse_silence_bucket_key(d)
        return cls._silence_tag_for_bucket_key(bk)
    
    def _embedding_for_silence_duration(self, duration_sec: float) -> torch.Tensor:
        """Return the average BERT embedding for a given coarse silence bucket."""
        d_key = self._quantize_silence_duration_sec(duration_sec)
        if d_key <= 0:
            bk = "0"
        else:
            bk = self._coarse_silence_bucket_key(d_key)
        if bk in self._silence_duration_embedding_cache:
            return self._silence_duration_embedding_cache[bk].clone()
        label = self._silence_tag_for_bucket_key(bk)
        feats = self.extract_text_features(label, max_length=64)
        vec = feats.mean(dim=0)
        vec = torch.nan_to_num(vec, nan=0.0)
        self._silence_duration_embedding_cache[bk] = vec.clone()
        return vec.clone()

    @classmethod
    def _representative_silence_duration_sec_for_bucket(cls, bucket_key: str) -> float:
        """
        Representative seconds used to build the representative "0 waveform" per coarse bucket.
        """
        if bucket_key == "0":
            return 0.0
        if bucket_key == "1_2":
            return 2.0
        if bucket_key == "3_5":
            return 4.0
        if bucket_key == "5_8":
            return 7.0
        if bucket_key == "GE8":
            return 8.0
        return 8.0

    def _audio_feature_vec_for_silence_bucket(self, bucket_key: str) -> torch.Tensor:
        """
        Return a fixed (audio_dim,) vector for silence AUDIO features.
        Use a representative "0 waveform" (based on the bucket) and mean-pool wav2vec2 outputs over time.
        """
        if bucket_key in self._silence_audio_feature_cache:
            return self._silence_audio_feature_cache[bucket_key].clone()

        rep_seconds = self._representative_silence_duration_sec_for_bucket(bucket_key)
        hidden_size = int(getattr(self.audio_model.config, "hidden_size", 768))

        if rep_seconds <= 0:
            vec = torch.zeros(hidden_size, dtype=torch.float32)
            self._silence_audio_feature_cache[bucket_key] = vec.clone()
            return vec.clone()

        sample_rate = 16000
        num_samples = int(rep_seconds * sample_rate)
        if num_samples <= 0:
            vec = torch.zeros(hidden_size, dtype=torch.float32)
            self._silence_audio_feature_cache[bucket_key] = vec.clone()
            return vec.clone()

        # wav2vec2 expects a 1-channel waveform (the processor handles required preprocessing).
        wave_form = torch.zeros(num_samples, dtype=torch.float32)
        inputs = self.audio_processor(
            wave_form, sampling_rate=sample_rate, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.audio_model(**inputs)
            features = outputs.last_hidden_state.squeeze(0).cpu()  # (seq_len, hidden_dim)

        # Replace NaNs with 0.0 (keep consistent with other features).
        if torch.isnan(features).any():
            features = torch.nan_to_num(features, nan=0.0)

        vec = features.mean(dim=0)
        vec = torch.nan_to_num(vec, nan=0.0)
        self._silence_audio_feature_cache[bucket_key] = vec.clone()
        return vec.clone()
    
    def extract_audio_features(self, audio_path: Path) -> torch.Tensor:
        """Extract audio features (wav2vec2)."""
        return self._extract_wav2vec2_features(audio_path)
    
    def _extract_wav2vec2_features(self, audio_path: Path) -> torch.Tensor:
        """Extract audio features with wav2vec2."""
        wave_form, sample_rate = torchaudio.load(str(audio_path))
        
        # Convert stereo to mono.
        if wave_form.shape[0] > 1:
            wave_form = wave_form.mean(dim=0, keepdim=True)
        
        # Resample to 16 kHz.
        wave_form = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(wave_form)
        sample_rate = 16000
        wave_form = wave_form.squeeze(0)
        
        # Extract features.
        inputs = self.audio_processor(wave_form, sampling_rate=sample_rate, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.audio_model(**inputs)
            features = outputs.last_hidden_state.squeeze(0).cpu()  # (seq_len, hidden_dim)
        
        # NaN check.
        if torch.isnan(features).any():
            features = torch.nan_to_num(features, nan=0.0)
        
        return features
    
    def get_word_timestamps(self, audio_path: Path) -> List[Tuple[str, float, float]]:
        """Get word-level timestamps from Whisper."""
        try:
            # Preload with librosa to avoid FFmpeg issues.
            audio_array, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            result = self.whisper_model.transcribe(
                audio_array,
                language=self.language,
                word_timestamps=True
            )
            
            words = []
            for segment in result.get("segments", []):
                # If the "words" field exists, use it.
                if "words" in segment:
                    for word_info in segment.get("words", []):
                        word = word_info.get("word", "").strip()
                        start = word_info.get("start", 0)
                        end = word_info.get("end", 0)
                        if word:
                            words.append((word, start, end))
                else:
                    # If "words" is not available, use the whole segment.
                    text = segment.get("text", "").strip()
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    if text:
                        # Split text into words (simple version).
                        for word in text.split():
                            word = word.strip()
                            if word:
                                # Evenly split the time range.
                                duration = (end - start) / len(text.split())
                                word_start = start + len(words) * duration
                                word_end = word_start + duration
                                words.append((word, word_start, word_end))
            
            return words
        except Exception as e:
            logger.error(f"Error getting word timestamps from {audio_path}: {e}")
            return []
    
    def _get_ordered_segments(
        self,
        word_timestamps: List[Tuple[str, float, float]],
        total_duration: float,
        min_word_gap_sec: float = 0.05,
    ) -> List[Tuple[str, float, float, Optional[Tuple[str, float, float]]]]:
        """
        Build time-ordered segments from word timestamps and total duration.

        Each element is (kind, start, end, word_data):
        - kind is "word" or "silence"
        - word_data is (word, start, end) for "word", and None for "silence".

        min_word_gap_sec: Minimum gap (seconds) between adjacent words to emit a "silence candidate".
        Whether to include a silence segment is controlled in align_features after quantization via min_silence_duration.
        """
        segments: List[Tuple[str, float, float, Optional[Tuple[str, float, float]]]] = []
        if not word_timestamps:
            return segments
        
        sorted_words = sorted(word_timestamps, key=lambda x: x[1])
        
        for i, (word, start, end) in enumerate(sorted_words):
            segments.append(("word", start, end, (word, start, end)))
            # Candidate silence: gap until the next word. Ignore very short noise gaps.
            if i + 1 < len(sorted_words):
                next_start = sorted_words[i + 1][1]
                gap = next_start - end
                if gap >= min_word_gap_sec:
                    segments.append(("silence", end, next_start, None))
        
        return segments
    
    def align_features(
        self,
        text: str,
        audio_features: torch.Tensor,
        word_timestamps: List[Tuple[str, float, float]],
        debug: bool = False,
        include_silence: bool = False,
        min_silence_duration: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor, torch.Tensor]:
        """
        Align text and audio features using Whisper word timestamps.

        If include_silence=True, insert silence candidates between words using coarse bucket tags
        (<SILENCE1_2>, <SILENCE3_5>, <SILENCE5_8>, <SILENCE_GE8>) as BERT embeddings.
        First, quantize to 1-second steps, then map q into [3,5), [5,8), [8,∞), and q<=2 buckets.

        Silences whose quantized q is <= min_silence_duration are not included.
        
        Returns:
            (aligned_audio_features, aligned_text_features, audio_valid_length, text_valid_length, audio_mask, text_mask)
            - aligned_audio_features: (num_segments, audio_dim) word + silence-level audio features
            - aligned_text_features: (num_segments, text_dim) word + silence-level text features
              (silence uses embeddings for <SILENCE{seconds}>).
        """
        bert_max_length = 512  # BERT max length. Do not pad to a fixed length.
        audio_dim = audio_features.shape[1]
        num_frames = audio_features.shape[0]
        total_duration = num_frames / self.segment_length
        
        # 1. Extract text features (variable-length for words).
        text_features = self.extract_text_features(text, bert_max_length)
        text_dim = text_features.shape[1]
        
        # 2. Get BERT offset mappings (same padding/truncation as extract_text_features).
        bert_inputs = self.bert_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=bert_max_length,
            return_attention_mask=True,
            return_offsets_mapping=True
        )
        bert_offset_mapping = bert_inputs["offset_mapping"][0]
        
        if include_silence:
            # Time-ordered segment sequence (word + silence candidates).
            # Silence inclusion is decided below after quantization using min_silence_duration.
            ordered = self._get_ordered_segments(
                word_timestamps, total_duration, min_word_gap_sec=0.05
            )
            aligned_audio_features = []
            aligned_text_features = []
            search_start = 0
            
            for kind, seg_start, seg_end, word_data in ordered:
                # Audio: mean-pool wav2vec2 frames for the corresponding time span.
                min_dur = 0.05
                if seg_end - seg_start < min_dur:
                    seg_end = seg_start + min_dur
                
                if kind == "silence":
                    silence_duration = seg_end - seg_start
                    q_sec = self._quantize_silence_duration_sec(silence_duration)
                    if q_sec <= min_silence_duration:
                        logger.debug(
                            f"  Skip silence ({seg_start:.2f}s-{seg_end:.2f}s, raw={silence_duration:.2f}s "
                            f"quant={q_sec:g}s <= min_silence_duration={min_silence_duration})"
                        )
                        continue

                if kind == "silence":
                    # Even if the original-side audio contains noise, silence AUDIO is always built as "0 waveform -> wav2vec2 -> mean".
                    bk = self._coarse_silence_bucket_key(q_sec)
                    audio_vec = self._audio_feature_vec_for_silence_bucket(bk).to(audio_features.dtype)
                else:
                    start_seg = max(0, math.floor(seg_start * self.segment_length))
                    end_seg = min(num_frames, math.ceil(seg_end * self.segment_length))
                    if end_seg > start_seg:
                        audio_vec = audio_features[start_seg:end_seg].mean(dim=0)
                    else:
                        audio_vec = audio_features.mean(dim=0)
                aligned_audio_features.append(audio_vec)
                
                if kind == "silence":
                    silence_duration = seg_end - seg_start
                    silence_tag = self._silence_tag_for_duration(silence_duration)
                    aligned_text_features.append(
                        self._embedding_for_silence_duration(silence_duration)
                    )
                    q_sec = self._quantize_silence_duration_sec(silence_duration)
                    bk = self._coarse_silence_bucket_key(q_sec)
                    logger.info(
                        f"  Silence ({seg_start:.2f}s-{seg_end:.2f}s, raw={silence_duration:.2f}s quant={q_sec:g}s bucket={bk}) -> {silence_tag}"
                    )
                else:
                    word, start_time, end_time = word_data
                    text_lower = text.lower()
                    word_lower = word.lower().strip()
                    punctuation_chars = ['.', ',', ';', ':', '!', '?']
                    word_base = word_lower
                    trailing_punct = None
                    if word_base and word_base[-1] in punctuation_chars:
                        trailing_punct = word_base[-1]
                        word_base = word_base[:-1].strip()
                    found_pos = text_lower.find(word_base, search_start)
                    word_char_start = found_pos
                    word_char_end = found_pos + len(word_base) if found_pos != -1 else -1
                    if found_pos != -1 and trailing_punct:
                        punct_search_start = word_char_end
                        punct_search_end = min(len(text_lower), punct_search_start + 2)
                        punct_text = text_lower[punct_search_start:punct_search_end]
                        if trailing_punct in punct_text:
                            punct_pos = punct_text.find(trailing_punct)
                            word_char_end = punct_search_start + punct_pos + 1
                    matched_token_indices = []
                    if found_pos == -1:
                        bert_token_mean = text_features.mean(dim=0)
                    else:
                        for bert_idx in range(text_features.shape[0]):
                            char_start, char_end = bert_offset_mapping[bert_idx].tolist()
                            if char_start < word_char_end and char_end > word_char_start:
                                matched_token_indices.append(bert_idx)
                        if not matched_token_indices:
                            bert_token_mean = text_features.mean(dim=0)
                            search_start = word_char_end
                        else:
                            bert_token_mean = text_features[matched_token_indices].mean(dim=0)
                            search_start = word_char_end
                    aligned_text_features.append(bert_token_mean)
                    logger.info(f"  Word '{word}' ({start_time:.2f}s-{end_time:.2f}s) -> Tokens{matched_token_indices}")
            
            num_segments = len(aligned_audio_features)
            if num_segments == 0:
                aligned_audio_features = torch.zeros(0, audio_dim)
                aligned_text_features = torch.zeros(0, text_dim)
                audio_valid_length = 0
                text_valid_length = 0
                audio_mask = torch.zeros(0, dtype=torch.bool)
                text_mask = torch.zeros(0, dtype=torch.bool)
            else:
                aligned_audio_features = torch.stack(aligned_audio_features)
                aligned_text_features = torch.stack(aligned_text_features)
                aligned_audio_features = torch.nan_to_num(aligned_audio_features, nan=0.0)
                aligned_text_features = torch.nan_to_num(aligned_text_features, nan=0.0)
                audio_valid_length = num_segments
                text_valid_length = num_segments
                audio_mask = torch.zeros(num_segments, dtype=torch.bool)
                text_mask = torch.zeros(num_segments, dtype=torch.bool)
            return aligned_audio_features, aligned_text_features, audio_valid_length, text_valid_length, audio_mask, text_mask
        
        # Traditional word-only alignment.
        aligned_audio_features = []
        aligned_text_features = []
        search_start = 0
        for word_idx, (word, start_time, end_time) in enumerate(word_timestamps):
            # Search the word in the text (time-ordered).
            # In BERT, punctuation becomes its own token (e.g., "hello ," -> ["hello", ","]).
            # Therefore, split punctuation and search separately.
            text_lower = text.lower()
            word_lower = word.lower().strip()
            
            # Split trailing punctuation (e.g., "hello," -> "hello" + ",").
            punctuation_chars = ['.', ',', ';', ':', '!', '?']
            word_base = word_lower
            trailing_punct = None
            
            # Detect trailing punctuation.
            if word_base and word_base[-1] in punctuation_chars:
                trailing_punct = word_base[-1]
                word_base = word_base[:-1].strip()
            
            # Search the word part first.
            found_pos = text_lower.find(word_base, search_start)
            word_char_start = found_pos
            word_char_end = found_pos + len(word_base) if found_pos != -1 else -1
            
            # If trailing punctuation exists, include the punctuation right after the word in the search range.
            if found_pos != -1 and trailing_punct:
                # Include the punctuation position in the search range (consider possible spaces).
                # Find punctuation in the text (immediately after the word; consider spaces).
                punct_search_start = word_char_end
                punct_search_end = min(len(text_lower), punct_search_start + 2)  # Space + punctuation
                punct_text = text_lower[punct_search_start:punct_search_end]
                
                # If punctuation is found, extend the range to include it.
                if trailing_punct in punct_text:
                    punct_pos = punct_text.find(trailing_punct)
                    word_char_end = punct_search_start + punct_pos + 1
            
            matched_token_indices = []
            
            # wav2vec2 feature: mean of the audio segment (computed even if it doesn't match text).
            # Guarantee a minimum duration (50ms) for zero or extremely short segments.
            min_duration = 0.05  # Minimum duration (50ms)
            if end_time - start_time < min_duration:
                end_time = start_time + min_duration
            
            start_segment = max(0, math.floor(start_time * self.segment_length))
            end_segment = min(audio_features.shape[0], math.ceil(end_time * self.segment_length))
            if end_segment > start_segment:
                audio_segment_mean = audio_features[start_segment:end_segment].mean(dim=0)
            else:
                audio_segment_mean = audio_features.mean(dim=0)
            
            if found_pos == -1:
                # If the word is not found in the text, use the audio segment feature and the global mean for text.
                bert_token_mean = text_features.mean(dim=0)
                
                logger.debug(f"  Word[{word_idx}]: '{word}' -> Audio segment used, text=mean (not found in text)")
            else:
                # Find BERT tokens corresponding to this word.
                # Consider the case where punctuation is a standalone token.
                for bert_idx in range(text_features.shape[0]):
                    char_start, char_end = bert_offset_mapping[bert_idx].tolist()
                    # Check if the token character span overlaps the word span.
                    if char_start < word_char_end and char_end > word_char_start:
                        matched_token_indices.append(bert_idx)
                
                # If no BERT tokens are found, use the audio segment feature and the global mean for text.
                if not matched_token_indices:
                    bert_token_mean = text_features.mean(dim=0)
                    logger.debug(f"  Word[{word_idx}]: '{word}' -> Audio segment used, text=mean (no BERT tokens matched)")
                    # Update next search start position.
                    search_start = word_char_end
                else:
                    # BERT feature: mean of the matched BERT tokens.
                    bert_token_mean = text_features[matched_token_indices].mean(dim=0)
                    # Update next search start position.
                    search_start = word_char_end
            
            # Add all words (use audio segment features even if there is no match in text).
            aligned_audio_features.append(audio_segment_mean)
            aligned_text_features.append(bert_token_mean)
            
            logger.debug(f"  Word[{word_idx}]: '{word}' -> Tokens{matched_token_indices} "
                      f"({start_time:.2f}s-{end_time:.2f}s)")
        
        # 5. Convert to tensors (only matched words).
        num_matched_words = len(aligned_audio_features)
        if num_matched_words == 0:
            # If no matched words exist, return empty tensors.
            audio_dim = audio_features.shape[1]
            text_dim = text_features.shape[1]
            aligned_audio_features = torch.zeros(0, audio_dim)
            aligned_text_features = torch.zeros(0, text_dim)
            audio_valid_length = 0
            text_valid_length = 0
            audio_mask = torch.zeros(0, dtype=torch.bool)
            text_mask = torch.zeros(0, dtype=torch.bool)
        else:
            aligned_audio_features = torch.stack(aligned_audio_features)  # (num_matched_words, audio_dim)
            aligned_text_features = torch.stack(aligned_text_features)  # (num_matched_words, text_dim)
            
            # 6. NaN check.
            aligned_audio_features = torch.nan_to_num(aligned_audio_features, nan=0.0)
            aligned_text_features = torch.nan_to_num(aligned_text_features, nan=0.0)
            
            # 7. Create masks (all valid).
            audio_valid_length = num_matched_words
            text_valid_length = num_matched_words
            audio_mask = torch.zeros(num_matched_words, dtype=torch.bool)
            text_mask = torch.zeros(num_matched_words, dtype=torch.bool)
        
        return aligned_audio_features, aligned_text_features, audio_valid_length, text_valid_length, audio_mask, text_mask


def extract_features_for_dataset(
    data_dir: str,
    split: str,
    audio_model: str = "wav2vec2",
    whisper_model_size: str = "base",
    device: str = None,
    language: str = "en",
    use_augmented: bool = True,
    input_dir_suffix: str = "_preprocessed",  # Input directory suffix (e.g., "_preprocessed", "_speed2x")
    num_augmentations: int = 1,
    debug: bool = False,  # Debug mode (output matching information)
    save_features: bool = True,  # Save features to files (False = process only)
    log_dir: Path = None,  # Log directory
    log_level: int = logging.DEBUG,  # Log level
    include_silence: bool = False,  # Whether to include silence segments (default False for backward compatibility)
    min_silence_duration: float = 2.0,  # If quantized q (after 1s quantization) <= this, exclude from the sequence
):
    """
    Extract and save features for the whole dataset.
    
    Args:
        data_dir: Data directory
        split: train or val
        audio_model: Audio model name (wav2vec2 fixed)
        whisper_model_size: Whisper model size
        device: Device
        language: Audio language
        use_augmented: Whether to process augmented audios (train only)
        log_dir: Log directory
    """
    # Update the global logger (first time).
    global logger
    if log_dir:
        logger = setup_logger("extract_aligned_features", log_dir=Path(log_dir), log_level=log_level)
    else:
        # Even when log_dir is not specified, set up the logger with the chosen level.
        logger = setup_logger("extract_aligned_features", log_dir=None, log_level=log_level)
    
    data_path = Path(data_dir)
    extractor = FeatureExtractor(
        audio_model="wav2vec2",  # fixed
        whisper_model_size=whisper_model_size,
        device=device,
        language=language,
        use_silence_token=include_silence
    )
    
    # File name suffixes.
    text_suffix = "_bert"  # BERT feature suffix
    audio_suffix = "_wav2vec2"
    
    output_suffix = f"{text_suffix}{audio_suffix}.pt"
    
    # Process each class (original audio).
    for class_name in ["ad", "cn"]:
        # Determine input directory (when input_dir_suffix is specified).
        if input_dir_suffix:
            input_dir = data_path / split / f"{class_name}{input_dir_suffix}"
            if input_dir.exists():
                class_dir = input_dir
                # With suffix directory, use all wav files.
                audio_files = list(class_dir.glob("*.wav"))
            else:
                # Fallback: use the original directory.
                class_dir = data_path / split / class_name
                if not class_dir.exists():
                    logger.warning(f"{class_dir} does not exist, skipping...")
                    continue
                audio_files = list(class_dir.glob("*.mp3")) + list(class_dir.glob("*.wav"))
        else:
            # If the preprocessed directory exists, use it (default behavior).
            preprocessed_dir = data_path / split / f"{class_name}_preprocessed"
            if preprocessed_dir.exists() and list(preprocessed_dir.glob("*_subject.wav")):
                class_dir = preprocessed_dir
                # In preprocessed directory, use only *_subject.wav
                audio_files = list(class_dir.glob("*_subject.wav"))
            else:
                class_dir = data_path / split / class_name
                if not class_dir.exists():
                    logger.warning(f"{class_dir} does not exist, skipping...")
                    continue
                # In the original directory, use all files.
                audio_files = list(class_dir.glob("*.mp3")) + list(class_dir.glob("*.wav"))
        
        logger.info(f"\n=== Processing {class_name.upper()} (original) ===")
        
        # Decide output directory (only when save_features=True).
        if save_features:
            if input_dir_suffix != "_preprocessed":
                output_dir = data_path / split / f"{class_name}{input_dir_suffix}"
            else:
                output_dir = data_path / split / class_name
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.info("  [No-save mode] Features will be processed but not saved to disk")
        
        for audio_file in audio_files:
            uid = audio_file.stem
            logger.info(f"Processing {uid}...")
            
            # Prepare output paths (recompute features even if files already exist).
            if save_features:
                text_output_path = output_dir / f"{uid}{text_suffix}.pt"
                audio_output_path = output_dir / f"{uid}{text_suffix}{audio_suffix}.pt"
                lengths_mask_path = output_dir / f"{uid}{text_suffix}_lengths_mask.pt"
            
            try:
                # Get text and word timestamps (load from JSON or compute with Whisper).
                text = None
                word_timestamps = None
                
                # Load from JSON (including word timestamps).
                timestamps_json = data_path / split / f"aug{num_augmentations}" / "shuffle" / f"{class_name}_shuffled_texts.json"
                if timestamps_json.exists():
                    import json
                    with open(timestamps_json, "r", encoding="utf-8") as f:
                        timestamps_data = json.load(f)
                    
                    # Search by audio file relative path.
                    rel_path = str(audio_file.relative_to(data_path))
                    if rel_path in timestamps_data:
                        data = timestamps_data[rel_path]
                        # For original audio, use original_text and original_word_timestamps.
                        text = data.get("original_text", "")
                        word_timestamps_list = data.get("original_word_timestamps", [])
                        if word_timestamps_list:
                            word_timestamps = [(w, s, e) for w, s, e in word_timestamps_list]
                
                # If missing in JSON, load from CSV.
                if not text:
                    text_csv = data_path / split / f"aug{num_augmentations}" / "shuffle" / f"{class_name}_shuffled_texts.csv"
                    if text_csv.exists():
                        # Read CSV with all columns as strings to avoid type inference issues
                        df = pd.read_csv(text_csv, dtype=str)
                        row = df[df["audio_path"].str.contains(uid, na=False)]
                        if not row.empty:
                            # Access Series values correctly using bracket notation
                            # For original audio, use original_text.
                            original_text = row["original_text"].iloc[0] if "original_text" in row.columns else ""
                            text = original_text if original_text and str(original_text) != "nan" else ""
                
                # If text and timestamps are both missing, compute with Whisper (once).
                if not text or not word_timestamps:
                    # Preload with librosa to avoid FFmpeg issues.
                    audio_array, sr = librosa.load(str(audio_file), sr=16000, mono=True)
                    result = extractor.whisper_model.transcribe(
                        audio_array, language=language, word_timestamps=True
                    )
                    if not text:
                        text = result["text"].strip()
                    if not word_timestamps:
                        word_timestamps = extractor.get_word_timestamps(audio_file)
                elif not word_timestamps:
                    # If text exists but timestamps are missing, compute timestamps only.
                    word_timestamps = extractor.get_word_timestamps(audio_file)
                
                # Extract audio features.
                audio_features = extractor.extract_audio_features(audio_file)
                
                # Alignment.
                aligned_audio_features, text_features, audio_valid_length, text_valid_length, audio_mask, text_mask = extractor.align_features(
                    text, audio_features, word_timestamps, debug=debug, include_silence=include_silence, min_silence_duration=min_silence_duration
                )
                
                # Skip when lengths are 0 (no matched words).
                if audio_valid_length == 0 or text_valid_length == 0:
                    logger.info(f"  No matched words for {uid} (audio_valid_length={audio_valid_length}, text_valid_length={text_valid_length}), skipping...")
                    continue
                
                # Save (only when save_features=True).
                if save_features:
                    # Save text features.
                    torch.save(text_features, text_output_path)
                    
                    # Save aligned audio features.
                    torch.save(aligned_audio_features, audio_output_path)
                    
                    # Save valid lengths and masks.
                    torch.save({
                        "audio_valid_length": audio_valid_length,
                        "text_valid_length": text_valid_length,
                        "audio_mask": audio_mask,
                        "text_mask": text_mask
                    }, lengths_mask_path)
                    
                    logger.info(f"  Saved features for {uid}")
                else:
                    logger.info(f"  Processed features for {uid} (not saved)")
                
            except Exception as e:
                # Avoid Unicode encoding errors in error messages.
                error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                logger.error(f"Error processing {uid}: {error_msg}")
                import traceback
                logger.error(traceback.format_exc())
                continue
    
    # Also process augmented audios (train split only, when enabled).
    if split == "train" and use_augmented:
        for class_name in ["ad", "cn"]:
            augmented_dir = data_path / split / f"aug{num_augmentations}" / f"{class_name}_augmented"
            metadata_file = augmented_dir / "metadata_with_reasr.json"
            
            if not augmented_dir.exists() or not metadata_file.exists():
                logger.info(f"\n=== Skipping {class_name.upper()} augmented (no metadata) ===")
                continue
            
            logger.info(f"\n=== Processing {class_name.upper()} (augmented) ===")
            
            # Load metadata.
            import json
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata_raw = json.load(f)
            
            # Backward compatibility: if metadata is a dict, use the "entries" field; if it's already a list, use it as-is.
            if isinstance(metadata_raw, dict):
                metadata = metadata_raw.get("entries", [])
            else:
                metadata = metadata_raw
            
            for item in metadata:
                augmented_audio_path = data_path / item["augmented_audio"]
                if not augmented_audio_path.exists():
                    continue
                
                uid = augmented_audio_path.stem
                logger.info(f"Processing {uid}...")
                
                try:
                    # Save (only when save_features=True).
                    if save_features:
                        text_output_path = augmented_dir / f"{uid}{text_suffix}.pt"
                        audio_output_path = augmented_dir / f"{uid}{text_suffix}{audio_suffix}.pt"
                        lengths_mask_path = augmented_dir / f"{uid}{text_suffix}_lengths_mask.pt"

                    # Recompute even if feature files already exist.
                    
                    # Get text (prefer reasr_text; otherwise use shuffled_text).
                    text = item.get("reasr_text", item.get("shuffled_text", ""))
                    if not text:
                        # Preload with librosa to avoid FFmpeg issues.
                        audio_array, sr = librosa.load(str(augmented_audio_path), sr=16000, mono=True)
                        result = extractor.whisper_model.transcribe(audio_array, language=language)
                        text = result["text"].strip()
                    
                    # Get word timestamps.
                    word_timestamps = extractor.get_word_timestamps(augmented_audio_path)
                    
                    # Extract audio features.
                    audio_features = extractor.extract_audio_features(augmented_audio_path)
                    
                    # Alignment.
                    aligned_audio_features, text_features, audio_valid_length, text_valid_length, audio_mask, text_mask = extractor.align_features(
                        text, audio_features, word_timestamps, debug=debug, include_silence=include_silence, min_silence_duration=min_silence_duration
                    )
                    
                    # Skip when lengths are 0 (no matched words).
                    if audio_valid_length == 0 or text_valid_length == 0:
                        logger.info(f"  No matched words for {uid} (audio_valid_length={audio_valid_length}, text_valid_length={text_valid_length}), skipping...")
                        continue
                    
                    # Save (only when save_features=True).
                    if save_features:
                        # Save text features.
                        torch.save(text_features, text_output_path)
                        
                        # Save aligned audio features.
                        torch.save(aligned_audio_features, audio_output_path)
                        
                        # Save valid lengths and masks.
                        torch.save({
                            "audio_valid_length": audio_valid_length,
                            "text_valid_length": text_valid_length,
                            "audio_mask": audio_mask,
                            "text_mask": text_mask
                        }, lengths_mask_path)
                        
                        logger.info(f"  Saved features for {uid}")
                    else:
                        logger.info(f"  Processed features for {uid} (not saved)")
                    
                except Exception as e:
                    # Avoid Unicode encoding errors in error messages.
                    error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                    logger.error(f"Error processing {uid}: {error_msg}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract aligned audio-text features")
    parser.add_argument("--data_dir", type=str, default="data_all", help="Data directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Split to process")
    parser.add_argument("--audio_model", type=str, default="wav2vec2",
                       help="Audio model (wav2vec2 fixed)")
    parser.add_argument("--whisper_model", type=str, default="base",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--language", type=str, default="en", help="Audio language")
    parser.add_argument("--use_augmented", action="store_true", default=True, help="Process augmented audios (train only)")
    parser.add_argument("--no_augmented", dest="use_augmented", action="store_false", help="Skip augmented audios")
    parser.add_argument("--input_dir_suffix", type=str, default="_preprocessed",
                       help="Input directory suffix (e.g., '_preprocessed', '_speed2x'). If specified, uses {class_name}{suffix} directory.")
    parser.add_argument("--num_augmentations", type=int, default=1,
                       help="Number of augmentations per audio (for selecting aug folder)")
    parser.add_argument("--debug", action="store_true", default=False,
                       help="Enable debug mode to output word timestamp matching information")
    parser.add_argument("--no_save", action="store_true", default=False,
                       help="Process features but do not save to disk (useful for testing/debugging)")
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Directory for log files")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Log level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--min_silence_duration", type=float, default=2.00,
                       help="Drop silence if 1s-quantized length q <= this (seconds). Default 2 keeps coarse buckets from 3_5 upward only")
    parser.add_argument("--include_silence", action="store_true", default=True,
                       help="Include silence in alignment")
    args = parser.parse_args()
    
    # Convert log level from string to numeric value.
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    log_level = log_level_map[args.log_level]
    
    # In debug mode, use DEBUG level.
    if args.debug:
        log_level = logging.DEBUG
    
    # Initialize logger.
    logger = setup_logger("extract_aligned_features", log_dir=Path(args.log_dir), log_level=log_level)
    
    extract_features_for_dataset(
        data_dir=args.data_dir,
        split=args.split,
        audio_model=args.audio_model,
        whisper_model_size=args.whisper_model,
        device=args.device,
        language=args.language,
        use_augmented=args.use_augmented,
        input_dir_suffix=args.input_dir_suffix,
        num_augmentations=args.num_augmentations,
        debug=args.debug,
        save_features=not args.no_save,  # If --no_save is specified, save_features=False.
        log_dir=Path(args.log_dir),
        log_level=log_level,  # Pass the log level retrieved from the command line.
        include_silence=args.include_silence,
        min_silence_duration=args.min_silence_duration
    )

