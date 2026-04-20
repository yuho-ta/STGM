"""
Train audio classification model.
Uses aligned features.
"""
import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend (no GUI needed)
import matplotlib.pyplot as plt
try:
    from sklearn.metrics import precision_score, recall_score, f1_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    # Fallback when scikit-learn is not installed
    def precision_score(y_true, y_pred, zero_division=0.0):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        if tp + fp == 0:
            return zero_division
        return float(tp / (tp + fp))
    
    def recall_score(y_true, y_pred, zero_division=0.0):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        if tp + fn == 0:
            return zero_division
        return float(tp / (tp + fn))
    
    def f1_score(y_true, y_pred, zero_division=0.0):
        precision = precision_score(y_true, y_pred, zero_division=zero_division)
        recall = recall_score(y_true, y_pred, zero_division=zero_division)
        if precision + recall == 0:
            return zero_division
        return float(2 * (precision * recall) / (precision + recall))


def _precision_recall_f1_per_class(y_true, y_pred, zero_division=0.0):
    """
    Compute per-class precision, recall, F1 for binary classification.
    Class 0 = ad, Class 1 = cn.
    Returns dict with keys: precision_ad, recall_ad, f1_ad, precision_cn, recall_cn, f1_cn (values in 0-100).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if _HAS_SKLEARN:
        prec = precision_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=zero_division)
        rec = recall_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=zero_division)
        f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=zero_division)
        return {
            "precision_ad": float(prec[0]) * 100.0, "recall_ad": float(rec[0]) * 100.0, "f1_ad": float(f1[0]) * 100.0,
            "precision_cn": float(prec[1]) * 100.0, "recall_cn": float(rec[1]) * 100.0, "f1_cn": float(f1[1]) * 100.0,
        }
    out = {}
    for label, name in [(0, "ad"), (1, "cn")]:
        pred_k = (y_pred == label)
        true_k = (y_true == label)
        tp = (pred_k & true_k).sum()
        pred_pos = pred_k.sum()
        true_pos = true_k.sum()
        prec = zero_division if pred_pos == 0 else float(tp / pred_pos)
        rec = zero_division if true_pos == 0 else float(tp / true_pos)
        f1 = zero_division if (prec + rec == 0) else float(2 * prec * rec / (prec + rec))
        out[f"precision_{name}"] = prec * 100.0
        out[f"recall_{name}"] = rec * 100.0
        out[f"f1_{name}"] = f1 * 100.0
    return out
try:
    from .model import CrossAttentionTransformer
except ImportError:
    # Fallback when running directly
    from model import CrossAttentionTransformer

class AlignedFeatureDataset(Dataset):
    """Dataset that loads aligned features."""
    
    def __init__(
        self,
        data_dir: Path,
        split: str,
        audio_model: str = "wav2vec2",
        max_length: int = 200,
        use_augmented: bool = True,
        num_augmentations: int = None,
        input_dir_suffix: str = None,  # Input directory suffix (e.g., "_speed2x")
    ):
        """
        Args:
            data_dir: Data directory
            split: train or val
            audio_model: Audio model name (wav2vec2 fixed)
            max_length: Maximum sequence length
            use_augmented: Whether to use augmented audios (train only)
            num_augmentations: Number of augmentations (None = load all)
            input_dir_suffix: Input directory suffix (e.g., "_speed2x"). If set, load from {class_name}{suffix}.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length
        self.use_augmented = use_augmented
        self.num_augmentations = num_augmentations
        self.input_dir_suffix = input_dir_suffix
        
        # Fixed filename suffixes
        # Text features are extracted from BERT
        self.text_suffix = "_bert"
        self.audio_suffix = "_wav2vec2"
        
        # Load samples
        all_samples = self._load_samples()
        
        # Filter out samples with zero length
        self.samples = []
        for sample in all_samples:
            # Check valid lengths (if lengths_mask file exists)
            if sample["lengths_mask_path"] is not None and sample["lengths_mask_path"].exists():
                try:
                    lengths_mask = torch.load(sample["lengths_mask_path"])
                    audio_valid = lengths_mask["audio_valid_length"]
                    text_valid = lengths_mask["text_valid_length"]
                    # Skip zero-length samples
                    if audio_valid == 0 or text_valid == 0:
                        print(f"Warning: Skipping sample {sample['uid']} with zero length (audio_valid_length={audio_valid}, text_valid_length={text_valid})")
                        continue
                except Exception as e:
                    print(f"Warning: Failed to check lengths for {sample['uid']}: {e}, keeping sample")
            
            self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples from {split} (filtered {len(all_samples) - len(self.samples)} zero-length samples)")
        sys.stdout.flush()
    
    def _extract_subject_id(self, uid: str) -> str:
        """
        Extract subject ID from UID.
        Example: "adrso025_subject" -> "adrso025"
                 "adrso025_subject_aug_0" -> "adrso025"
        """
        # Remove suffixes like "_subject" and "_aug_*"
        subject_id = uid.replace("_subject", "").replace("_aug_0", "").replace("_aug_1", "").replace("_aug_2", "")
        # Remove any remaining "_aug_*" pattern
        if "_aug_" in subject_id:
            subject_id = subject_id.split("_aug_")[0]
        return subject_id
    
    def _load_samples(self) -> List[Dict]:
        """Load dataset samples."""
        samples = []
        
        # Process each class (original audios)
        for class_name in ["ad", "cn"]:
            # Determine input directory (use input_dir_suffix if provided)
            if self.input_dir_suffix == "_preprocessed":
                class_dir = self.data_dir / self.split / class_name
            elif self.input_dir_suffix:
                class_dir = self.data_dir / self.split / f"{class_name}{self.input_dir_suffix}"
            else:
                class_dir = self.data_dir / self.split / class_name
            
            if not class_dir.exists():
                continue
            
            # Find text feature files
            text_pattern = f"*{self.text_suffix}.pt"
            text_files = list(class_dir.glob(text_pattern))
            
            for text_file in text_files:
                uid = text_file.stem.replace(self.text_suffix, "")
                
                # Find corresponding audio feature file
                audio_file = class_dir / f"{uid}{self.text_suffix}{self.audio_suffix}.pt"
                
                if audio_file.exists():
                    # Path to valid-lengths and masks
                    lengths_mask_file = class_dir / f"{uid}{self.text_suffix}_lengths_mask.pt"
                    subject_id = self._extract_subject_id(uid)
                    samples.append({
                        "text_features_path": text_file,
                        "audio_features_path": audio_file,
                        "lengths_mask_path": lengths_mask_file if lengths_mask_file.exists() else None,
                        "label": 0 if class_name == "ad" else 1,
                        "uid": uid,
                        "subject_id": subject_id,
                        "is_original": True,
                    })
                else:
                    print(f"Warning: Audio features not found for {uid}")
        
        # Load augmented audios (train only when enabled)
        if self.split == "train" and self.use_augmented:
            for class_name in ["ad", "cn"]:
                # If num_augmentations is set, use the corresponding aug folder
                if self.num_augmentations is not None:
                    augmented_dir = self.data_dir / self.split / f"aug{self.num_augmentations}" / f"{class_name}_augmented"
                else:
                    augmented_dir = self.data_dir / self.split / "aug" / f"{class_name}_augmented"
                if not augmented_dir.exists():
                    continue
                
                # If num_augmentations is set, load only the specified augmentation indices
                if self.num_augmentations is not None:
                    # Only load files for _aug_0, _aug_1, ..., _aug_{num_augmentations-1}
                    for aug_idx in range(self.num_augmentations):
                        # Find text feature files for this augmentation index
                        text_pattern = f"*_aug_{aug_idx}{self.text_suffix}.pt"
                        text_files = list(augmented_dir.glob(text_pattern))
                        
                        for text_file in text_files:
                            uid = text_file.stem.replace(self.text_suffix, "")
                            
                            # Find corresponding audio feature file
                            audio_file = augmented_dir / f"{uid}{self.text_suffix}{self.audio_suffix}.pt"
                            
                            if audio_file.exists():
                                # Path to valid-lengths and masks
                                lengths_mask_file = augmented_dir / f"{uid}{self.text_suffix}_lengths_mask.pt"
                                subject_id = self._extract_subject_id(uid)
                                samples.append({
                                    "text_features_path": text_file,
                                    "audio_features_path": audio_file,
                                    "lengths_mask_path": lengths_mask_file if lengths_mask_file.exists() else None,
                                    "label": 0 if class_name == "ad" else 1,
                                    "uid": uid,
                                    "subject_id": subject_id,
                                    "is_original": False,
                                })
                else:
                    # If num_augmentations is not set, load all augmented files (backward compatible)
                    # Find text feature files
                    text_pattern = f"*{self.text_suffix}.pt"
                    text_files = list(augmented_dir.glob(text_pattern))
                    
                    for text_file in text_files:
                        uid = text_file.stem.replace(self.text_suffix, "")
                        
                        # Find corresponding audio feature file
                        audio_file = augmented_dir / f"{uid}{self.text_suffix}{self.audio_suffix}.pt"
                        
                        if audio_file.exists():
                            # Path to valid-lengths and masks
                            lengths_mask_file = augmented_dir / f"{uid}{self.text_suffix}_lengths_mask.pt"
                            subject_id = self._extract_subject_id(uid)
                            samples.append({
                                "text_features_path": text_file,
                                "audio_features_path": audio_file,
                                "lengths_mask_path": lengths_mask_file if lengths_mask_file.exists() else None,
                                "label": 0 if class_name == "ad" else 1,
                                "uid": uid,
                                "subject_id": subject_id,
                                "is_original": False,
                            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load features
        text_features = torch.load(sample["text_features_path"])
        audio_features = torch.load(sample["audio_features_path"])
        
        # Load valid lengths and masks (compute if not saved)
        if sample["lengths_mask_path"] is not None:
            lengths_mask = torch.load(sample["lengths_mask_path"])
            audio_valid = lengths_mask["audio_valid_length"]
            text_valid = lengths_mask["text_valid_length"]
            audio_mask = lengths_mask["audio_mask"]
            text_mask = lengths_mask["text_mask"]
        else:
            # Fallback: compute masks/lengths when lengths_mask is missing
            print(f"Warning: Lengths mask not found for {sample['uid']}, calculating...")
            audio_valid = (audio_features.abs().max(dim=1)[0] >= 1e-6).sum().item()
            text_valid = (text_features.norm(dim=1) > 1e-6).sum().item()
            # Create masks
            audio_mask = torch.zeros(self.max_length, dtype=torch.bool)
            audio_mask[audio_valid:] = True
            text_mask = torch.zeros(self.max_length, dtype=torch.bool)
            text_mask[text_valid:] = True
        
        # Error if valid length is zero (should have been filtered during feature extraction)
        if audio_valid == 0 or text_valid == 0:
            raise ValueError(f"Invalid sample {sample['uid']}: audio_valid_length={audio_valid}, text_valid_length={text_valid}. This should have been filtered during feature extraction.")
        
        return {
            "audio_features": audio_features,
            "text_features": text_features,
            "audio_length": audio_valid,
            "text_length": text_valid,
            "audio_mask": audio_mask,
            "text_mask": text_mask,
            "label": sample["label"],
            "uid": sample["uid"],
            "subject_id": sample.get("subject_id", ""),
            "is_original": sample.get("is_original", True),
        }


def create_collate_fn(max_length: int = 200):
    """Create a collate function (fixed max_length)."""
    def collate_fn(batch):
        """Collate function."""
        audio_features_list = [item["audio_features"] for item in batch]
        text_features_list = [item["text_features"] for item in batch]
        audio_lengths = torch.tensor([item["audio_length"] for item in batch])
        text_lengths = torch.tensor([item["text_length"] for item in batch])
        
        # Pad masks to max_length and then stack
        padded_audio_masks = []
        for item in batch:
            audio_mask = item["audio_mask"]
            if audio_mask.shape[0] < max_length:
                # Pad with True (True = padding positions)
                padding = torch.ones(max_length - audio_mask.shape[0], dtype=torch.bool)
                padded_audio_masks.append(torch.cat([audio_mask, padding], dim=0))
            elif audio_mask.shape[0] > max_length:
                # Truncate (usually not needed)
                padded_audio_masks.append(audio_mask[:max_length])
            else:
                # Already at max_length
                padded_audio_masks.append(audio_mask)
        audio_masks = torch.stack(padded_audio_masks)  # (batch, max_length)
        
        padded_text_masks = []
        for item in batch:
            text_mask = item["text_mask"]
            if text_mask.shape[0] < max_length:
                # Pad with True (True = padding positions)
                padding = torch.ones(max_length - text_mask.shape[0], dtype=torch.bool)
                padded_text_masks.append(torch.cat([text_mask, padding], dim=0))
            elif text_mask.shape[0] > max_length:
                # Truncate (usually not needed)
                padded_text_masks.append(text_mask[:max_length])
            else:
                # Already at max_length
                padded_text_masks.append(text_mask)
        text_masks = torch.stack(padded_text_masks)  # (batch, max_length)
        
        labels = torch.tensor([item["label"] for item in batch])
        subject_ids = [item.get("subject_id", "") for item in batch]
        is_original = torch.tensor([item.get("is_original", True) for item in batch], dtype=torch.bool)
        
        # Features should already be padded to max_length, but verify/adjust
        # Pad/truncate audio features
        padded_audio = []
        for f in audio_features_list:
            if f.shape[0] < max_length:
                # Pad if needed
                padding = torch.zeros(max_length - f.shape[0], f.shape[1])
                padded_audio.append(torch.cat([f, padding], dim=0))
            elif f.shape[0] > max_length:
                # Truncate if needed (usually not needed)
                padded_audio.append(f[:max_length])
            else:
                # Already at max_length
                padded_audio.append(f)
        audio_features = torch.stack(padded_audio)  # (batch, max_length, audio_dim)
        
        # Pad/truncate text features
        padded_text = []
        for f in text_features_list:
            if f.shape[0] < max_length:
                # Pad if needed
                padding = torch.zeros(max_length - f.shape[0], f.shape[1])
                padded_text.append(torch.cat([f, padding], dim=0))
            elif f.shape[0] > max_length:
                # Truncate if needed (usually not needed)
                padded_text.append(f[:max_length])
            else:
                # Already at max_length
                padded_text.append(f)
        text_features = torch.stack(padded_text)  # (batch, max_length, text_dim)
        
        # Masks are already padded to max_length (processed above)
        
        return {
            "audio_features": audio_features,
            "text_features": text_features,
            "audio_lengths": audio_lengths,
            "text_lengths": text_lengths,
            "audio_masks": audio_masks,
            "text_masks": text_masks,
            "labels": labels,
            "subject_ids": subject_ids,
            "is_original": is_original,
        }
    return collate_fn


def train_model(
    data_dir: str = "data",
    batch_size: int = 32,
    num_epochs: int = 200,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    early_stopping: bool = True,
    early_stopping_patience: int = 20,
    device: str = None,
    audio_model: str = "wav2vec2",
    output_json: str = None,
    d_model: int = 768,
    nhead: int = 12,
    num_layers: int = 1,
    dropout: float = 0.3,
    dim_feedforward: int = 3072,
    pooling: str = "mean",
    max_length: int = 200,
    audio_dim: int = None,
    text_dim: int = None,
    num_classes: int = 1,
    hidden_mlp_size: int = 256,
    use_augmented: bool = True,
    num_augmentations: int = None,
    seed: int = 42,
    mode: str = "multimodal",  # "audio", "text", "multimodal"
    pos_weight: float = None,  # Positive class weight (auto-computed if None)
    input_dir_suffix: str = None,  # (Backward compatible) shared train/val input directory suffix
    train_input_suffix: str = None,  # Train split suffix (overrides input_dir_suffix)
    val_input_suffix: str = None,  # Val split suffix (overrides input_dir_suffix)
):
    """
    Train the model.
    
    Args:
        data_dir: Data directory
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        early_stopping: Whether to use early stopping
        early_stopping_patience: Early stopping patience
        device: Device (auto-select if None)
        audio_model: Audio model name (wav2vec2 fixed)
        d_model: Model dimension (hidden_size)
        nhead: Number of attention heads
        num_layers: Number of Transformer layers
        dropout: Dropout rate
        dim_feedforward: Feed-forward dimension (intermediate_size)
        pooling: Pooling strategy (mean, cls, attn, hierarchical)
        max_length: Max sequence length
        audio_dim: Audio feature dimension (auto-detect if None)
        text_dim: Text feature dimension (auto-detect if None)
        num_classes: Number of classes
        hidden_mlp_size: Hidden MLP size of classifier
        use_augmented: Whether to use augmented audios (train only)
        num_augmentations: Number of augmentations (None = load all)
        seed: Random seed
        mode: Mode ("audio", "text", "multimodal")
        pos_weight: Positive class weight (auto-computed if None)
        input_dir_suffix: (Backward compatible) shared train/val input directory suffix
        train_input_suffix: Train split suffix (overrides input_dir_suffix)
        val_input_suffix: Val split suffix (overrides input_dir_suffix)
    """
    # Seed setup (for reproducibility)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Validate mode
    if mode not in ["audio", "text", "multimodal"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'audio', 'text', or 'multimodal'")
    
    print(f"Using device: {device}")
    print(f"Mode: {mode}")
    print(f"Random seed: {seed}")
    sys.stdout.flush()
    device = torch.device(device)
    
    # Decide final suffix values (individual settings take precedence)
    train_suffix = train_input_suffix if train_input_suffix is not None else input_dir_suffix
    val_suffix = val_input_suffix if val_input_suffix is not None else input_dir_suffix
    
    # Create datasets
    train_dataset = AlignedFeatureDataset(
        Path(data_dir),
        split="train",
        audio_model=audio_model,
        max_length=max_length,
        use_augmented=use_augmented,
        num_augmentations=num_augmentations,
        input_dir_suffix=train_suffix
    )
    val_dataset = AlignedFeatureDataset(
        Path(data_dir),
        split="val",
        audio_model=audio_model,
        max_length=max_length,
        use_augmented=False,  # validation is always False
        num_augmentations=None,  # validation never uses augmented data
        input_dir_suffix=val_suffix
    )
    
    if len(train_dataset) == 0:
        print("Error: No training samples found. Please run extract_aligned_features.py first.")
        sys.stdout.flush()
        return None
    
    # Auto-detect feature dimensions
    sample = train_dataset[0]
    if mode in ["audio", "multimodal"]:
        if audio_dim is None:
            audio_dim = sample["audio_features"].shape[1]
        print(f"Audio feature dimension: {audio_dim}")
    if mode in ["text", "multimodal"]:
        if text_dim is None:
            text_dim = sample["text_features"].shape[1]
        print(f"Text feature dimension: {text_dim}")
    sys.stdout.flush()
    
    # Create collate_fn (fixed max_length)
    collate_fn = create_collate_fn(max_length=max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create model
    model_kwargs = {
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "num_classes": num_classes,
        "dropout": dropout,
        "dim_feedforward": dim_feedforward,
        "pooling": pooling,
        "hidden_mlp_size": hidden_mlp_size,
        "mode": mode,
    }
    
    if mode in ["audio", "multimodal"]:
        model_kwargs["audio_dim"] = audio_dim
    if mode in ["text", "multimodal"]:
        model_kwargs["text_dim"] = text_dim
    
    model = CrossAttentionTransformer(**model_kwargs).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    sys.stdout.flush()
    
    # Compute class distribution and set pos_weight
    if num_classes == 1 and pos_weight is None:
        # Compute class distribution from training data
        train_labels = [sample["label"] for sample in train_dataset.samples]
        num_neg = sum(1 for label in train_labels if label == 0)
        num_pos = sum(1 for label in train_labels if label == 1)
        
        if num_pos > 0 and num_neg > 0:
            # pos_weight = num_neg / num_pos (larger when negatives are more frequent)
            pos_weight_value = num_neg / num_pos
            print(f"Class distribution: Negative={num_neg}, Positive={num_pos}")
            print(f"Computed pos_weight: {pos_weight_value:.4f}")
        else:
            pos_weight_value = 1.0
            print(f"Warning: Cannot compute pos_weight (num_neg={num_neg}, num_pos={num_pos}), using 1.0")
    elif num_classes == 1:
        pos_weight_value = pos_weight
        print(f"Using specified pos_weight: {pos_weight_value:.4f}")
    else:
        pos_weight_value = None
    
    sys.stdout.flush()
    
    # Loss function and optimizer
    if num_classes == 1:
        # For binary classification: BCEWithLogitsLoss
        if pos_weight_value is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    # Increase patience and set min_lr to avoid LR getting too small
    # Use mode='min' since we monitor validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=2, verbose=True, min_lr=1e-8
    )
    
    # Training loop
    best_val_loss = float('inf')  # Early stopping based on validation loss
    best_val_acc = 0.0
    best_val_precision = 0.0
    best_val_recall = 0.0
    best_val_f1 = 0.0
    best_val_precision_ad = 0.0
    best_val_recall_ad = 0.0
    best_val_f1_ad = 0.0
    best_val_precision_cn = 0.0
    best_val_recall_cn = 0.0
    best_val_f1_cn = 0.0
    best_train_acc = 0.0
    best_train_precision = 0.0
    best_train_recall = 0.0
    best_train_f1 = 0.0
    best_train_precision_ad = 0.0
    best_train_recall_ad = 0.0
    best_train_f1_ad = 0.0
    best_train_precision_cn = 0.0
    best_train_recall_cn = 0.0
    best_train_f1_cn = 0.0
    early_stopping_counter = 0
    
    # Record loss history (for plotting)
    train_loss_history = []
    val_loss_history = []
    epochs_completed = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_classification_loss = 0.0
        train_correct = 0
        train_total = 0
        train_all_preds = []
        train_all_labels = []

        for batch in train_loader:
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            # Pass required features depending on mode
            model_inputs = {}
            if mode in ["audio", "multimodal"]:
                model_inputs["audio_features"] = batch["audio_features"].to(device)
                model_inputs["audio_lengths"] = batch["audio_lengths"].to(device)
                model_inputs["audio_mask"] = batch["audio_masks"].to(device)
            if mode in ["text", "multimodal"]:
                model_inputs["text_features"] = batch["text_features"].to(device)
                model_inputs["text_lengths"] = batch["text_lengths"].to(device)
                model_inputs["text_mask"] = batch["text_masks"].to(device)
            
            model_outputs = model(**model_inputs)

            outputs = model_outputs
            
            # Classification loss
            if num_classes == 1:
                # For binary classification, convert labels to float
                labels_float = labels.float().unsqueeze(1)
                classification_loss = criterion(outputs, labels_float)
            else:
                classification_loss = criterion(outputs, labels)
            
            total_loss = classification_loss
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += total_loss.item()
            train_classification_loss += classification_loss.item()
            if num_classes == 1:
                predicted = (torch.sigmoid(outputs.data) > 0.5).squeeze(1).long()
            else:
                _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Save predictions and labels to compute precision/recall/F1
            train_all_preds.extend(predicted.cpu().numpy())
            train_all_labels.extend(labels.cpu().numpy())
        
        train_acc = 100 * train_correct / train_total
        # Compute precision/recall/F1
        train_precision = precision_score(train_all_labels, train_all_preds, zero_division=0.0) * 100
        train_recall = recall_score(train_all_labels, train_all_preds, zero_division=0.0) * 100
        train_f1 = f1_score(train_all_labels, train_all_preds, zero_division=0.0) * 100
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_all_preds = []
        val_all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"].to(device)
                
                # Pass required features depending on mode
                model_inputs = {}
                if mode in ["audio", "multimodal"]:
                    model_inputs["audio_features"] = batch["audio_features"].to(device)
                    model_inputs["audio_lengths"] = batch["audio_lengths"].to(device)
                    model_inputs["audio_mask"] = batch["audio_masks"].to(device)
                if mode in ["text", "multimodal"]:
                    model_inputs["text_features"] = batch["text_features"].to(device)
                    model_inputs["text_lengths"] = batch["text_lengths"].to(device)
                    model_inputs["text_mask"] = batch["text_masks"].to(device)
                
                model_outputs = model(**model_inputs)
                outputs = model_outputs
                
                if num_classes == 1:
                    labels_float = labels.float().unsqueeze(1)
                    loss = criterion(outputs, labels_float)
                else:
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                if num_classes == 1:
                    predicted = (torch.sigmoid(outputs.data) > 0.5).squeeze(1).long()
                else:
                    _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Save predictions and labels to compute precision/recall/F1
                val_all_preds.extend(predicted.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())
        
        val_acc = 100 * val_correct / val_total
        # Compute precision/recall/F1
        val_precision = precision_score(val_all_labels, val_all_preds, zero_division=0.0) * 100
        val_recall = recall_score(val_all_labels, val_all_preds, zero_division=0.0) * 100
        val_f1 = f1_score(val_all_labels, val_all_preds, zero_division=0.0) * 100
        
        # Per-class precision/recall/F1 (binary classification only)
        if num_classes == 1:
            train_per_class = _precision_recall_f1_per_class(train_all_labels, train_all_preds)
            val_per_class = _precision_recall_f1_per_class(val_all_labels, val_all_preds)
        
        # Compute average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Append to loss history
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        epochs_completed.append(epoch + 1)
        
        # Step LR scheduler using validation loss
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}", end="")
        
        print(f"    Train Acc: {train_acc:.2f}%, Precision: {train_precision:.2f}%, Recall: {train_recall:.2f}%, F1: {train_f1:.2f}%")
        
        
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"    Val Acc: {val_acc:.2f}%, Precision: {val_precision:.2f}%, Recall: {val_recall:.2f}%, F1: {val_f1:.2f}%")
        if num_classes == 1:
            print(f"    Val AD  P/R/F1: {val_per_class['precision_ad']:.2f}% / {val_per_class['recall_ad']:.2f}% / {val_per_class['f1_ad']:.2f}%  |  CN  P/R/F1: {val_per_class['precision_cn']:.2f}% / {val_per_class['recall_cn']:.2f}% / {val_per_class['f1_cn']:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        sys.stdout.flush()
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_val_precision = val_precision
            best_val_recall = val_recall
            best_val_f1 = val_f1
            best_train_acc = train_acc
            best_train_precision = train_precision
            best_train_recall = train_recall
            best_train_f1 = train_f1
            if num_classes == 1:
                best_val_precision_ad = val_per_class["precision_ad"]
                best_val_recall_ad = val_per_class["recall_ad"]
                best_val_f1_ad = val_per_class["f1_ad"]
                best_val_precision_cn = val_per_class["precision_cn"]
                best_val_recall_cn = val_per_class["recall_cn"]
                best_val_f1_cn = val_per_class["f1_cn"]
                best_train_precision_ad = train_per_class["precision_ad"]
                best_train_recall_ad = train_per_class["recall_ad"]
                best_train_f1_ad = train_per_class["f1_ad"]
                best_train_precision_cn = train_per_class["precision_cn"]
                best_train_recall_cn = train_per_class["recall_cn"]
                best_train_f1_cn = train_per_class["f1_cn"]
            early_stopping_counter = 0
            # Include augmented/non_augmented in model name
            if use_augmented and num_augmentations is not None:
                aug_suffix = f"augmented_{num_augmentations}"
            else:
                aug_suffix = "non_augmented"
            
            model_name = f"best_model_{mode}_bert_{audio_model}_{aug_suffix}.pth"
            
            # Determine model save location (output_json dir if provided; otherwise current dir)
            if output_json:
                model_dir = Path(output_json).parent
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / model_name
            else:
                model_path = Path(model_name)
            
            torch.save(model.state_dict(), str(model_path))
            print(f"  Saved best model (Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%) to {model_path.absolute()}")
            sys.stdout.flush()
        else:
            early_stopping_counter += 1
        
        # Early stopping check
        if early_stopping and early_stopping_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs (patience: {early_stopping_patience})")
            sys.stdout.flush()
            break
    
    print(f"\nTraining completed! Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.2f}%")
    sys.stdout.flush()
    
    # Create and save the loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_completed, train_loss_history, label='Train Loss', marker='o', markersize=3)
    plt.plot(epochs_completed, val_loss_history, label='Val Loss', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot (output_json dir if provided; otherwise current dir)
    if output_json:
        plot_dir = Path(output_json).parent
        plot_dir.mkdir(parents=True, exist_ok=True)
        # Generate plot name by removing extension from model name
        plot_name = f"loss_plot_{mode}_bert_{audio_model}"
        if use_augmented and num_augmentations is not None:
            plot_name += f"_augmented_{num_augmentations}"
        else:
            plot_name += "_non_augmented"
        plot_path = plot_dir / f"{plot_name}.png"
    else:
        plot_name = f"loss_plot_{mode}_bert_{audio_model}"
        if use_augmented and num_augmentations is not None:
            plot_name += f"_augmented_{num_augmentations}"
        else:
            plot_name += "_non_augmented"
        plot_path = Path(f"{plot_name}.png")
    
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {plot_path.absolute()}")
    sys.stdout.flush()
    
    # Return results
    results = {
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_precision": best_val_precision,
        "best_val_recall": best_val_recall,
        "best_val_f1": best_val_f1,
        "best_train_acc": best_train_acc,
        "best_train_precision": best_train_precision,
        "best_train_recall": best_train_recall,
        "best_train_f1": best_train_f1
    }
    if num_classes == 1:
        results.update({
            "best_val_precision_ad": best_val_precision_ad,
            "best_val_recall_ad": best_val_recall_ad,
            "best_val_f1_ad": best_val_f1_ad,
            "best_val_precision_cn": best_val_precision_cn,
            "best_val_recall_cn": best_val_recall_cn,
            "best_val_f1_cn": best_val_f1_cn,
            "best_train_precision_ad": best_train_precision_ad,
            "best_train_recall_ad": best_train_recall_ad,
            "best_train_f1_ad": best_train_f1_ad,
            "best_train_precision_cn": best_train_precision_cn,
            "best_train_recall_cn": best_train_recall_cn,
            "best_train_f1_cn": best_train_f1_cn,
        })
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train audio classification model with aligned features")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--early_stopping", action="store_true", default=True, help="Enable early stopping")
    parser.add_argument("--no_early_stopping", dest="early_stopping", action="store_false", help="Disable early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--audio_model", type=str, default="wav2vec2",
                       help="Audio model (wav2vec2 fixed)")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension (hidden_size)")
    parser.add_argument("--nhead", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--dim_feedforward", type=int, default=3072, help="Feedforward dimension (intermediate_size)")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls", "attn", "hierarchical"],
                       help="Pooling strategy")
    parser.add_argument("--max_length", type=int, default=200, help="Max sequence length")
    parser.add_argument("--audio_dim", type=int, default=None, help="Audio feature dimension (auto-detect if None)")
    parser.add_argument("--text_dim", type=int, default=None, help="Text feature dimension (auto-detect if None)")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of classes")
    parser.add_argument("--hidden_mlp_size", type=int, default=256, help="Classifier MLP hidden size")
    parser.add_argument("--use_augmented", action="store_true", default=True, help="Use augmented audios (train only)")
    parser.add_argument("--no_augmented", dest="use_augmented", action="store_false", help="Skip augmented audios")
    parser.add_argument("--num_augmentations", type=int, default=None, help="Number of augmentations to use (e.g., 1 for _aug_0 only, 2 for _aug_0 and _aug_1). If not specified, use all augmentations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="multimodal", choices=["audio", "text", "multimodal"],
                       help="Training mode: 'audio' (audio only), 'text' (text only), or 'multimodal' (both)")
    parser.add_argument("--pos_weight", type=float, default=None,
                       help="Positive class weight for BCEWithLogitsLoss (auto-computed if None)")
    parser.add_argument("--input_dir_suffix", type=str, default=None,
                       help="Input directory suffix (e.g., '_speed2x'). If specified, loads features from {class_name}{suffix} directory.")
    parser.add_argument("--train_input_suffix", type=str, default=None,
                       help="Train split input directory suffix (overrides --input_dir_suffix for train if specified).")
    parser.add_argument("--val_input_suffix", type=str, default=None,
                       help="Val split input directory suffix (overrides --input_dir_suffix for val if specified).")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save results as JSON file")
    
    args = parser.parse_args()
    
    results = train_model(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        device=args.device,
        audio_model=args.audio_model,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        pooling=args.pooling,
        max_length=args.max_length,
        audio_dim=args.audio_dim,
        text_dim=args.text_dim,
        num_classes=args.num_classes,
        hidden_mlp_size=args.hidden_mlp_size,
        use_augmented=args.use_augmented,
        num_augmentations=args.num_augmentations,
        seed=args.seed,
        mode=args.mode,
        pos_weight=args.pos_weight,
        input_dir_suffix=args.input_dir_suffix,
        train_input_suffix=args.train_input_suffix,
        val_input_suffix=args.val_input_suffix,
        output_json=args.output_json,
    )
    
    # Save results to JSON 
    if args.output_json and results is not None:
        import json
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")
        sys.stdout.flush()
