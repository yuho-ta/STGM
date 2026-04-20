"""
Script to run predictions on test data and compute accuracy by comparing with ground-truth labels
"""
import os
import sys
import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
try:
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
except ImportError:
    # Fallback when scikit-learn is not installed
    def accuracy_score(y_true, y_pred):
        return sum(y_true == y_pred) / len(y_true)
    
    def precision_score(y_true, y_pred, zero_division=0.0):
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        if tp + fp == 0:
            return zero_division
        return tp / (tp + fp)
    
    def recall_score(y_true, y_pred, zero_division=0.0):
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        if tp + fn == 0:
            return zero_division
        return tp / (tp + fn)
    
    def f1_score(y_true, y_pred, zero_division=0.0):
        precision = precision_score(y_true, y_pred, zero_division=zero_division)
        recall = recall_score(y_true, y_pred, zero_division=zero_division)
        if precision + recall == 0:
            return zero_division
        return 2 * (precision * recall) / (precision + recall)
    
    def confusion_matrix(y_true, y_pred):
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        return [[tn, fp], [fn, tp]]

try:
    from .model import CrossAttentionTransformer
    from .train_classifier import AlignedFeatureDataset, create_collate_fn, _precision_recall_f1_per_class
except ImportError:
    # Fallback when running directly
    from model import CrossAttentionTransformer
    from train_classifier import AlignedFeatureDataset, create_collate_fn, _precision_recall_f1_per_class


def load_ground_truth_labels(csv_path: str) -> Dict[str, int]:
    """
    Load ground-truth labels CSV
    
    Args:
        csv_path: Path to CSV file (includes ID and Dx columns)
    
    Returns:
        A dictionary {ID: label} (label: 0=ProbableAD, 1=Control)
    """
    labels = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Remove quotes from the ID
            id = row['ID'].strip('"')
            dx = row['Dx'].strip('"')
            
            # Convert Dx to label
            if dx == "ProbableAD":
                label = 0  # ad
            elif dx == "Control":
                label = 1  # cn
            else:
                print(f"Warning: Unknown Dx value '{dx}' for ID '{id}', skipping...")
                continue
            
            labels[id] = label
    
    return labels


def predict_test_data(
    test_data_dir: str,
    model_path: str,
    ground_truth_csv: str,
    batch_size: int = 32,
    device: str = None,
    audio_model: str = "wav2vec2",
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
    mode: str = "multimodal",
    output_json: str = None,
) -> Dict:
    """
    Run predictions on test data and compute accuracy by comparing with ground-truth labels
    
    Args:
        test_data_dir: Test data directory (includes test/test/ad and test/test/cn)
        model_path: Path to the model checkpoint
        ground_truth_csv: Path to ground-truth labels CSV file
        batch_size: Batch size
        device: Device (auto-select if None)
        audio_model: Audio model name (wav2vec2 fixed)
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of Transformer layers
        dropout: Dropout rate
        dim_feedforward: Feedforward layer dimension
        pooling: Pooling strategy
        max_length: Max sequence length
        audio_dim: Audio feature dimension (auto-detect if None)
        text_dim: Text feature dimension (auto-detect if None)
        num_classes: Number of classification classes
        hidden_mlp_size: Hidden MLP size of the classifier
        mode: Mode ("audio", "text", "multimodal")
        output_json: Path to the JSON file to save results
    Returns:
        A dictionary of prediction results and accuracy metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device = torch.device(device)
    print(f"Using device: {device}")
    print(f"Mode: {mode}")
    sys.stdout.flush()
    
    # Load ground-truth labels
    print(f"Loading ground truth labels from {ground_truth_csv}")
    ground_truth = load_ground_truth_labels(ground_truth_csv)
    print(f"Loaded {len(ground_truth)} ground truth labels")
    sys.stdout.flush()
    
    # Create the test dataset
    test_dataset = AlignedFeatureDataset(
        Path(test_data_dir),
        split="test",
        audio_model=audio_model,
        max_length=max_length,
        use_augmented=False,  # Do not use augmented data for test data
        num_augmentations=None,
        input_dir_suffix=None
    )
    
    if len(test_dataset) == 0:
        print("Error: No test samples found. Please run preprocessing and feature extraction first.")
        sys.stdout.flush()
        return None
    
    # Auto-detect feature dimensions
    sample = test_dataset[0]
    if mode in ["audio", "multimodal"]:
        if audio_dim is None:
            audio_dim = sample["audio_features"].shape[1]
        print(f"Audio feature dimension: {audio_dim}")
    if mode in ["text", "multimodal"]:
        if text_dim is None:
            text_dim = sample["text_features"].shape[1]
        print(f"Text feature dimension: {text_dim}")
    sys.stdout.flush()
    
    # Create collate_fn
    collate_fn = create_collate_fn(max_length=max_length)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create the model
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
    
    # Load model weights
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully")
    sys.stdout.flush()
    
    # Run predictions
    all_preds = []
    all_labels = []
    all_uids = []
    all_probs = []
    
    print("Running predictions...")
    sys.stdout.flush()
    
    with torch.no_grad():
        for batch in test_loader:
            # Pass only the required features depending on the mode
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
                # For binary classification
                probs = torch.sigmoid(outputs.data).squeeze(1).cpu().numpy()
                predicted = (probs > 0.5).astype(int)
            else:
                probs = torch.softmax(outputs.data, dim=1).cpu().numpy()
                predicted = probs.argmax(axis=1)
            
            # Get UID (for each sample in the batch)
            batch_start_idx = len(all_uids)
            for i in range(len(predicted)):
                sample_idx = batch_start_idx + i
                if sample_idx < len(test_dataset.samples):
                    uid = test_dataset.samples[sample_idx]["uid"]
                    
                    # Get the ground-truth label (extract ID from UID; ID comes from the filename)
                    # UID is usually the base name of the filename (without extension)
                    # Example: adrsdt1_processed_bert -> adrsdt1
                    # Or: adrsdt1_subject -> adrsdt1
                    # Or: adrsdt1 -> adrsdt1
                    # Remove suffixes like _processed and _aug_*
                    uid_base = uid.replace("_processed", "").replace("_bert", "").replace("_wav2vec2", "").replace("_subject", "")
                    # Remove the _aug_* pattern
                    if "_aug_" in uid_base:
                        uid_base = uid_base.split("_aug_")[0]
                    
                    # Get the ground-truth label
                    if uid_base in ground_truth:
                        true_label = ground_truth[uid_base]
                        all_labels.append(true_label)
                    else:
                        # If UID is not found, warn and skip
                        print(f"Warning: Ground truth label not found for UID '{uid_base}' (original UID: '{uid}'), skipping...")
                        continue
                    
                    all_preds.append(predicted[i])
                    all_probs.append(float(probs[i] if num_classes == 1 else probs[i][predicted[i]]))
                    all_uids.append(uid_base)
    
    if len(all_labels) == 0:
        print("Error: No matching ground truth labels found. Please check UID matching.")
        sys.stdout.flush()
        return None
    
    # Compute accuracy metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, zero_division=0.0) * 100
    recall = recall_score(all_labels, all_preds, zero_division=0.0) * 100
    f1 = f1_score(all_labels, all_preds, zero_division=0.0) * 100
    cm = confusion_matrix(all_labels, all_preds)
    
    # Per-class precision/recall/F1 (only for binary classification)
    per_class = None
    if num_classes == 1:
        per_class = _precision_recall_f1_per_class(all_labels, all_preds)
    
    # Display results
    print(f"\n{'='*60}")
    print("Test Results")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_labels)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1-Score: {f1:.2f}%")
    if per_class is not None:
        print(f"  AD  P/R/F1: {per_class['precision_ad']:.2f}% / {per_class['recall_ad']:.2f}% / {per_class['f1_ad']:.2f}%")
        print(f"  CN  P/R/F1: {per_class['precision_cn']:.2f}% / {per_class['recall_cn']:.2f}% / {per_class['f1_cn']:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  True Negative (TN): {cm[0][0]}")
    print(f"  False Positive (FP): {cm[0][1]}")
    print(f"  False Negative (FN): {cm[1][0]}")
    print(f"  True Positive (TP): {cm[1][1]}")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    # Summarize results into a dictionary
    results = {
        "total_samples": len(all_labels),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1])
        },
        "predictions": [
            {
                "uid": uid,
                "predicted": int(pred),
                "true_label": int(label),
                "probability": prob
            }
            for uid, pred, label, prob in zip(all_uids, all_preds, all_labels, all_probs)
        ]
    }
    if per_class is not None:
        results["precision_ad"] = per_class["precision_ad"]
        results["recall_ad"] = per_class["recall_ad"]
        results["f1_ad"] = per_class["f1_ad"]
        results["precision_cn"] = per_class["precision_cn"]
        results["recall_cn"] = per_class["recall_cn"]
        results["f1_cn"] = per_class["f1_cn"]
    
    # Save results to a JSON file
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")
        sys.stdout.flush()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict on test data and calculate accuracy")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Test data directory (should contain test/test/ad and test/test/cn)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--ground_truth_csv", type=str, required=True, help="Path to ground truth CSV file (ID, Dx columns)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--audio_model", type=str, default="wav2vec2", help="Audio model (wav2vec2 fixed)")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension (hidden_size)")
    parser.add_argument("--nhead", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--dim_feedforward", type=int, default=3072, help="Feedforward dimension (intermediate_size)")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls", "attn"], help="Pooling strategy")
    parser.add_argument("--max_length", type=int, default=200, help="Max sequence length")
    parser.add_argument("--audio_dim", type=int, default=None, help="Audio feature dimension (auto-detect if None)")
    parser.add_argument("--text_dim", type=int, default=None, help="Text feature dimension (auto-detect if None)")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of classes")
    parser.add_argument("--hidden_mlp_size", type=int, default=256, help="Classifier MLP hidden size")
    parser.add_argument("--mode", type=str, default="multimodal", choices=["audio", "text", "multimodal"],
                       help="Training mode: 'audio' (audio only), 'text' (text only), or 'multimodal' (both)")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save results as JSON file")
    
    args = parser.parse_args()
    
    results = predict_test_data(
        test_data_dir=args.test_data_dir,
        model_path=args.model_path,
        ground_truth_csv=args.ground_truth_csv,
        batch_size=args.batch_size,
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
        mode=args.mode,
        output_json=args.output_json
    )
    
    if results is None:
        sys.exit(1)

