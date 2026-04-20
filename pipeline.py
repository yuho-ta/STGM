"""
Fish Speech TTS Augmentation: Main script to run the full pipeline
"""
import os
import sys
import subprocess
from pathlib import Path
import argparse
import logging
from datetime import datetime
import json
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Logging configuration
try:
    from utils.logger import setup_logger
except ImportError:
    # Fallback when utils.logger cannot be imported
    import logging
    def setup_logger(name, log_dir=None, log_level=logging.DEBUG, log_to_file=True, log_to_console=True):
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        logger.handlers.clear()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        if log_to_file:
            if log_dir is None:
                log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{name}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Log file: {log_file.absolute()}")
        
        return logger

def get_python_executable():
    """Get the path of the Python executable to use"""
    # Use it if the venv environment exists
    venv_python = Path("venv") / "Scripts" / "python.exe" if os.name == 'nt' else Path("venv") / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    # Use the system Python
    return sys.executable


def load_training_setting(mode: str, logger: logging.Logger = None) -> dict:
    """
    Load training hyperparameters from a JSON file based on the model mode.

    Expected files (under ./training_settings/):
    - MULTIMODAL_SETTING.json
    - AUDIO_SETTING.json
    - TEXT_SETTING.json
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    defaults = {
        "train_args": {
            "epochs": 400,
            "lr": 0.0001,
            "weight_decay": 0.05,
            "early_stopping_patience": 10,
        },
        "model_common_args": {
            "batch_size": 44,
            "d_model": 768,
            "nhead": 12,
            "num_layers": 1,
            "dropout": 0.2,
            "dim_feedforward": 3072,
            "pooling": "mean",  # mean, cls, attn, hierarchical
            "num_classes": 1,
            "hidden_mlp_size": 256,
        },
    }

    mode_key = (mode or "").strip().lower()
    filename_map = {
        "multimodal": "MULTIMODAL_SETTING.json",
        "audio": "AUDIO_SETTING.json",
        "text": "TEXT_SETTING.json",
    }
    filename = filename_map.get(mode_key)
    if not filename:
        logger.warning(f"No training setting file mapping found for mode={mode!r}. Using defaults.")
        return defaults

    settings_path = Path(__file__).parent / "training_settings" / filename
    if not settings_path.exists():
        logger.warning(f"Training setting file not found: {settings_path}. Using defaults.")
        return defaults

    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load training setting JSON: {settings_path}. Error: {e}. Using defaults.")
        return defaults

    # Minimal validation with fallbacks
    train_args = loaded.get("train_args", {}) if isinstance(loaded, dict) else {}
    model_common_args = loaded.get("model_common_args", {}) if isinstance(loaded, dict) else {}

    def _get(d: dict, key: str, default_value):
        return d.get(key, default_value) if isinstance(d, dict) else default_value

    return {
        "train_args": {
            "epochs": _get(train_args, "epochs", defaults["train_args"]["epochs"]),
            "lr": _get(train_args, "lr", defaults["train_args"]["lr"]),
            "weight_decay": _get(train_args, "weight_decay", defaults["train_args"]["weight_decay"]),
            "early_stopping_patience": _get(train_args, "early_stopping_patience", defaults["train_args"]["early_stopping_patience"]),
        },
        "model_common_args": {
            "batch_size": _get(model_common_args, "batch_size", defaults["model_common_args"]["batch_size"]),
            "d_model": _get(model_common_args, "d_model", defaults["model_common_args"]["d_model"]),
            "nhead": _get(model_common_args, "nhead", defaults["model_common_args"]["nhead"]),
            "num_layers": _get(model_common_args, "num_layers", defaults["model_common_args"]["num_layers"]),
            "dropout": _get(model_common_args, "dropout", defaults["model_common_args"]["dropout"]),
            "dim_feedforward": _get(model_common_args, "dim_feedforward", defaults["model_common_args"]["dim_feedforward"]),
            "pooling": _get(model_common_args, "pooling", defaults["model_common_args"]["pooling"]),
            "num_classes": _get(model_common_args, "num_classes", defaults["model_common_args"]["num_classes"]),
            "hidden_mlp_size": _get(model_common_args, "hidden_mlp_size", defaults["model_common_args"]["hidden_mlp_size"]),
        },
    }

def aggregate_cv_results(fold_results: list, n_folds: int, logger: logging.Logger = None):
    """
    Aggregate 5-fold CV results and compute mean and standard deviation
    
    Args:
        fold_results: List of results for each fold
        n_folds: Number of folds
        logger: Logger
    
    Returns:
        Dictionary of aggregated results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if len(fold_results) == 0:
        return {}
    
    # Collect values for each metric (including per-class ad/cn)
    metrics = {
        "best_val_acc": [],
        "best_val_precision": [],
        "best_val_recall": [],
        "best_val_f1": [],
        "best_val_precision_ad": [],
        "best_val_recall_ad": [],
        "best_val_f1_ad": [],
        "best_val_precision_cn": [],
        "best_val_recall_cn": [],
        "best_val_f1_cn": [],
        "best_train_acc": [],
        "best_train_precision": [],
        "best_train_recall": [],
        "best_train_f1": [],
        "best_train_precision_ad": [],
        "best_train_recall_ad": [],
        "best_train_f1_ad": [],
        "best_train_precision_cn": [],
        "best_train_recall_cn": [],
        "best_train_f1_cn": [],
    }
    
    for fold_result in fold_results:
        for metric_name in metrics.keys():
            if metric_name in fold_result:
                metrics[metric_name].append(fold_result[metric_name])
    
    # Compute mean and standard deviation
    aggregated = {
        "n_folds": n_folds,
        "completed_folds": len(fold_results),
        "folds": fold_results
    }
    
    for metric_name, values in metrics.items():
        if len(values) > 0:
            if HAS_NUMPY:
                aggregated[f"{metric_name}_mean"] = float(np.mean(values))
                aggregated[f"{metric_name}_std"] = float(np.std(values))
            else:
                # Fallback when NumPy is not available
                mean_val = sum(values) / len(values)
                variance = sum((x - mean_val) ** 2 for x in values) / len(values)
                std_val = variance ** 0.5
                aggregated[f"{metric_name}_mean"] = float(mean_val)
                aggregated[f"{metric_name}_std"] = float(std_val)
            aggregated[f"{metric_name}_values"] = [float(v) for v in values]
    
    # Output to log
    logger.info(f"\n{'='*60}")
    logger.info("Cross-Validation Results Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Completed folds: {len(fold_results)}/{n_folds}")
    logger.info(f"\nValidation Metrics (mean ± std):")
    if "best_val_acc_mean" in aggregated:
        logger.info(f"  Accuracy: {aggregated['best_val_acc_mean']:.2f}% ± {aggregated['best_val_acc_std']:.2f}%")
        logger.info(f"  Precision: {aggregated['best_val_precision_mean']:.2f}% ± {aggregated['best_val_precision_std']:.2f}%")
        logger.info(f"  Recall: {aggregated['best_val_recall_mean']:.2f}% ± {aggregated['best_val_recall_std']:.2f}%")
        logger.info(f"  F1-Score: {aggregated['best_val_f1_mean']:.2f}% ± {aggregated['best_val_f1_std']:.2f}%")
        if "best_val_precision_ad_mean" in aggregated:
            logger.info(f"  AD  Precision/Recall/F1: {aggregated['best_val_precision_ad_mean']:.2f}% / {aggregated['best_val_recall_ad_mean']:.2f}% / {aggregated['best_val_f1_ad_mean']:.2f}% (± {aggregated['best_val_precision_ad_std']:.2f}% / {aggregated['best_val_recall_ad_std']:.2f}% / {aggregated['best_val_f1_ad_std']:.2f}%)")
            logger.info(f"  CN  Precision/Recall/F1: {aggregated['best_val_precision_cn_mean']:.2f}% / {aggregated['best_val_recall_cn_mean']:.2f}% / {aggregated['best_val_f1_cn_mean']:.2f}% (± {aggregated['best_val_precision_cn_std']:.2f}% / {aggregated['best_val_recall_cn_std']:.2f}% / {aggregated['best_val_f1_cn_std']:.2f}%)")
    logger.info(f"\nTraining Metrics (mean ± std):")
    if "best_train_acc_mean" in aggregated:
        logger.info(f"  Accuracy: {aggregated['best_train_acc_mean']:.2f}% ± {aggregated['best_train_acc_std']:.2f}%")
        logger.info(f"  Precision: {aggregated['best_train_precision_mean']:.2f}% ± {aggregated['best_train_precision_std']:.2f}%")
        logger.info(f"  Recall: {aggregated['best_train_recall_mean']:.2f}% ± {aggregated['best_train_recall_std']:.2f}%")
        logger.info(f"  F1-Score: {aggregated['best_train_f1_mean']:.2f}% ± {aggregated['best_train_f1_std']:.2f}%")
        if "best_train_precision_ad_mean" in aggregated:
            logger.info(f"  AD  Precision/Recall/F1: {aggregated['best_train_precision_ad_mean']:.2f}% / {aggregated['best_train_recall_ad_mean']:.2f}% / {aggregated['best_train_f1_ad_mean']:.2f}% (± {aggregated['best_train_precision_ad_std']:.2f}% / {aggregated['best_train_recall_ad_std']:.2f}% / {aggregated['best_train_f1_ad_std']:.2f}%)")
            logger.info(f"  CN  Precision/Recall/F1: {aggregated['best_train_precision_cn_mean']:.2f}% / {aggregated['best_train_recall_cn_mean']:.2f}% / {aggregated['best_train_f1_cn_mean']:.2f}% (± {aggregated['best_train_precision_cn_std']:.2f}% / {aggregated['best_train_recall_cn_std']:.2f}% / {aggregated['best_train_f1_cn_std']:.2f}%)")
    logger.info(f"{'='*60}")
    
    return aggregated

def run_step(script_name: str, args: list, description: str, logger: logging.Logger = None):
    """Run a pipeline step"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Step: {description}")
    logger.info(f"{'='*60}")
    
    python_exe = get_python_executable()
    # Add the -u flag to disable buffering
    cmd = [python_exe, "-u", script_name] + args
    logger.info(f"Running: {' '.join(cmd)}")
    
    # Stream output in real time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,  # Line buffering
        universal_newlines=True,
    )
    
    for line in process.stdout:
        line = line.rstrip()
        if not line:
            continue
        logger.info(line)
    
    process.wait()
    
    if process.returncode != 0:
        logger.error(f"Error: {script_name} failed with return code {process.returncode}")
        return False
    
    logger.info(f"Step completed successfully: {description}")
    return True

def prepare_test_data(
    test_audio_dir: str,
    test_data_dir: str,
    ground_truth_csv: str,
    logger: logging.Logger = None
):
    """
    Arrange test data into the appropriate directory structure
    
    Args:
        test_audio_dir: Test audio directory (test-dist/audio)
        test_data_dir: Output test data directory (test/test)
        ground_truth_csv: Path to the ground-truth labels CSV file
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    import shutil
    import csv
    
    # Load ground-truth labels
    ground_truth = {}
    with open(ground_truth_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id = row['ID'].strip('"')
            dx = row['Dx'].strip('"')
            ground_truth[id] = dx
    
    # Create the test data directory
    test_data_path = Path(test_data_dir)
    test_data_path.mkdir(parents=True, exist_ok=True)
    
    # Create ad and cn directories
    ad_dir = test_data_path / "test" / "ad"
    cn_dir = test_data_path / "test" / "cn"
    ad_dir.mkdir(parents=True, exist_ok=True)
    cn_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy test audio files
    test_audio_path = Path(test_audio_dir)
    audio_files = list(test_audio_path.glob("*.wav")) + list(test_audio_path.glob("*.mp3"))
    
    copied_ad = 0
    copied_cn = 0
    
    for audio_file in audio_files:
        # Extract the ID from the filename (excluding the extension)
        file_id = audio_file.stem
        
        # Check ground-truth labels
        if file_id in ground_truth:
            dx = ground_truth[file_id]
            if dx == "ProbableAD":
                dst_file = ad_dir / audio_file.name
                if not dst_file.exists():
                    shutil.copy2(audio_file, dst_file)
                    copied_ad += 1
            elif dx == "Control":
                dst_file = cn_dir / audio_file.name
                if not dst_file.exists():
                    shutil.copy2(audio_file, dst_file)
                    copied_cn += 1
        else:
            logger.warning(f"Ground truth label not found for {file_id}, skipping...")
    
    logger.info(f"Prepared test data: {copied_ad} AD files, {copied_cn} CN files")
    return True

def run_all_data_preprocessing(
    audio_dir: str = "audio",
    data_dir: str = "data",
    whisper_model: str = "base",
    language: str = "en",
    hf_token: str = None,
    max_files: int = None,
    skip_steps: list = None,
    auto_skip: bool = True,
    logger: logging.Logger = None
):
    """
    Run preprocess and feature extraction only once for all data
    
    Args:
        audio_dir: Original audio directory
        data_dir: Output data directory (temporary directory for all data)
        skip_steps: List of steps to skip
        auto_skip: Whether to automatically skip completed steps
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if skip_steps is None:
        skip_steps = []
    
    # Create a temporary directory structure for all data
    # Place all data under data_all/train/ad and data_all/train/cn
    all_data_dir = Path(data_dir).parent / "data_all"
    all_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all data from the original audio_dir (treated as train)
    for split in ["train"]:  # Treat all data as train
        for class_name in ["ad", "cn"]:
            src_dir = Path(audio_dir) / class_name
            dst_dir = all_data_dir / split / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            if src_dir.exists():
                # Copy files (skip if already exists)
                import shutil
                # Allow original data as .mp3 or .wav
                audio_files = list(src_dir.glob("*.mp3")) + list(src_dir.glob("*.wav"))
                for audio_file in audio_files:
                    dst_file = dst_dir / audio_file.name
                    if not dst_file.exists():
                        shutil.copy2(audio_file, dst_file)
                logger.info(f"Copied {len(audio_files)} files from {src_dir} to {dst_dir}")
    
    # Step 1: Preprocess all data
    if "preprocess" not in skip_steps:
        logger.info(f"\n{'='*60}")
        logger.info("Step: Preprocess all data (once for all folds)")
        logger.info(f"{'='*60}")
        
        for class_name in ["ad", "cn"]:
            cmd_args = [
                "--data_dir", str(all_data_dir),
                "--split", "train",
                "--class_name", class_name,
                "--language", language,
                "--whisper_model", whisper_model
            ]
            if hf_token:
                cmd_args.extend(["--hf_token", hf_token])
            if max_files:
                cmd_args.extend(["--max_files", str(max_files)])
            
            success = run_step(
                "feature_extraction/preprocess_audio.py",
                cmd_args,
                f"Preprocess all audios for {class_name.upper()}",
                logger=logger
            )
            if not success:
                logger.error(f"Failed to preprocess {class_name} (all data)")
                return False
    
    # Step 2: Feature extraction for all data
    if "extract_features" not in skip_steps:
        logger.info(f"\n{'='*60}")
        logger.info("Step: Extract features for all data (once for all folds)")
        logger.info(f"{'='*60}")

        cmd_args = [
            "--data_dir", str(all_data_dir),
            "--split", "train",
            "--whisper_model", whisper_model,
            "--language", language,
            "--no_augmented",  # Augmented data does not exist yet for all data
            "--log_level", "INFO"
        ]
        
        success = run_step(
            "feature_extraction/extract_aligned_features.py",
            cmd_args,
            "Extract aligned features for all data",
            logger=logger
        )
        if not success:
            logger.error("Failed to extract features for all data")
            return False
    
    logger.info(f"\n{'='*60}")
    logger.info("All data preprocessing completed!")
    logger.info(f"Output directory: {all_data_dir}")
    logger.info(f"{'='*60}")
    
    return True

def run_full_pipeline(
    audio_dir: str = "audio",
    data_dir: str = "data",
    val_ratio: float = 0.2,
    whisper_model: str = "base",
    language: str = "en",
    num_references: int = 5,
    num_augmentations: int = 3,
    fish_speech_path: str = None,
    random_seed: int = 42,
    skip_steps: list = None,
    use_augmented: bool = True,
    log_dir: str = "logs",
    auto_skip: bool = True,
    hf_token: str = None,
    max_files: int = None,
    n_folds: int = None,
    mode: str = "multimodal",  # "audio", "text", "multimodal"
    test_audio_dir: str = None,  # Test audio directory (test-dist/audio)
    test_ground_truth_csv: str = None,  # Ground-truth labels CSV (test-dist/task1.csv)
):
    """
    Run the full pipeline (supports 5-fold CV)
    
    Steps (5-fold CV):
    1. Preprocess all data (run once)
    2. Feature extraction for all data (run once)
    3. Split the dataset into 5 folds
    4. For each fold:
       - augmentation (train only)
       - augmented data feature extraction (train only)
       - training
    
    Steps (standard run):
    1. Split the dataset into train/val
    2. Audio preprocessing (speaker separation / noise removal)
    3. Prepare reference audio
    4. Text shuffle (each class)
    5. Generate TTS with Fish Speech (each class)
    6. Re-ASR with Whisper (each class)
    7. Feature extraction (wav2vec2 + BERT; Whisper for ASR/timestamps)
    8. Train the classification model
    
    Args:
        n_folds: Number of folds (if specified, run 5-fold CV and run the pipeline for each fold)
    """
    # Set up the logger (INFO level)
    logger = setup_logger("pipeline", log_dir=Path(log_dir), log_level=logging.INFO)
    
    logger.info("=" * 60)
    logger.info("TTS Augmentation Pipeline Started")
    logger.info("=" * 60)
    logger.info(f"Audio directory: {audio_dir}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Validation ratio: {val_ratio}")
    logger.info(f"Whisper model: {whisper_model}")
    logger.info(f"Language: {language}")
    logger.info(f"Number of references: {num_references}")
    logger.info(f"Number of augmentations: {num_augmentations}")
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"Use augmented: {use_augmented}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Skip steps: {skip_steps if skip_steps else 'None'}")
    logger.info(f"Auto skip completed steps: {auto_skip}")
    if n_folds:
        logger.info(f"Cross-validation: {n_folds}-fold CV")
    if max_files:
        logger.info(f"[TEST MODE] Maximum files per class: {max_files}")
    if test_audio_dir and test_ground_truth_csv:
        logger.info(f"Test data: {test_audio_dir}")
        logger.info(f"Test ground truth: {test_ground_truth_csv}")
    
    if skip_steps is None:
        skip_steps = []
    
    # For 5-fold CV
    if n_folds is not None and n_folds > 1:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting {n_folds}-fold Cross-Validation Pipeline")
        logger.info(f"{'='*60}")
        
        # Step 0: Preprocess and feature extraction for all data (run once)
        logger.info(f"\n{'='*60}")
        logger.info("Phase 1: Preprocess and extract features for all data (once)")
        logger.info(f"{'='*60}")
        
        all_data_success = run_all_data_preprocessing(
            audio_dir=audio_dir,
            data_dir=data_dir,
            whisper_model=whisper_model,
            language=language,
            hf_token=hf_token,
            max_files=max_files,
            skip_steps=skip_steps,
            auto_skip=auto_skip,
            logger=logger
        )
        
        if not all_data_success:
            logger.error("Failed to preprocess all data. Aborting pipeline.")
            return False
        
        # Step 1: Split into 5 folds (including preprocessed files)
        logger.info(f"\n{'='*60}")
        logger.info("Phase 2: Split data into folds (including preprocessed files)")
        logger.info(f"{'='*60}")
        
        if "split" not in skip_steps:
            all_data_dir = Path(data_dir).parent / "data_all"
            python_exe = get_python_executable()
            cmd = [
                python_exe, "-u", "split_dataset.py",
                "--audio_dir", audio_dir,
                "--output_dir", data_dir,
                "--val_ratio", str(val_ratio),
                "--seed", str(random_seed),
                "--n_folds", str(n_folds),
                "--preprocessed_dir", str(all_data_dir)
            ]
            logger.info(f"Running: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if process.returncode != 0:
                logger.error(f"Failed to create fold splits: {process.stderr}")
                return False
            
            logger.info("Fold splits created successfully (including preprocessed files)!")
        
        # Step 2: For each fold, run from augmentation onward
        logger.info(f"\n{'='*60}")
        logger.info("Phase 3: Run augmentation and training for each fold")
        logger.info(f"{'='*60}")
        
        all_folds_success = True
        fold_results = []  # Save results for each fold
        
        for fold_idx in range(n_folds):
            logger.info(f"\n{'='*60}")
            logger.info(f"Fold {fold_idx + 1}/{n_folds}")
            logger.info(f"{'='*60}")
            
            # Per-fold data directory
            fold_data_dir = str(Path(data_dir).parent / str(fold_idx) / "data")
            
            # Run the pipeline for each fold (skip preprocess and feature extraction for original data)
            # However, augmented data feature extraction must run for each fold
            fold_skip_steps = skip_steps.copy()
            if "preprocess" not in fold_skip_steps:
                fold_skip_steps.append("preprocess")
            # Do not skip extract_features (augmented data feature extraction is required)
            
            fold_result = run_single_fold_pipeline(
                audio_dir=audio_dir,
                data_dir=fold_data_dir,
                fold_idx=fold_idx,
                val_ratio=val_ratio,
                whisper_model=whisper_model,
                language=language,
                num_references=num_references,
                num_augmentations=num_augmentations,
                fish_speech_path=fish_speech_path,
                random_seed=random_seed,
                skip_steps=fold_skip_steps,
                use_augmented=use_augmented,
                auto_skip=auto_skip,
                hf_token=hf_token,
                max_files=max_files,
                mode=mode,
                test_audio_dir=test_audio_dir,
                test_ground_truth_csv=test_ground_truth_csv,
                logger=logger
            )
            
            if fold_result is None:
                logger.error(f"Fold {fold_idx + 1} failed!")
                all_folds_success = False
            else:
                logger.info(f"Fold {fold_idx + 1} completed successfully!")
                # Save results (when it's a dictionary)
                if isinstance(fold_result, dict):
                    fold_result["fold"] = fold_idx + 1
                    fold_results.append(fold_result)
        
        logger.info(f"\n{'='*60}")
        if all_folds_success:
            logger.info(f"All {n_folds} folds completed successfully!")
        else:
            logger.warning(f"Some folds failed. Please check the logs.")
        logger.info(f"{'='*60}")
        
        # Aggregate the 5-fold CV results and save them to JSON
        if len(fold_results) > 0:
            cv_results = aggregate_cv_results(fold_results, n_folds, logger=logger)
            aug_suffix = f"augmented_{num_augmentations}" if use_augmented else "non_augmented"
            output_json_path = Path(data_dir).parent / f"cv_results_{mode}_{aug_suffix}.json"
            try:
                import json
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(cv_results, f, indent=2, ensure_ascii=False)
                logger.info(f"\n{'='*60}")
                logger.info(f"Cross-validation results saved to: {output_json_path}")
                logger.info(f"{'='*60}")
            except Exception as e:
                logger.error(f"Failed to save CV results: {e}")
        
        return all_folds_success
    
    else:
        # Standard single-fold run
        return run_single_fold_pipeline(
            audio_dir=audio_dir,
            data_dir=data_dir,
            fold_idx=None,
            val_ratio=val_ratio,
            whisper_model=whisper_model,
            language=language,
            num_references=num_references,
            num_augmentations=num_augmentations,
            fish_speech_path=fish_speech_path,
            random_seed=random_seed,
            skip_steps=skip_steps,
            use_augmented=use_augmented,
            auto_skip=auto_skip,
            hf_token=hf_token,
            max_files=max_files,
            mode=mode,
            test_audio_dir=test_audio_dir,
            test_ground_truth_csv=test_ground_truth_csv,
            logger=logger
        )

def run_single_fold_pipeline(
    audio_dir: str = "audio",
    data_dir: str = "data",
    fold_idx: int = None,
    val_ratio: float = 0.2,
    whisper_model: str = "base",
    language: str = "en",
    num_references: int = 5,
    num_augmentations: int = 3,
    fish_speech_path: str = None,
    random_seed: int = 42,
    skip_steps: list = None,
    use_augmented: bool = True,
    auto_skip: bool = True,
    hf_token: str = None,
    max_files: int = None,
    mode: str = "multimodal",  # "audio", "text", "multimodal"
    test_audio_dir: str = None,  # Test audio directory (test-dist/audio)
    test_ground_truth_csv: str = None,  # Ground-truth labels CSV (test-dist/task1.csv)
    logger: logging.Logger = None
):
    """
    Run the pipeline for a single fold (internal function)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    success = True
    
    # Step 1: Split the dataset
    # If fold_idx is specified (5-fold CV), it's already split so skip
    if "split" not in skip_steps and fold_idx is None:
        cmd_args = [
            "--audio_dir", audio_dir,
            "--output_dir", data_dir,
            "--val_ratio", str(val_ratio),
            "--seed", str(random_seed)
        ]
        
        success = run_step(
            "split_dataset.py",
            cmd_args,
            "Split dataset into train/val",
            logger=logger
        )
        if not success:
            logger.error("Pipeline failed at step: Split dataset")
            return False
    elif fold_idx is not None:
        logger.info(f"Fold {fold_idx + 1}: Split step skipped (already completed in fold creation)")
    
    # Step 2: Audio preprocessing (speaker separation / noise removal)
    if "preprocess" not in skip_steps:
        for split in ["train", "val"]:
            for class_name in ["ad", "cn"]:
                cmd_args = [
                    "--data_dir", data_dir,
                    "--split", split,
                    "--class_name", class_name,
                    "--language", language,
                    "--whisper_model", whisper_model
                ]
                if hf_token:
                    cmd_args.extend(["--hf_token", hf_token])
                if max_files:
                    cmd_args.extend(["--max_files", str(max_files)])
                
                success = run_step(
                    "feature_extraction/preprocess_audio.py",
                    cmd_args,
                    f"Preprocess audios for {class_name.upper()} ({split})",
                    logger=logger
                )
                if not success:
                    logger.warning(f"Failed to preprocess {class_name} ({split}), continuing...")
    
    # Step 3: Prepare reference audio (skip - use original audio files as reference)
    # Note: In fish_tts_generate.py, the original audio files are used as reference audio, so this step is unnecessary
    logger.info("Skipping reference audio preparation - using original audio files as reference")
    
    # Step 4: Text shuffle (each class, train only)
    # Skip if use_augmented is False (since augmentation is not used)
    if "shuffle" not in skip_steps and use_augmented:
        for class_name in ["ad", "cn"]:
            success = run_step(
                "augmentation/text_shuffle.py",
                [
                    "--data_dir", data_dir,
                    "--split", "train",
                    "--class_name", class_name,
                    "--whisper_model", whisper_model,
                    "--language", language,
                    "--seed", str(random_seed),
                    "--num_augmentations", str(num_augmentations)
                ],
                f"Shuffle texts for {class_name.upper()} (train)",
                logger=logger
            )
            if not success:
                logger.warning(f"Failed to shuffle texts for {class_name}, continuing...")
    elif not use_augmented:
        logger.info("Step 'shuffle' skipped (use_augmented=False)")
    
    # Step 5: Generate TTS with Fish Speech (each class, train only)
    # Skip if use_augmented is False (since augmentation is not used)
    if "tts" not in skip_steps and use_augmented:
        for class_name in ["ad", "cn"]:
            cmd_args = [
                "--data_dir", data_dir,
                "--split", "train",
                "--class_name", class_name,
                "--num_augmentations", str(num_augmentations),
                "--seed", str(random_seed)
            ]
            if fish_speech_path:
                cmd_args.extend(["--fish_speech_path", fish_speech_path])
            
            success = run_step(
                "augmentation/fish_tts_generate.py",
                cmd_args,
                f"Generate TTS with Fish Speech for {class_name.upper()}",
                logger=logger
            )
            if not success:
                logger.warning(f"Failed to generate TTS for {class_name}, continuing...")
    elif not use_augmented:
        logger.info("Step 'tts' skipped (use_augmented=False)")
    
    # Step 6: Re-ASR with Whisper (each class, train only)
    # Skip if use_augmented is False (since augmentation is not used)
    if "reasr" not in skip_steps and use_augmented:
        for class_name in ["ad", "cn"]:
            success = run_step(
                "augmentation/whisper_reasr.py",
                [
                    "--data_dir", data_dir,
                    "--split", "train",
                    "--class_name", class_name,
                    "--whisper_model", whisper_model,
                    "--language", language,
                    "--num_augmentations", str(num_augmentations)
                ],
                f"Re-ASR augmented audios for {class_name.upper()}",
                logger=logger
            )
            if not success:
                logger.warning(f"Failed to re-ASR for {class_name}, continuing...")
    elif not use_augmented:
        logger.info("Step 'reasr' skipped (use_augmented=False)")
    
    # Step 7: Feature extraction (wav2vec2 + BERT; Whisper for ASR/timestamps)
    if "extract_features" not in skip_steps:
        # In 5-fold CV, features for the original data are already copied,
        # so extract only features for augmented data
        if fold_idx is not None:
            # For 5-fold CV: extract only augmented data features (train only)
            if use_augmented:
                cmd_args = [
                    "--data_dir", data_dir,
                    "--split", "train",
                    "--whisper_model", whisper_model,
                    "--language", language,
                    "--use_augmented",  # Process augmented data only (original data is already copied)
                    "--num_augmentations", str(num_augmentations),
                    "--log_level", "INFO"
                ]
                
                success = run_step(
                    "feature_extraction/extract_aligned_features.py",
                    cmd_args,
                    "Extract aligned features for augmented data (train)",
                    logger=logger
                )
                if not success:
                    logger.warning("Failed to extract features for augmented data, continuing...")
            
            # Skip val features because they are already copied
            logger.info("Step 'extract_features' for val skipped (already copied from all data)")
        else:
            # Normal run: process both original and augmented data
            for split in ["train", "val"]:
                cmd_args = [
                    "--data_dir", data_dir,
                    "--split", split,
                    "--whisper_model", whisper_model,
                    "--language", language,
                    "--num_augmentations", str(num_augmentations),
                    "--log_level", "INFO"
                ]
                # If using augmented audio (train only)
                if use_augmented and split == "train":
                    cmd_args.append("--use_augmented")
                else:
                    cmd_args.append("--no_augmented")
                
                success = run_step(
                    "feature_extraction/extract_aligned_features.py",
                    cmd_args,
                    f"Extract aligned features for {split}",
                    logger=logger
                )
                if not success:
                    logger.warning(f"Failed to extract features for {split}, continuing...")
    
    # Step 8: Train the classification model
    if "train" not in skip_steps:
        # Set the results JSON path (only when fold_idx is specified)
        results_json_path = None
        if fold_idx is not None:
            aug_suffix = f"augmented_{num_augmentations}" if use_augmented else "non_augmented"
            results_json_path = Path(data_dir).parent / f"fold_{fold_idx}_results_{mode}_{aug_suffix}.json"
        
        # Load training hyperparameters from mode-specific settings
        setting = load_training_setting(mode, logger=logger)
        train_args = setting["train_args"]
        model_common_args_values = setting["model_common_args"]

        model_common_args = []
        for k, v in model_common_args_values.items():
            model_common_args.extend([f"--{k}", str(v)])
        model_common_args.extend(["--mode", mode])

        cmd_args = [
            "--data_dir", data_dir,
            "--epochs", str(train_args["epochs"]),
            "--lr", str(train_args["lr"]),
            "--weight_decay", str(train_args["weight_decay"]),
            "--early_stopping_patience", str(train_args["early_stopping_patience"]),
            "--seed", str(random_seed),
            "--num_augmentations", str(num_augmentations),
        ] + model_common_args
        # If using augmented audio
        if use_augmented:
            cmd_args.append("--use_augmented")
        else:
            cmd_args.append("--no_augmented")
        
        # Add the results JSON path (when fold_idx is specified)
        if results_json_path:
            cmd_args.extend(["--output_json", str(results_json_path)])
        
        # early_stopping defaults to True; add --no_early_stopping only if you want to disable it
        
        success = run_step(
            "training/train_classifier.py",
            cmd_args,
            "Train classification model",
            logger=logger
        )
        if not success:
            logger.warning("Model training failed")
    
    # Step 9: Predict on test data (only when test_audio_dir and test_ground_truth_csv are provided)
    if "test" not in skip_steps and test_audio_dir and test_ground_truth_csv:
        logger.info(f"\n{'='*60}")
        logger.info("Step: Test data prediction")
        logger.info(f"{'='*60}")
        
        # Prepare the test data directory
        # Use the test-dist directory as test_data_dir because test-dist/audio is used directly
        test_audio_path = Path(test_audio_dir)
        test_data_dir = str(test_audio_path.parent)  # Parent of test-dist/audio = test-dist
        
        # Split and place into test-dist/test/ad and test-dist/test/cn
        if auto_skip and (Path(test_data_dir) / "test" / "ad").exists() and (Path(test_data_dir) / "test" / "cn").exists():
            logger.info("Test data already prepared, skipping preparation...")
        else:
            prepare_test_data(
                test_audio_dir=test_audio_dir,
                test_data_dir=test_data_dir,
                ground_truth_csv=test_ground_truth_csv,
                logger=logger
            )
        
        # Preprocess test data
        for class_name in ["ad", "cn"]:
            cmd_args = [
                "--data_dir", test_data_dir,
                "--split", "test",
                "--class_name", class_name,
                "--language", language,
                "--whisper_model", whisper_model
            ]
            if hf_token:
                cmd_args.extend(["--hf_token", hf_token])
            
            success = run_step(
                "feature_extraction/preprocess_audio.py",
                cmd_args,
                f"Preprocess test audios for {class_name.upper()}",
                logger=logger
            )
            if not success:
                logger.warning(f"Failed to preprocess test {class_name}, continuing...")
        
        # Extract features for test data
        cmd_args = [
            "--data_dir", test_data_dir,
            "--split", "test",
            "--whisper_model", whisper_model,
            "--language", language,
            "--no_augmented",  # Do not use augmented data for test data
            "--log_level", "INFO"
        ]
        
        success = run_step(
            "feature_extraction/extract_aligned_features.py",
            cmd_args,
            "Extract aligned features for test data",
            logger=logger
        )
        if not success:
            logger.warning("Failed to extract features for test data, continuing...")
        
        # Get model path (latest best_model)
        # Use a model name that includes augmented/non_augmented info
        aug_suffix = f"augmented_{num_augmentations}" if use_augmented else "non_augmented"
        model_name = f"best_model_{mode}_bert_wav2vec2_{aug_suffix}.pth"
        
        # Determine where to save the model (same directory as per-fold results JSON)
        if fold_idx is not None:
            # Build the per-fold results JSON path and load the model from that directory
            results_json_path = Path(data_dir).parent / f"fold_{fold_idx}_results_{mode}_{aug_suffix}.json"
            model_path = results_json_path.parent / model_name
        else:
            model_path = Path(model_name)
        
        # If not found in the current directory, check parent directories and models directories
        if not model_path.exists():
            # Check the per-fold model directory
            if fold_idx is not None:
                # Check the per-fold models directory
                fold_model_dir = Path(data_dir).parent / "models"
                if fold_model_dir.exists():
                    model_files = list(fold_model_dir.glob(f"best_model_{mode}_bert_wav2vec2_{aug_suffix}.pth"))
                    if model_files:
                        model_path = model_files[0]
                    else:
                        # Check the current directory
                        model_path = Path.cwd() / model_name
                        if not model_path.exists():
                            logger.warning(f"Model file not found, skipping test prediction")
                            model_path = None
                else:
                    # Check the current directory
                    model_path = Path.cwd() / model_name
                    if not model_path.exists():
                        logger.warning(f"Model file not found, skipping test prediction")
                        model_path = None
            else:
                # In normal runs, check the current directory
                model_path = Path.cwd() / model_name
                if not model_path.exists():
                    # Check data_dir/models directory
                    models_dir = Path(data_dir) / "models"
                    if models_dir.exists():
                        model_files = list(models_dir.glob(f"best_model_{mode}_bert_wav2vec2_{aug_suffix}.pth"))
                        if model_files:
                            model_path = model_files[0]
                        else:
                            logger.warning(f"Model file not found: {model_name}, skipping test prediction")
                            model_path = None
                    else:
                        logger.warning(f"Model file not found: {model_name}, skipping test prediction")
                        model_path = None
        
        if model_path and model_path.exists():
            # Predict on test data (aug_suffix is already defined)
            test_results_json = Path(data_dir).parent / f"test_results_{mode}_{aug_suffix}.json"
            if fold_idx is not None:
                test_results_json = Path(data_dir).parent / f"fold_{fold_idx}_test_results_{mode}_{aug_suffix}.json"
            
            cmd_args = [
                "--test_data_dir", test_data_dir,
                "--model_path", str(model_path),
                "--ground_truth_csv", test_ground_truth_csv,
                "--output_json", str(test_results_json)
            ] + model_common_args
            success = run_step(
                "training/test_predict.py",
                cmd_args,
                "Predict on test data and calculate accuracy",
                logger=logger
            )
            if not success:
                logger.warning("Test prediction failed")
            else:
                logger.info(f"Test results saved to {test_results_json}")
        else:
            logger.warning("Model file not found, skipping test prediction")
    elif test_audio_dir or test_ground_truth_csv:
        logger.info("Test data prediction skipped (both --test_audio_dir and --test_ground_truth_csv must be specified)")
    
    logger.info(f"\n{'='*60}")
    if fold_idx is not None:
        logger.info(f"Fold {fold_idx + 1} pipeline completed!")
    else:
        logger.info("Pipeline completed!")
    logger.info(f"{'='*60}")
    
    # If fold_idx is specified, load and return the results JSON file
    if fold_idx is not None:
        aug_suffix = f"augmented_{num_augmentations}" if use_augmented else "non_augmented"
        results_json_path = Path(data_dir).parent / f"fold_{fold_idx}_results_{mode}_{aug_suffix}.json"
        if results_json_path.exists():
            try:
                with open(results_json_path, 'r', encoding='utf-8') as f:
                    fold_result = json.load(f)
                    return fold_result
            except Exception as e:
                logger.warning(f"Failed to read results JSON: {e}")
                return None
        else:
            # If the JSON file does not exist (e.g., train step was skipped)
            logger.warning(f"Results JSON file not found: {results_json_path}")
            return None
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full TTS augmentation pipeline")
    parser.add_argument("--audio_dir", type=str, default="audio", help="Input audio directory")
    parser.add_argument("--data_dir", type=str, default="data", help="Output data directory")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--whisper_model", type=str, default="base", help="Whisper model size")
    parser.add_argument("--language", type=str, default="en", help="Audio language (en/ja)")
    parser.add_argument("--num_references", type=int, default=5, help="Number of reference audios per class")
    parser.add_argument("--num_augmentations", type=int, default=2, help="Number of augmentations per audio")
    parser.add_argument("--fish_speech_path", type=str, default=None, help="Path to Fish Speech")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip", nargs="+", default=[], help="Steps to skip (split/preprocess/references/shuffle/tts/reasr/extract_features/train/test)")
    parser.add_argument("--use_augmented", action="store_true", default=True, help="Use augmented audios for training")
    parser.add_argument("--no_augmented", dest="use_augmented", action="store_false", help="Skip augmented audios")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for log files")
    parser.add_argument("--auto_skip", action="store_true", default=True, help="Automatically skip completed steps")
    parser.add_argument("--no_auto_skip", dest="auto_skip", action="store_false", help="Do not skip completed steps")
    parser.add_argument("--hf_token", type=str, default=None, 
                       help="Hugging Face token for speaker diarization (or set HF_TOKEN/HUGGINGFACE_TOKEN environment variable)")
    parser.add_argument("--max_files", type=int, default=None, 
                       help="Maximum number of files to process per class (for testing). If not specified, process all files.")
    parser.add_argument("--n_folds", type=int, default=None, 
                       help="Number of folds for cross-validation (e.g., 5 for 5-fold CV). If not specified, run single train/val split.")
    parser.add_argument("--mode", type=str, default="multimodal", choices=["audio", "text", "multimodal"],
                       help="Training mode: 'audio' (audio only), 'text' (text only), or 'multimodal' (both)")
    parser.add_argument("--test_audio_dir", type=str, default="test-dist/audio",
                       help="Test audio directory (e.g., test-dist/audio). If specified, test data will be processed and evaluated.")
    parser.add_argument("--test_ground_truth_csv", type=str, default="test-dist/task1.csv",
                       help="Test ground truth CSV file (e.g., test-dist/task1.csv). Must be specified with --test_audio_dir.")
    
    args = parser.parse_args()
    
    # For 5-fold CV, first run the data split for all folds
    if args.n_folds is not None and args.n_folds > 1:
        logger = setup_logger("pipeline", log_dir=Path(args.log_dir), log_level=logging.INFO)
        logger.info(f"Creating {args.n_folds}-fold cross-validation splits...")
        
        # Run split_dataset.py to split all folds
        python_exe = get_python_executable()
        cmd = [
            python_exe, "-u", "split_dataset.py",
            "--audio_dir", args.audio_dir,
            "--output_dir", args.data_dir,
            "--val_ratio", str(args.val_ratio),
            "--seed", str(args.seed),
            "--n_folds", str(args.n_folds)
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.returncode != 0:
            logger.error(f"Failed to create fold splits: {process.stderr}")
            sys.exit(1)
        
        logger.info("Fold splits created successfully!")
    
    run_full_pipeline(
        audio_dir=args.audio_dir,
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        whisper_model=args.whisper_model,
        language=args.language,
        num_references=args.num_references,
        num_augmentations=args.num_augmentations,
        fish_speech_path=args.fish_speech_path,
        random_seed=args.seed,
        skip_steps=args.skip,
        use_augmented=args.use_augmented,
        log_dir=args.log_dir,
        auto_skip=args.auto_skip,
        hf_token=args.hf_token,
        max_files=args.max_files,
        n_folds=args.n_folds,
        mode=args.mode,
        test_audio_dir=args.test_audio_dir,
        test_ground_truth_csv=args.test_ground_truth_csv,
    )

