"""
Transformer fine-tuning module.

Fine-tunes the transformer pipeline (`roberta-base` backbone) on the GPU using
`train.spacy` and `dev.spacy`. Produces checkpoints at `models/ner_trf/model-best`
and `models/ner_trf/model-last`.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scripts
import spacy
from spacy.cli.train import train as spacy_train
from spacy.tokens import DocBin

logger = logging.getLogger("ojt_pipeline.training")


def ensure_config_exists(
    config_path: str = "config_trf.cfg",
    lang: str = "en",
    optimize: str = "accuracy"
) -> str:
    """Ensures a GPU transformer config exists, creating one via spaCy CLI if absent."""
    if not os.path.exists(config_path):
        logger.info(f"Generating default transformer config '{config_path}'...")
        os.system(
            f"{sys.executable} -m spacy init config {config_path} "
            f"-l {lang} -p ner --optimize {optimize} --gpu --force"
        )
    return config_path


def train_ner_trf(
    config_path: str = "config_trf.cfg",
    output_dir: str = "models/ner_trf",
    train_path: str = "data/training/train.spacy",
    dev_path: str = "data/training/dev.spacy",
    max_steps: int = 2500,
    eval_frequency: int = 50,
    patience: int = 400,
    use_gpu: int = 0,
) -> Dict[str, Any]:
    """Trains/fine-tunes the transformer NER pipeline on GPU with patience-based early stopping.
    
    Args:
        config_path: Path to spaCy config file.
        output_dir: Destination directory for model checkpoints.
        train_path: Path to training DocBin (.spacy).
        dev_path: Path to evaluation DocBin (.spacy).
        max_steps: Maximum training steps (default: 2500).
        eval_frequency: Frequency of evaluation on dev set (default: 50).
        patience: Steps without improvement on dev set before early stopping (default: 400).
        use_gpu: GPU device ID (0 for RTX 3060).
        
    Returns:
        Dict with paths to best model and dev evaluation metrics.
    """
    scripts.init_gpu()
    ensure_config_exists(config_path)

    os.makedirs(output_dir, exist_ok=True)

    overrides = {
        "paths.train": train_path,
        "paths.dev": dev_path,
        "training.max_steps": max_steps,
        "training.eval_frequency": eval_frequency,
        "training.patience": patience,
    }

    logger.info(
        f"Starting transformer training on GPU {use_gpu} for up to {max_steps} steps "
        f"(eval_frequency={eval_frequency}, patience={patience} steps)..."
    )
    spacy_train(config_path, output_dir, use_gpu=use_gpu, overrides=overrides)

    best_model_path = os.path.join(output_dir, "model-best")
    last_model_path = os.path.join(output_dir, "model-last")

    logger.info(f"Training completed. Best model checkpoint: {best_model_path}")
    return {
        "best_model_path": best_model_path,
        "last_model_path": last_model_path,
        "status": "SUCCESS",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune transformer NER on GPU.")
    parser.add_argument("--steps", type=int, default=2500, help="Maximum training steps")
    parser.add_argument("--eval-freq", type=int, default=50, help="Evaluation frequency")
    parser.add_argument("--patience", type=int, default=400, help="Early stopping patience (steps)")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    res = train_ner_trf(
        max_steps=args.steps,
        eval_frequency=args.eval_freq,
        patience=args.patience,
        use_gpu=args.gpu_id
    )
    print("Fine-tuning output:", res)
