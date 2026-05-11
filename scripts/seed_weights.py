"""
seed_weights.py
===============
Quick-seed script that trains BGC-Prophet (annotator + classifier) on a tiny
synthetic dataset and saves valid weight files for ESM2 35M/150M/650M models.

No internet connection required — embeddings are random tensors of the correct
dimensionality for the chosen ESM2 variant.  The sole purpose is to produce
loadable .pt files that the main pipeline can consume immediately.

NOTE: The default ESM2-8M model already ships with official pre-trained weights
from the BGC-Prophet authors at models/model/annotator.pt and classifier.pt.
This script is only needed for larger models (35M, 150M, 650M).

The annotator and classifier are always trained at the **native embedding
dimension** of the chosen ESM2 model — no PCA or projection is used.

Output
------
  models/model/<esm_model_name>/annotator.pt
  models/model/<esm_model_name>/classifier.pt

Usage
-----
  # Seed 35M weights (480-dim) — default
  python scripts/seed_weights.py

  # Seed a specific model
  python scripts/seed_weights.py --model esm2_t12_35M_UR50D
  python scripts/seed_weights.py --model esm2_t30_150M_UR50D
  python scripts/seed_weights.py --model esm2_t33_650M_UR50D

  # All larger models in one go
  for model in esm2_t12_35M_UR50D esm2_t30_150M_UR50D esm2_t33_650M_UR50D; do
      python scripts/seed_weights.py --model "$model"
  done
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# ESM2 model registry — must stay in sync with bgc_predictor.py / train_prophet.py
# ---------------------------------------------------------------------------
# NOTE: 8M is excluded — it uses official BGC-Prophet weights (nhead=3) at
# models/model/ root. This script only seeds weights for larger ESM2 models.
ESM2_REGISTRY: dict[str, dict] = {
    "esm2_t12_35M_UR50D": {"embed_dim": 480, "params": "35M"},
    "esm2_t30_150M_UR50D": {"embed_dim": 640, "params": "150M"},
    "esm2_t33_650M_UR50D": {"embed_dim": 1280, "params": "650M"},
}

# BGC-Prophet architecture constants — fixed for all model sizes
WINDOW_SIZE = 128  # proteins per window
NUM_CLASSES = 7  # BGC class labels (matches _PROPHET_TYPE_LABELS)
NHEAD = 5  # attention heads (embed_dim must be divisible by 5)
NUM_ENCODER_LAYERS = 2  # transformer encoder depth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


class SyntheticBGCDataset(Dataset):
    """Random tensors shaped exactly like real BGC-Prophet training windows.

    Each item is a tuple of:
        emb         : (WINDOW_SIZE, embed_dim)  float32
        binary      : (WINDOW_SIZE,)            float32  per-position BGC label
        class_labels: (NUM_CLASSES,)            float32  multi-hot class label
        pad_mask    : (WINDOW_SIZE,)            float32  1=padded, 0=real
        is_positive : scalar                    float32  1=BGC window, 0=non-BGC
    """

    def __init__(self, n_windows: int, embed_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        n_pos = n_windows // 2
        n_neg = n_windows - n_pos

        # Embeddings: (N, WINDOW_SIZE, embed_dim)
        emb = rng.standard_normal((n_windows, WINDOW_SIZE, embed_dim)).astype(
            np.float32
        )

        # Padding masks: first 32-128 tokens real (0), rest padded (1)
        pad_masks = np.ones((n_windows, WINDOW_SIZE), dtype=np.float32)
        for i in range(n_windows):
            real_len = rng.integers(32, WINDOW_SIZE + 1)
            pad_masks[i, :real_len] = 0.0

        # Per-position binary labels (annotator target): (N, WINDOW_SIZE)
        binary = np.zeros((n_windows, WINDOW_SIZE), dtype=np.float32)
        for i in range(n_pos):
            real_positions = np.where(pad_masks[i] == 0)[0]
            n_label = max(1, int(len(real_positions) * 0.3))
            chosen = rng.choice(real_positions, size=n_label, replace=False)
            binary[i, chosen] = 1.0

        # Multi-hot class labels (classifier target): (N, NUM_CLASSES)
        class_labels = np.zeros((n_windows, NUM_CLASSES), dtype=np.float32)
        for i in range(n_pos):
            n_cls = rng.integers(1, 3)
            idxs = rng.choice(NUM_CLASSES, size=n_cls, replace=False)
            class_labels[i, idxs] = 1.0

        # is_positive flag: (N,)
        is_pos = np.array([1.0] * n_pos + [0.0] * n_neg, dtype=np.float32)

        self.emb = torch.from_numpy(emb)
        self.binary = torch.from_numpy(binary)
        self.class_labels = torch.from_numpy(class_labels)
        self.pad_masks = torch.from_numpy(pad_masks)
        self.is_pos = torch.from_numpy(is_pos)

    def __len__(self) -> int:
        return len(self.emb)

    def __getitem__(self, idx: int):
        return (
            self.emb[idx],
            self.binary[idx],
            self.class_labels[idx],
            self.pad_masks[idx],
            self.is_pos[idx],
        )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_annotator(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    save_path: Path,
) -> None:
    """Train the annotator (per-position binary classification).

    transformerEncoderNet.forward(src) takes only src — no mask argument.
    Loss is BCE masked to non-padded positions only.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()
    best_val = float("inf")

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for emb, binary, _, pad_mask, _ in train_loader:
            emb = emb.to(device)
            binary = binary.to(device)
            pad_mask = pad_mask.to(device)

            optimizer.zero_grad()
            out = model(emb)  # (B, W) — no mask arg
            real = pad_mask == 0
            if real.sum() == 0:
                continue
            loss = criterion(out[real], binary[real])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for emb, binary, _, pad_mask, _ in val_loader:
                emb = emb.to(device)
                binary = binary.to(device)
                pad_mask = pad_mask.to(device)
                out = model(emb)
                real = pad_mask == 0
                if real.sum() == 0:
                    continue
                val_loss += criterion(out[real], binary[real]).item()

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)
        logger.info(
            "[Annotator] Epoch %3d/%d  train=%.4f  val=%.4f",
            epoch,
            epochs,
            avg_train,
            avg_val,
        )

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), save_path)
            logger.info("  ✓ Saved best annotator (val=%.4f) → %s", best_val, save_path)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    save_path: Path,
) -> None:
    """Train the classifier (BGC-class multi-label prediction).

    transformerClassifier.forward(src, src_key_padding_mask) takes mask as
    a positional argument.  The model already applies Sigmoid, so BCELoss
    (not BCEWithLogitsLoss) is used.  Only positive windows contribute.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()  # model already applies Sigmoid
    best_val = float("inf")

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for emb, _, class_labels, pad_mask, is_pos in train_loader:
            pos_idx = is_pos.bool()
            if pos_idx.sum() == 0:
                continue
            emb = emb[pos_idx].to(device)
            class_labels = class_labels[pos_idx].to(device)
            pad_mask = pad_mask[pos_idx].to(device)

            optimizer.zero_grad()
            out = model(emb, pad_mask.bool())  # (B, NUM_CLASSES)
            loss = criterion(out, class_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for emb, _, class_labels, pad_mask, is_pos in val_loader:
                pos_idx = is_pos.bool()
                if pos_idx.sum() == 0:
                    continue
                emb = emb[pos_idx].to(device)
                class_labels = class_labels[pos_idx].to(device)
                pad_mask = pad_mask[pos_idx].to(device)
                out = model(emb, pad_mask.bool())
                val_loss += criterion(out, class_labels).item()
                n_val += 1

        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_loss / max(n_val, 1)
        logger.info(
            "[Classifier] Epoch %3d/%d  train=%.4f  val=%.4f",
            epoch,
            epochs,
            avg_train,
            avg_val,
        )

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), save_path)
            logger.info(
                "  ✓ Saved best classifier (val=%.4f) → %s", best_val, save_path
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Seed BGC-Prophet weights from synthetic data (no internet required).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            [
                "Examples:",
                "  python scripts/seed_weights.py                          # 35M (default)",
                "  python scripts/seed_weights.py --model esm2_t12_35M_UR50D",
                "  python scripts/seed_weights.py --model esm2_t30_150M_UR50D",
                "  python scripts/seed_weights.py --model esm2_t33_650M_UR50D",
            ]
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default="esm2_t12_35M_UR50D",
        choices=list(ESM2_REGISTRY.keys()),
        help="ESM2 model to seed weights for (default: esm2_t12_35M_UR50D).",
    )
    p.add_argument(
        "--epochs", type=int, default=10, help="Training epochs (default: 10)"
    )
    p.add_argument(
        "--n-windows",
        type=int,
        default=100,
        help="Synthetic windows to generate (default: 100)",
    )
    p.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    p.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models" / "model",
        help="Root model directory (default: models/model)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = ESM2_REGISTRY[args.model]
    embed_dim = spec["embed_dim"]
    dim_ff = embed_dim * 4

    logger.info("═" * 60)
    logger.info("Seeding BGC-Prophet weights for %s", args.model)
    logger.info("  embed_dim : %d", embed_dim)
    logger.info("  dim_ff    : %d", dim_ff)
    logger.info("  windows   : %d  |  epochs: %d", args.n_windows, args.epochs)
    logger.info("═" * 60)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # Output paths
    out_dir = args.output_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    annotator_path = out_dir / "annotator.pt"
    classifier_path = out_dir / "classifier.pt"

    # Import BGC-Prophet model classes
    try:
        from bgc_prophet.train.model import transformerEncoderNet
        from bgc_prophet.train.classifier import transformerClassifier
    except ImportError as exc:
        logger.error("bgc_prophet package not found: %s", exc)
        logger.error("Install with: pip install bgc-prophet")
        sys.exit(1)

    # Dataset
    logger.info(
        "Generating %d synthetic windows (embed_dim=%d) …", args.n_windows, embed_dim
    )
    dataset = SyntheticBGCDataset(
        n_windows=args.n_windows, embed_dim=embed_dim, seed=args.seed
    )
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ── Annotator ──────────────────────────────────────────────────────────
    logger.info("Training annotator …")
    annotator = transformerEncoderNet(
        d_model=embed_dim,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        max_len=WINDOW_SIZE,
        dim_feedforward=dim_ff,
    )
    train_annotator(
        annotator,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.lr,
        annotator_path,
    )
    del annotator
    if device.type in ("cuda", "mps"):
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── Classifier ─────────────────────────────────────────────────────────
    logger.info("Training classifier …")
    classifier = transformerClassifier(
        d_model=embed_dim,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        max_len=WINDOW_SIZE,
        dim_feedforward=dim_ff,
        labels_num=NUM_CLASSES,
    )
    train_classifier(
        classifier,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.lr,
        classifier_path,
    )
    del classifier
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info("")
    logger.info("═" * 60)
    logger.info("Weights saved:")
    logger.info("  %s", annotator_path)
    logger.info("  %s", classifier_path)
    logger.info("═" * 60)
    logger.info("Run the pipeline with:")
    logger.info(
        "  python main.py --mock --esm-model %s --model-dir %s",
        args.model,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
