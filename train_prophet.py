#!/usr/bin/env python3
"""
Train BGC-Prophet annotator and classifier for any ESM2 model size.

This script retrains the BGC-Prophet TransformerEncoder models at each ESM2
embedding dimension, so that larger models (35M, 150M, 650M) get native-dimension
weights. The default ESM2-8M model already ships with official pre-trained weights
from the BGC-Prophet authors at models/model/annotator.pt and classifier.pt.
This script is only needed for non-8M models.

Usage:
    python train_prophet.py \\
        --esm-model esm2_t30_150M_UR50D \\
        --data-dir data/training \\
        --output-dir models/model \\
        --epochs 50 \\
        --batch-size 16 \\
        --device auto

Output:
    models/model/{esm_model_name}/annotator.pt
    models/model/{esm_model_name}/classifier.pt
    models/model/{esm_model_name}/training_meta.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_prophet")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# NOTE: 8M is excluded here — the default model uses the official upstream
# BGC-Prophet weights at models/model/ root (nhead=3). This script only trains
# weights for larger models (35M, 150M, 650M) at their native embedding dimensions.
ESM2_REGISTRY: Dict[str, Dict] = {
    "esm2_t12_35M_UR50D": {"layers": 12, "embed_dim": 480, "params": "35M"},
    "esm2_t30_150M_UR50D": {"layers": 30, "embed_dim": 640, "params": "150M"},
    "esm2_t33_650M_UR50D": {"layers": 33, "embed_dim": 1280, "params": "650M"},
}

# BGC class labels — same order as bgc_predictor.py _PROPHET_TYPE_LABELS
BGC_CLASS_LABELS: List[str] = [
    "Alkaloid",
    "Terpene",
    "NRP",
    "Polyketide",
    "RiPP",
    "Saccharide",
    "Other",
]
BGC_CLASS_MAP: Dict[str, int] = {name: i for i, name in enumerate(BGC_CLASS_LABELS)}
NUM_CLASSES: int = len(BGC_CLASS_LABELS)

# MIBiG biosynthetic class name normalization
_MIBIG_CLASS_NORMALIZE: Dict[str, str] = {
    "Alkaloid": "Alkaloid",
    "Terpene": "Terpene",
    "NRP": "NRP",
    "NRPS": "NRP",
    "Polyketide": "Polyketide",
    "PKS": "Polyketide",
    "RiPP": "RiPP",
    "Saccharide": "Saccharide",
    "Other": "Other",
}

WINDOW_SIZE: int = 128
ESM2_MAX_SEQ_LEN: int = 1022

MIBIG_URL: str = "https://dl.secondarymetabolites.org/mibig/mibig_json_3.1.tar.gz"
MIBIG_TARBALL: str = "mibig_json_3.1.tar.gz"

MIBIG_FASTA_URL: str = (
    "https://dl.secondarymetabolites.org/mibig/mibig_prot_seqs_3.1.fasta"
)
MIBIG_FASTA_FILE: str = "mibig_prot_seqs_3.1.fasta"
# Amino acid frequencies (natural, for generating realistic negative sequences)
_AA_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWY"
_AA_FREQUENCIES: List[float] = [
    0.0825,
    0.0137,
    0.0545,
    0.0675,
    0.0386,
    0.0707,
    0.0227,
    0.0596,
    0.0584,
    0.0966,
    0.0242,
    0.0406,
    0.0470,
    0.0393,
    0.0553,
    0.0656,
    0.0534,
    0.0687,
    0.0108,
    0.0292,
]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BGCEntry:
    """A parsed MIBiG BGC cluster with its gene protein sequences."""

    bgc_id: str
    biosyn_classes: List[str]  # Normalized class names
    class_vector: np.ndarray  # (7,) multi-hot float32
    protein_sequences: List[str]  # Protein amino acid sequences
    protein_ids: List[str]  # Unique identifiers per protein


@dataclass
class TrainingWindow:
    """A 128-gene training window with labels."""

    window_id: str
    protein_sequences: List[str]  # Up to 128 protein sequences
    binary_labels: np.ndarray  # (128,) float32 — 1=BGC gene, 0=pad/NonBGC
    class_labels: np.ndarray  # (7,) float32 — multi-hot BGC class
    padding_mask: np.ndarray  # (128,) bool — True where padded
    is_positive: bool  # True if BGC-positive window


# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------


def download_mibig(data_dir: Path, max_retries: int = 3) -> Path:
    """Download and extract MIBiG JSON archive.

    Returns path to the directory containing extracted JSON files.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = data_dir / MIBIG_TARBALL

    # Find extracted JSON directory (may already exist)
    json_dir = _find_mibig_json_dir(data_dir)
    if json_dir is not None:
        logger.info("MIBiG JSONs already extracted at %s", json_dir)
        return json_dir

    # Download tarball if not cached
    if not tarball_path.exists():
        logger.info("Downloading MIBiG from %s ...", MIBIG_URL)
        for attempt in range(1, max_retries + 1):
            try:
                urllib.request.urlretrieve(MIBIG_URL, str(tarball_path))
                logger.info("Download complete (%s).", tarball_path)
                break
            except Exception as exc:
                logger.warning(
                    "Download attempt %d/%d failed: %s", attempt, max_retries, exc
                )
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to download MIBiG after {max_retries} attempts"
                    ) from exc
                time.sleep(2**attempt)

    # Extract
    logger.info("Extracting %s ...", tarball_path.name)
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(path=str(data_dir))

    json_dir = _find_mibig_json_dir(data_dir)
    if json_dir is None:
        raise RuntimeError(
            f"Extraction succeeded but no JSON directory found in {data_dir}"
        )
    logger.info("MIBiG extracted to %s", json_dir)
    return json_dir


def _find_mibig_json_dir(data_dir: Path) -> Optional[Path]:
    """Locate the MIBiG JSON directory inside data_dir."""
    for candidate in sorted(data_dir.iterdir()):
        if candidate.is_dir() and "mibig" in candidate.name.lower():
            # Check if it contains .json files directly or in a subdirectory
            jsons = list(candidate.glob("*.json"))
            if jsons:
                return candidate
            # Some archives nest: mibig_json_3.1/mibig_json_3.1/*.json
            for sub in candidate.iterdir():
                if sub.is_dir():
                    jsons = list(sub.glob("*.json"))
                    if jsons:
                        return sub
    return None


def download_mibig_fasta(data_dir: Path, max_retries: int = 3) -> Path:
    """Download the MIBiG v3.1 proteins FASTA file.

    MIBiG v3.1 JSON files do not contain protein sequences; sequences are
    distributed as a separate FASTA file at MIBIG_FASTA_URL.

    Returns path to the local FASTA file.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = data_dir / MIBIG_FASTA_FILE

    if fasta_path.exists():
        logger.info("MIBiG FASTA already cached at %s", fasta_path)
        return fasta_path

    logger.info("Downloading MIBiG proteins FASTA from %s ...", MIBIG_FASTA_URL)
    for attempt in range(1, max_retries + 1):
        try:
            urllib.request.urlretrieve(MIBIG_FASTA_URL, str(fasta_path))
            logger.info("Download complete (%s).", fasta_path)
            return fasta_path
        except Exception as exc:
            logger.warning(
                "Download attempt %d/%d failed: %s", attempt, max_retries, exc
            )
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to download MIBiG FASTA after {max_retries} attempts"
                ) from exc
            time.sleep(2**attempt)

    # Unreachable but satisfies type checker
    return fasta_path


def parse_mibig_fasta(fasta_path: Path) -> Dict[str, List[str]]:
    """Parse the MIBiG proteins FASTA into a BGC-ID → sequences mapping.

    MIBiG v3.1 FASTA headers have the format:
        >BGC0000001|c1|1-1083|+|AEK75490.1|protein_methyltransferase|AEK75490.1
    where field[0] (pipe-separated) is the BGC accession.

    Returns:
        Dict mapping each BGC ID to its list of protein sequences.
    """
    protein_map: Dict[str, List[str]] = {}
    current_bgc: Optional[str] = None
    current_seq_parts: List[str] = []

    logger.info("Parsing MIBiG FASTA %s ...", fasta_path.name)

    with open(fasta_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Flush previous sequence
                if current_bgc is not None and current_seq_parts:
                    seq = "".join(current_seq_parts).upper().rstrip("*")
                    if len(seq) >= 10:
                        protein_map.setdefault(current_bgc, []).append(seq)
                # Parse new header: >BGC0000001|c1|...
                header = line[1:]  # strip '>'
                fields = header.split("|")
                current_bgc = fields[0].strip() if fields else None
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

    # Flush last sequence
    if current_bgc is not None and current_seq_parts:
        seq = "".join(current_seq_parts).upper().rstrip("*")
        if len(seq) >= 10:
            protein_map.setdefault(current_bgc, []).append(seq)

    total_proteins = sum(len(v) for v in protein_map.values())
    logger.info(
        "Loaded %d proteins across %d BGC clusters from FASTA.",
        total_proteins,
        len(protein_map),
    )
    return protein_map


def parse_mibig_entries(
    json_dir: Path,
    protein_map: Dict[str, List[str]],
) -> List[BGCEntry]:
    """Parse MIBiG JSON files into BGCEntry objects.

    Biosynthetic class information comes from the JSON files; protein
    sequences come from the pre-built ``protein_map`` (parsed from the
    MIBiG proteins FASTA, since v3.1 JSONs do not embed sequences).

    Args:
        json_dir:    Directory containing MIBiG *.json files.
        protein_map: Dict mapping BGC ID → list of protein sequences,
                     as returned by :func:`parse_mibig_fasta`.

    Returns:
        List of :class:`BGCEntry` objects with sequences attached.
    """
    entries: List[BGCEntry] = []
    json_files = sorted(json_dir.glob("*.json"))
    logger.info("Parsing %d MIBiG JSON files ...", len(json_files))

    skipped = 0
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Skipping malformed JSON %s: %s", jf.name, exc)
            skipped += 1
            continue

        # Navigate MIBiG JSON structure (v3.x wraps everything under 'cluster')
        cluster = data.get("cluster", data)
        bgc_id = cluster.get("mibig_accession", jf.stem)

        # Look up protein sequences from FASTA-derived map
        proteins = protein_map.get(bgc_id, [])
        if not proteins:
            # No proteins in FASTA for this BGC — skip
            skipped += 1
            continue

        # Sanitize sequences: remove non-standard AAs, enforce minimum length
        clean_proteins: List[str] = []
        for seq in proteins:
            seq = seq.strip().rstrip("*").upper()
            if not all(c in _AA_ALPHABET for c in seq):
                seq = "".join(c for c in seq if c in _AA_ALPHABET)
            if len(seq) >= 10:
                clean_proteins.append(seq)

        if not clean_proteins:
            skipped += 1
            continue

        # Extract biosynthetic classes
        raw_classes = cluster.get("biosyn_class", [])
        if isinstance(raw_classes, str):
            raw_classes = [raw_classes]

        norm_classes: List[str] = []
        for rc in raw_classes:
            normalized = _MIBIG_CLASS_NORMALIZE.get(rc, None)
            if normalized is None:
                # Try case-insensitive match
                for key, val in _MIBIG_CLASS_NORMALIZE.items():
                    if key.lower() == rc.lower():
                        normalized = val
                        break
                if normalized is None:
                    normalized = "Other"
            norm_classes.append(normalized)

        if not norm_classes:
            norm_classes = ["Other"]

        # Build multi-hot class vector
        class_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
        for nc in norm_classes:
            idx = BGC_CLASS_MAP.get(nc, BGC_CLASS_MAP["Other"])
            class_vec[idx] = 1.0

        # Use BGC accession IDs as protein identifiers where available;
        # fall back to positional IDs if the FASTA had no per-protein IDs.
        gene_list = cluster.get("genes", {}).get("annotations", [])
        gene_ids = [
            g.get("id", g.get("name", f"{bgc_id}_gene_{i}"))
            for i, g in enumerate(gene_list)
        ]
        # Pad or trim gene_ids to match clean_proteins length
        if len(gene_ids) < len(clean_proteins):
            gene_ids += [
                f"{bgc_id}_prot_{i}" for i in range(len(gene_ids), len(clean_proteins))
            ]
        protein_ids = gene_ids[: len(clean_proteins)]

        entries.append(
            BGCEntry(
                bgc_id=bgc_id,
                biosyn_classes=norm_classes,
                class_vector=class_vec,
                protein_sequences=clean_proteins,
                protein_ids=protein_ids,
            )
        )

    logger.info(
        "Parsed %d BGC entries (%d skipped) with %d total proteins.",
        len(entries),
        skipped,
        sum(len(e.protein_sequences) for e in entries),
    )
    return entries


def generate_negative_sequences(num_proteins: int, seed: int = 42) -> List[str]:
    """Generate synthetic NonBGC protein sequences.

    Uses natural amino acid frequencies to produce realistic-looking proteins
    of varying lengths (100-600 residues). These serve as negative training
    examples since they lack BGC function.
    """
    rng = random.Random(seed)
    proteins: List[str] = []
    for _ in range(num_proteins):
        length = rng.randint(100, 600)
        seq = "".join(rng.choices(_AA_ALPHABET, weights=_AA_FREQUENCIES, k=length))
        proteins.append(seq)
    return proteins


def create_training_windows(
    bgc_entries: List[BGCEntry],
    num_negative_windows: int = 500,
    seed: int = 42,
) -> List[TrainingWindow]:
    """Create 128-gene training windows from BGC entries and negative data.

    Positive windows:
        - Center each BGC's genes in a 128-gene window
        - Genes belonging to the BGC are labeled 1, padding is 0
        - Class labels are multi-hot from the BGC entry

    Negative windows:
        - 128 synthetic protein sequences per window
        - All labels are 0 (NonBGC)
        - Class labels are all-zero
    """
    windows: List[TrainingWindow] = []
    rng = random.Random(seed)

    # --- Positive windows ---
    for entry in bgc_entries:
        n_genes = len(entry.protein_sequences)

        if n_genes >= WINDOW_SIZE:
            # Slide non-overlapping windows across the genes
            for start in range(0, n_genes, WINDOW_SIZE):
                end = min(start + WINDOW_SIZE, n_genes)
                win_seqs = entry.protein_sequences[start:end]
                n_actual = len(win_seqs)

                binary = np.zeros(WINDOW_SIZE, dtype=np.float32)
                binary[:n_actual] = 1.0
                pad_mask = np.ones(WINDOW_SIZE, dtype=bool)
                pad_mask[:n_actual] = False

                # Pad sequences list to WINDOW_SIZE
                win_seqs_padded = win_seqs + [""] * (WINDOW_SIZE - n_actual)

                windows.append(
                    TrainingWindow(
                        window_id=f"{entry.bgc_id}_w{start}",
                        protein_sequences=win_seqs_padded,
                        binary_labels=binary,
                        class_labels=entry.class_vector.copy(),
                        padding_mask=pad_mask,
                        is_positive=True,
                    )
                )
        else:
            # Center genes in the window
            pad_before = (WINDOW_SIZE - n_genes) // 2
            binary = np.zeros(WINDOW_SIZE, dtype=np.float32)
            binary[pad_before : pad_before + n_genes] = 1.0
            pad_mask = np.ones(WINDOW_SIZE, dtype=bool)
            pad_mask[pad_before : pad_before + n_genes] = False

            win_seqs = (
                [""] * pad_before
                + entry.protein_sequences
                + [""] * (WINDOW_SIZE - pad_before - n_genes)
            )

            windows.append(
                TrainingWindow(
                    window_id=f"{entry.bgc_id}_w0",
                    protein_sequences=win_seqs,
                    binary_labels=binary,
                    class_labels=entry.class_vector.copy(),
                    padding_mask=pad_mask,
                    is_positive=True,
                )
            )

    n_positive = len(windows)
    logger.info("Created %d positive training windows.", n_positive)

    # --- Negative windows ---
    total_neg_proteins = num_negative_windows * WINDOW_SIZE
    neg_proteins = generate_negative_sequences(total_neg_proteins, seed=seed)

    for wi in range(num_negative_windows):
        start = wi * WINDOW_SIZE
        win_seqs = neg_proteins[start : start + WINDOW_SIZE]
        windows.append(
            TrainingWindow(
                window_id=f"neg_{wi}",
                protein_sequences=win_seqs,
                binary_labels=np.zeros(WINDOW_SIZE, dtype=np.float32),
                class_labels=np.zeros(NUM_CLASSES, dtype=np.float32),
                padding_mask=np.zeros(WINDOW_SIZE, dtype=bool),  # all real (no padding)
                is_positive=False,
            )
        )

    logger.info(
        "Created %d negative windows. Total: %d windows.",
        num_negative_windows,
        len(windows),
    )

    # Shuffle
    rng.shuffle(windows)
    return windows


# ---------------------------------------------------------------------------
# ESM2 Embedding Extraction
# ---------------------------------------------------------------------------


def _detect_device(preference: str = "auto") -> torch.device:
    """Detect best available device: cuda > mps > cpu."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_esm2_model(
    model_name: str, device: torch.device
) -> Tuple["torch.nn.Module", Any, int]:
    """Load an ESM2 model and return (model, alphabet, repr_layer)."""
    import esm  # noqa: F811 — deferred import for clarity

    info = ESM2_REGISTRY[model_name]
    repr_layer = info["layers"]

    logger.info("Loading ESM2 model: %s (%s params) ...", model_name, info["params"])
    model, alphabet = getattr(esm.pretrained, model_name)()
    model = model.to(device).eval()
    logger.info("ESM2 loaded on %s.", device)

    return model, alphabet, repr_layer


def precompute_embeddings(
    windows: List[TrainingWindow],
    esm_model: "torch.nn.Module",
    alphabet: Any,
    repr_layer: int,
    embed_dim: int,
    device: torch.device,
    batch_size: int = 8,
    cache_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Pre-compute ESM2 embeddings for all unique proteins across windows.

    Returns dict mapping protein_sequence → (embed_dim,) numpy array.
    Uses mean-pooling over residue positions (excluding BOS/EOS tokens).
    Caches results to disk for re-use.
    """
    # Check cache
    if cache_path and cache_path.exists():
        logger.info("Loading cached embeddings from %s ...", cache_path)
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        logger.info("Loaded %d cached embeddings.", len(cached))
        return cached

    # Collect unique protein sequences
    unique_seqs: Dict[str, None] = {}  # ordered set
    for win in windows:
        for seq in win.protein_sequences:
            if seq and seq not in unique_seqs:
                unique_seqs[seq] = None

    seq_list = list(unique_seqs.keys())
    logger.info("Computing ESM2 embeddings for %d unique proteins ...", len(seq_list))

    batch_converter = alphabet.get_batch_converter()
    embeddings: Dict[str, np.ndarray] = {}

    total_batches = (len(seq_list) + batch_size - 1) // batch_size

    for bi in range(0, len(seq_list), batch_size):
        batch_seqs = seq_list[bi : bi + batch_size]
        batch_num = bi // batch_size + 1

        # Truncate sequences > 1022 residues
        data = [
            (f"prot_{bi + j}", seq[:ESM2_MAX_SEQ_LEN])
            for j, seq in enumerate(batch_seqs)
        ]

        try:
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = esm_model(batch_tokens, repr_layers=[repr_layer])
            representations = results["representations"][repr_layer]  # (B, L, D)

            # Mean-pool over residue positions (skip BOS at 0)
            for j, seq in enumerate(batch_seqs):
                seq_len = min(len(seq), ESM2_MAX_SEQ_LEN)
                # Token positions: 0=BOS, 1..seq_len=residues, seq_len+1=EOS
                emb = representations[j, 1 : seq_len + 1, :].mean(dim=0)  # (D,)
                embeddings[seq] = emb.cpu().numpy()

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.error(
                    "GPU OOM at batch %d/%d. Try reducing --batch-size. "
                    "Current batch has %d sequences, longest=%d residues.",
                    batch_num,
                    total_batches,
                    len(batch_seqs),
                    max(len(s) for s in batch_seqs),
                )
                # Clear GPU cache and retry with smaller sub-batches
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                for j, seq in enumerate(batch_seqs):
                    try:
                        single_data = [(f"prot_{bi + j}", seq[:ESM2_MAX_SEQ_LEN])]
                        _, _, single_tokens = batch_converter(single_data)
                        single_tokens = single_tokens.to(device)
                        with torch.no_grad():
                            res = esm_model(single_tokens, repr_layers=[repr_layer])
                        rep = res["representations"][repr_layer]
                        seq_len = min(len(seq), ESM2_MAX_SEQ_LEN)
                        emb = rep[0, 1 : seq_len + 1, :].mean(dim=0)
                        embeddings[seq] = emb.cpu().numpy()
                    except RuntimeError:
                        logger.warning(
                            "Skipping protein (len=%d) due to OOM even at batch_size=1.",
                            len(seq),
                        )
                        embeddings[seq] = np.zeros(embed_dim, dtype=np.float32)
            else:
                raise

        if batch_num % 50 == 0 or batch_num == total_batches:
            logger.info(
                "  Embedding progress: %d/%d batches (%.1f%%)",
                batch_num,
                total_batches,
                100.0 * batch_num / total_batches,
            )

    logger.info("Computed %d protein embeddings.", len(embeddings))

    # Cache to disk
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Cached embeddings to %s", cache_path)

    return embeddings


def build_window_tensors(
    windows: List[TrainingWindow],
    embeddings: Dict[str, np.ndarray],
    embed_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert TrainingWindows + embeddings into numpy arrays for the Dataset.

    Returns:
        emb_windows:  (N, 128, embed_dim) float32
        binary_labels: (N, 128) float32
        class_labels:  (N, 7) float32
        padding_masks: (N, 128) bool
        is_positive:   (N,) bool
    """
    n = len(windows)
    emb_windows = np.zeros((n, WINDOW_SIZE, embed_dim), dtype=np.float32)
    binary_labels = np.zeros((n, WINDOW_SIZE), dtype=np.float32)
    class_labels = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    padding_masks = np.ones((n, WINDOW_SIZE), dtype=bool)
    is_positive = np.zeros(n, dtype=bool)

    zero_emb = np.zeros(embed_dim, dtype=np.float32)

    for i, win in enumerate(windows):
        binary_labels[i] = win.binary_labels
        class_labels[i] = win.class_labels
        padding_masks[i] = win.padding_mask
        is_positive[i] = win.is_positive

        for j, seq in enumerate(win.protein_sequences):
            if seq:
                emb_windows[i, j] = embeddings.get(seq, zero_emb)
            # else: remains zero (padding)

    return emb_windows, binary_labels, class_labels, padding_masks, is_positive


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class BGCDataset(Dataset):
    """Dataset for BGC-Prophet training.

    Each item returns:
        embedding_window: (128, embed_dim) float32
        binary_labels:    (128,) float32  — per-gene BGC annotation
        class_labels:     (7,) float32    — multi-hot BGC class
        padding_mask:     (128,) bool     — True where padded
        is_positive:      bool            — True if BGC-positive window
    """

    def __init__(
        self,
        emb_windows: np.ndarray,
        binary_labels: np.ndarray,
        class_labels: np.ndarray,
        padding_masks: np.ndarray,
        is_positive: np.ndarray,
    ):
        self.emb_windows = torch.from_numpy(emb_windows)
        self.binary_labels = torch.from_numpy(binary_labels)
        self.class_labels = torch.from_numpy(class_labels)
        self.padding_masks = torch.from_numpy(padding_masks)
        self.is_positive = torch.from_numpy(is_positive)

    def __len__(self) -> int:
        return len(self.emb_windows)

    def __getitem__(self, idx: int):
        return (
            self.emb_windows[idx],
            self.binary_labels[idx],
            self.class_labels[idx],
            self.padding_masks[idx],
            self.is_positive[idx],
        )


# ---------------------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------------------


def train_annotator(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    save_path: Path,
) -> Dict[str, Any]:
    """Train the annotator (per-gene binary BGC classification).

    The annotator predicts a sigmoid probability for each of 128 gene
    positions indicating whether that gene belongs to a BGC.

    Loss: BCELoss (masked to ignore padded positions).
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss(reduction="none")  # per-element, we mask manually

    best_val_loss = float("inf")
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    logger.info("=" * 60)
    logger.info("ANNOTATOR TRAINING — %d epochs", epochs)
    logger.info("=" * 60)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in train_loader:
            emb, binary, _cls, pad_mask, _pos = [b.to(device) for b in batch]
            # emb: (B, 128, D), binary: (B, 128), pad_mask: (B, 128)

            optimizer.zero_grad()
            pred = model(emb)  # (B, 128) — sigmoid output

            loss_raw = criterion(pred, binary)  # (B, 128)
            # Mask out padded positions
            active_mask = ~pad_mask  # True where real genes exist
            loss_masked = (loss_raw * active_mask.float()).sum()
            n_active = active_mask.float().sum().clamp(min=1.0)
            loss = loss_masked / n_active

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * n_active.item()
            train_count += int(n_active.item())

        scheduler.step()
        avg_train = train_loss_sum / max(train_count, 1)
        history["train_loss"].append(avg_train)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                emb, binary, _cls, pad_mask, _pos = [b.to(device) for b in batch]
                pred = model(emb)
                loss_raw = criterion(pred, binary)
                active_mask = ~pad_mask
                loss_masked = (loss_raw * active_mask.float()).sum()
                n_active = active_mask.float().sum().clamp(min=1.0)
                val_loss_sum += (loss_masked / n_active).item() * n_active.item()
                val_count += int(n_active.item())

        avg_val = val_loss_sum / max(val_count, 1)
        history["val_loss"].append(avg_val)

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            marker = " ★ saved"
        else:
            marker = ""

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            logger.info(
                "  Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  lr=%.2e%s",
                epoch,
                epochs,
                avg_train,
                avg_val,
                scheduler.get_last_lr()[0],
                marker,
            )

    logger.info(
        "Annotator training complete. Best val_loss=%.4f (saved to %s)",
        best_val_loss,
        save_path,
    )
    return {"best_val_loss": best_val_loss, "history": history}


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    save_path: Path,
) -> Dict[str, Any]:
    """Train the classifier (7-class multi-label BGC type classification).

    Only BGC-positive windows are used for training. The classifier predicts
    which of the 7 BGC types a gene cluster belongs to.

    Loss: BCEWithLogitsLoss (multi-label).
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    logger.info("=" * 60)
    logger.info("CLASSIFIER TRAINING — %d epochs", epochs)
    logger.info("=" * 60)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in train_loader:
            emb, _binary, cls_labels, pad_mask, is_pos = [b.to(device) for b in batch]
            # Only use positive windows for classifier
            pos_idx = is_pos.bool()
            if not pos_idx.any():
                continue

            emb_pos = emb[pos_idx]
            cls_pos = cls_labels[pos_idx]
            pad_pos = pad_mask[pos_idx]

            optimizer.zero_grad()
            pred = model(emb_pos, pad_pos)  # (B_pos, 7)
            loss = criterion(pred, cls_pos)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * emb_pos.size(0)
            train_count += emb_pos.size(0)

        scheduler.step()
        avg_train = train_loss_sum / max(train_count, 1)
        history["train_loss"].append(avg_train)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                emb, _binary, cls_labels, pad_mask, is_pos = [
                    b.to(device) for b in batch
                ]
                pos_idx = is_pos.bool()
                if not pos_idx.any():
                    continue

                emb_pos = emb[pos_idx]
                cls_pos = cls_labels[pos_idx]
                pad_pos = pad_mask[pos_idx]

                pred = model(emb_pos, pad_pos)
                loss = criterion(pred, cls_pos)
                val_loss_sum += loss.item() * emb_pos.size(0)
                val_count += emb_pos.size(0)

        avg_val = val_loss_sum / max(val_count, 1)
        history["val_loss"].append(avg_val)

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            marker = " ★ saved"
        else:
            marker = ""

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            logger.info(
                "  Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  lr=%.2e%s",
                epoch,
                epochs,
                avg_train,
                avg_val,
                scheduler.get_last_lr()[0],
                marker,
            )

    logger.info(
        "Classifier training complete. Best val_loss=%.4f (saved to %s)",
        best_val_loss,
        save_path,
    )
    return {"best_val_loss": best_val_loss, "history": history}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    """Main training pipeline."""

    # ---- Validate ESM2 model ----
    if args.esm_model not in ESM2_REGISTRY:
        valid = ", ".join(ESM2_REGISTRY.keys())
        logger.error("Unknown ESM2 model '%s'. Valid: %s", args.esm_model, valid)
        sys.exit(1)

    esm_info = ESM2_REGISTRY[args.esm_model]
    embed_dim = esm_info["embed_dim"]
    repr_layer = esm_info["layers"]

    # ---- Device ----
    device = _detect_device(args.device)
    logger.info("Device: %s", device)

    # ---- Seed ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Paths ----
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / args.esm_model
    output_dir.mkdir(parents=True, exist_ok=True)

    annotator_path = output_dir / "annotator.pt"
    classifier_path = output_dir / "classifier.pt"
    meta_path = output_dir / "training_meta.json"
    cache_path = data_dir / f"embeddings_{args.esm_model}.pkl"

    logger.info("=" * 60)
    logger.info("BGC-Prophet Training Pipeline")
    logger.info("=" * 60)
    logger.info(
        "  ESM2 model:    %s (%s params, embed_dim=%d)",
        args.esm_model,
        esm_info["params"],
        embed_dim,
    )
    logger.info("  Data dir:      %s", data_dir)
    logger.info("  Output dir:    %s", output_dir)
    logger.info("  Epochs:        %d", args.epochs)
    logger.info("  Batch size:    %d", args.batch_size)
    logger.info("  Learning rate: %.2e", args.lr)
    logger.info("  Val split:     %.0f%%", args.val_split * 100)
    logger.info("  Seed:          %d", args.seed)

    # ---- Step 1: Download & parse MIBiG ----
    logger.info("-" * 60)
    logger.info("Step 1: Loading MIBiG training data")
    logger.info("-" * 60)

    json_dir = download_mibig(data_dir)
    fasta_path = download_mibig_fasta(data_dir)
    protein_map = parse_mibig_fasta(fasta_path)
    bgc_entries = parse_mibig_entries(json_dir, protein_map)

    # Optionally slice dataset for fast smoke-testing / quick iteration
    if args.max_entries and args.max_entries > 0:
        bgc_entries = bgc_entries[: args.max_entries]
        logger.info("Dataset sliced to %d entries (--max-entries).", len(bgc_entries))

    if not bgc_entries:
        logger.error("No BGC entries parsed from MIBiG. Cannot train.")
        sys.exit(1)

    # Print class distribution
    class_counts = np.zeros(NUM_CLASSES, dtype=int)
    for entry in bgc_entries:
        class_counts += (entry.class_vector > 0).astype(int)
    logger.info("BGC class distribution:")
    for i, name in enumerate(BGC_CLASS_LABELS):
        logger.info("  %-12s %d entries", name, class_counts[i])

    # ---- Step 2: Create training windows ----
    logger.info("-" * 60)
    logger.info("Step 2: Creating training windows")
    logger.info("-" * 60)

    num_neg = max(len(bgc_entries), 500)  # at least as many negatives as positives
    windows = create_training_windows(
        bgc_entries, num_negative_windows=num_neg, seed=args.seed
    )

    # ---- Step 3: Compute ESM2 embeddings ----
    logger.info("-" * 60)
    logger.info("Step 3: Computing ESM2 embeddings")
    logger.info("-" * 60)

    esm_model, alphabet, repr_layer = load_esm2_model(args.esm_model, device)

    embeddings = precompute_embeddings(
        windows=windows,
        esm_model=esm_model,
        alphabet=alphabet,
        repr_layer=repr_layer,
        embed_dim=embed_dim,
        device=device,
        batch_size=args.batch_size,
        cache_path=cache_path,
    )

    # Free ESM2 model to save GPU memory for training
    del esm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("ESM2 model unloaded to free memory.")

    # ---- Step 4: Build tensors & datasets ----
    logger.info("-" * 60)
    logger.info("Step 4: Building datasets")
    logger.info("-" * 60)

    emb_windows, binary_labels, class_labels, padding_masks, is_positive = (
        build_window_tensors(windows, embeddings, embed_dim)
    )

    # Free raw embeddings dict
    del embeddings

    full_dataset = BGCDataset(
        emb_windows, binary_labels, class_labels, padding_masks, is_positive
    )

    # Train/val split
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logger.info("Dataset: %d total, %d train, %d val", n_total, n_train, n_val)
    n_pos = int(is_positive.sum())
    logger.info("Positive windows: %d, Negative windows: %d", n_pos, n_total - n_pos)

    # ---- Step 5: Model architecture ----
    d_model = embed_dim
    nhead = 5
    num_encoder_layers = 2
    dim_feedforward = d_model * 4

    logger.info(
        "Model config: d_model=%d, nhead=%d, layers=%d, ff=%d",
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
    )

    # ---- Step 6: Train annotator ----
    logger.info("-" * 60)
    logger.info("Step 5: Training annotator")
    logger.info("-" * 60)

    try:
        from bgc_prophet.train.model import transformerEncoderNet
    except ImportError:
        logger.error(
            "Cannot import bgc_prophet.train.model.transformerEncoderNet. "
            "Install bgc-prophet: pip install bgc-prophet"
        )
        sys.exit(1)

    annotator = transformerEncoderNet(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        max_len=WINDOW_SIZE,
        dim_feedforward=dim_feedforward,
    )

    ann_results = train_annotator(
        model=annotator,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_path=annotator_path,
    )

    # Free annotator
    del annotator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Step 7: Train classifier ----
    logger.info("-" * 60)
    logger.info("Step 6: Training classifier")
    logger.info("-" * 60)

    try:
        from bgc_prophet.train.classifier import transformerClassifier
    except ImportError:
        logger.error(
            "Cannot import bgc_prophet.train.classifier.transformerClassifier. "
            "Install bgc-prophet: pip install bgc-prophet"
        )
        sys.exit(1)

    classifier = transformerClassifier(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        max_len=WINDOW_SIZE,
        dim_feedforward=dim_feedforward,
        labels_num=NUM_CLASSES,
    )

    cls_results = train_classifier(
        model=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_path=classifier_path,
    )

    del classifier

    # ---- Step 8: Save training metadata ----
    meta = {
        "esm_model": args.esm_model,
        "esm_params": esm_info["params"],
        "d_model": d_model,
        "nhead": nhead,
        "num_encoder_layers": num_encoder_layers,
        "dim_feedforward": dim_feedforward,
        "max_len": WINDOW_SIZE,
        "labels_num": NUM_CLASSES,
        "class_labels": BGC_CLASS_LABELS,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "n_train": n_train,
        "n_val": n_val,
        "n_positive": n_pos,
        "n_negative": n_total - n_pos,
        "best_annotator_val_loss": ann_results["best_val_loss"],
        "best_classifier_val_loss": cls_results["best_val_loss"],
        "device": str(device),
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("  ESM2 model:              %s", args.esm_model)
    logger.info("  d_model:                 %d", d_model)
    logger.info("  Annotator val_loss:      %.4f", ann_results["best_val_loss"])
    logger.info("  Classifier val_loss:     %.4f", cls_results["best_val_loss"])
    logger.info("  Annotator saved to:      %s", annotator_path)
    logger.info("  Classifier saved to:     %s", classifier_path)
    logger.info("  Metadata saved to:       %s", meta_path)
    logger.info("")
    logger.info("To use these weights in the pipeline:")
    logger.info("  python main.py --esm-model %s ...", args.esm_model)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train BGC-Prophet annotator & classifier for ESM2 35M/150M/650M models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The default ESM2-8M model already ships with official pre-trained weights
at models/model/annotator.pt and classifier.pt. This script is for larger models.

Examples:
  # Train for ESM2-35M
  python train_prophet.py --esm-model esm2_t12_35M_UR50D

  # Train for ESM2-150M (default)
  python train_prophet.py --esm-model esm2_t30_150M_UR50D

  # Train for ESM2-650M on GPU with small batch (for limited VRAM)
  python train_prophet.py --esm-model esm2_t33_650M_UR50D --batch-size 4 --device cuda

  # Train all larger models sequentially
  for model in esm2_t12_35M_UR50D esm2_t30_150M_UR50D esm2_t33_650M_UR50D; do
    python train_prophet.py --esm-model $model
  done
        """,
    )

    parser.add_argument(
        "--esm-model",
        type=str,
        default="esm2_t12_35M_UR50D",
        choices=list(ESM2_REGISTRY.keys()),
        help="ESM2 model variant to train for (default: esm2_t12_35M_UR50D)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/training",
        help="Directory for training data (MIBiG downloads, cached embeddings)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/model",
        help="Output directory for trained weights (default: models/model)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for both embedding extraction and training (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto' (cuda>mps>cpu), 'cuda', 'mps', or 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker threads (default: 2)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split fraction (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=0,
        help="Number of negative windows (0=auto, matches positive count, min 500)",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=0,
        help=(
            "Limit MIBiG entries used for training (0 = use all). "
            "Useful for fast smoke-tests, e.g. --max-entries 50."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
