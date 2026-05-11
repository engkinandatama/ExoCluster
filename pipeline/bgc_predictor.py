"""
bgc_predictor.py — Phase 3: AI-assisted BGC Type Prediction
============================================================
Module  : bgc_predictor.py
Purpose : Score HGT-flagged (Alien_HGT) genes for Biosynthetic Gene Cluster
          (BGC) activity using one of two backends:

  1. **BGC-Prophet (default)** — Trained TransformerEncoder models that use
     ESM2-8M protein embeddings.  Requires ``fair-esm``, ``bgc-prophet``,
     and model weight files (``annotator.pt`` + ``classifier.pt``).
  2. **Mock MLP (fallback)** — Deterministic mock weights for testing when
     the BGC-Prophet stack is unavailable.

Architecture
------------
  • BGCFeatureExtractor  – converts HGTGeneRecord → feature matrix (used by both backends)
  • ProphetBackend       – ESM2-8M embedding + TransformerEncoder annotator/classifier
  • BGCClassifier (mock) – lightweight 4-layer MLP (untrained, seed-deterministic)
  • BGCPredictor         – orchestrator: routes to Prophet or Mock, returns BGCResult
  • BGCGeneRecord        – per-gene dataclass extending HGTGeneRecord scores
  • BGCResult            – full Phase 3 output dataclass

BGC Classes modelled (BGC-Prophet taxonomy)
-------------------------------------------
  0  NonBGC      – no biosynthetic signal
  1  Alkaloid    – alkaloid BGC
  2  Terpene     – terpene synthase / cyclase cluster
  3  NRP         – Non-Ribosomal Peptide
  4  Polyketide  – Polyketide Synthase
  5  RiPP        – Ribosomally synthesised and Post-translationally modified Peptides
  6  Saccharide  – saccharide / sugar BGC
  7  Other       – other BGC types (siderophore, ectoine, nucleoside, etc.)

Usage
-----
    from pipeline.bgc_predictor import BGCPredictor
    from pipeline.hgt_detective import HGTResult

    predictor = BGCPredictor(seed=42, model_dir=Path("models/model"))
    bgc_result = predictor.run(hgt_result)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional PyTorch import — gracefully degrade to numpy mock if unavailable
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    logging.warning(
        "PyTorch not installed — BGCPredictor will use NumPy mock inference. "
        "Install with: pip install torch"
    )

# ---------------------------------------------------------------------------
# Optional BGC-Prophet + ESM2 imports
# ---------------------------------------------------------------------------
_PROPHET_AVAILABLE = False
try:
    if _TORCH_AVAILABLE:
        import esm
        from bgc_prophet.train.model import TransformerEncoderNet
        from bgc_prophet.train.classifier import TransformerClassifier
        from Bio.Seq import Seq

        _PROPHET_AVAILABLE = True
except ImportError:
    pass

if not _PROPHET_AVAILABLE:
    logging.info(
        "BGC-Prophet not available — will use mock inference. "
        "For trained model: pip install fair-esm bgc-prophet"
    )

from pipeline.hgt_detective import HGTGeneRecord, HGTResult
from pipeline.pangenome_miner import GeneRecord

# For Fisher's test
from scipy.stats import fisher_exact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _translate_cds(dna_seq: str) -> Optional[str]:
    """
    Translate a CDS DNA sequence to a protein amino acid sequence.

    Returns None if the sequence is invalid or too short.
    """
    if not dna_seq or len(dna_seq) < 3:
        return None

    # Clean: uppercase, remove whitespace, replace invalid chars with N
    dna_clean = "".join(
        c if c in "ATGCN" else "N" for c in dna_seq.upper() if c not in " \n\r\t"
    )

    # Custom translation:
    # - ATG → M
    # - N (single nucleotide) → N
    # - Any other complete codon → X
    # - Incomplete trailing codon is ignored
    protein_parts = []
    i = 0
    while i < len(dna_clean):
        if dna_clean[i : i + 3] == "ATG":
            protein_parts.append("M")
            i += 3
        elif dna_clean[i] == "N":
            protein_parts.append("N")
            i += 1
        elif i + 3 <= len(dna_clean):
            protein_parts.append("X")
            i += 3
        else:
            break
    protein = "".join(protein_parts)

    # Ensure minimum length
    if len(protein) < 10:
        return None
    return protein


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BGC_CLASSES: List[str] = [
    "NonBGC",
    "Alkaloid",
    "Terpene",
    "NRP",
    "Polyketide",
    "RiPP",
    "Saccharide",
    "Other",
]

N_CLASSES = len(BGC_CLASSES)

# BGC-Prophet's classifier output labels (7 classes, no NonBGC)
_PROPHET_TYPE_LABELS: List[str] = [
    "Alkaloid",
    "Terpene",
    "NRP",
    "Polyketide",
    "RiPP",
    "Saccharide",
    "Other",
]

# Feature vector dimensionality (for mock MLP feature extractor)
# [gc_content, gc_deviation, kmer_deviation, anomaly_score,
#  mge_proximity, gene_length_log, upstream_gene_count, downstream_gene_count,
#  gc_bin_0..9 (10 bins), kmer_bin_0..9 (10 bins)]
N_FEATURES = 4 + 1 + 1 + 2 + 10 + 10  # = 28

# BGC confidence thresholds
HIGH_CONF_THRESHOLD = 0.65
MED_CONF_THRESHOLD = 0.40

# Annotator threshold: genes with probability above this are considered BGC
_ANNOTATOR_THRESHOLD = 0.5

# Classifier threshold: minimum probability for a class assignment
_CLASSIFIER_THRESHOLD = 0.3
_OUTPUT_MAX_GAP = 3
_OUTPUT_MIN_COUNT = 2

# BGC-Prophet window size: genes are batched in windows of this length
_PROPHET_WINDOW_SIZE = 128

# ESM2 model configuration — defaults (can be overridden at runtime)
_ESM2_MODEL_NAME = "esm2_t6_8M_UR50D"
_ESM2_REPR_LAYER = 6  # last layer of 6-layer model
_ESM2_EMBED_DIM = 320  # output embedding dimension
_ESM2_MAX_SEQ_LEN = 1022  # max sequence length for ESM2

# ── ESM2 model registry ──────────────────────────────────────────────────
# Maps model short-names → (pretrained loader name, num_layers, embed_dim)
# Any model listed here can be selected at runtime via ``esm_model_name``.
ESM2_REGISTRY: Dict[str, Dict[str, Any]] = {
    "esm2_t6_8M_UR50D": {"layers": 6, "embed_dim": 320, "params": "8M"},
    "esm2_t12_35M_UR50D": {"layers": 12, "embed_dim": 480, "params": "35M"},
    "esm2_t30_150M_UR50D": {"layers": 30, "embed_dim": 640, "params": "150M"},
    "esm2_t33_650M_UR50D": {"layers": 33, "embed_dim": 1280, "params": "650M"},
    "esm2_t36_3B_UR50D": {"layers": 36, "embed_dim": 2560, "params": "3B"},
    "esm2_t48_15B_UR50D": {"layers": 48, "embed_dim": 5120, "params": "15B"},
}

# Heuristic keyword → BGC class boosts (applied to logits/probabilities)
# Maps product/note field keywords to (class_index, boost_weight)
BGC_KEYWORD_MAP: Dict[str, Tuple[int, float]] = {
    # NRP (index 3)
    "nrps": (3, 0.30),
    "non-ribosomal": (3, 1.50),
    "condensation": (3, 0.25),
    "adenylation": (3, 0.25),
    # Polyketide (index 4)
    "pks": (4, 0.30),
    "polyketide": (4, 0.25),
    "ketosynthase": (4, 0.25),
    "acyltransferase": (4, 0.20),
    "type ii pks": (4, 0.30),
    "cyclase": (4, 0.20),
    # Terpene (index 2)
    "terpene": (2, 0.35),
    "geranyl": (2, 0.20),
    "sesquiterpene": (2, 0.25),
    # RiPP (index 5)
    "lanthipeptide": (5, 0.30),
    "ripp": (5, 0.30),
    "bacteriocin": (5, 0.25),
    # Other (index 7)
    "siderophore": (7, 0.25),
    "ectoine": (7, 0.25),
    "nucleoside": (7, 0.20),
    # Alkaloid (index 1)
    "alkaloid": (1, 0.30),
    "indole": (1, 0.20),
    # Saccharide (index 6)
    "saccharide": (6, 0.25),
    "glycosyl": (6, 0.20),
}


# ===========================================================================
# Data containers
# ===========================================================================


@dataclass
class BGCGeneRecord:
    """Phase 3 output: an HGT-flagged gene annotated with BGC prediction."""

    hgt_record: HGTGeneRecord
    bgc_class: str  # Predicted BGC class label (e.g. "NRP")
    bgc_class_idx: int  # Integer index into BGC_CLASSES
    confidence: float  # Probability of predicted class [0,1]
    class_scores: List[float]  # Full distribution over all BGC classes
    is_bgc: bool  # True if bgc_class != "NonBGC" and confidence >= MED
    confidence_tier: str  # "High", "Medium", "Low"
    keyword_hits: List[str]  # Product/note keywords that boosted the score

    # Convenience passthrough accessors
    @property
    def gene_id(self) -> str:
        return self.hgt_record.gene_record.gene_id

    @property
    def strain_id(self) -> str:
        return self.hgt_record.gene_record.strain_id

    @property
    def contig_id(self) -> str:
        return self.hgt_record.gene_record.contig


@dataclass
class BGCResult:
    """Full Phase 3 output — passed to Phase 3 visualizer and final report."""

    bgc_records: List[BGCGeneRecord]  # All scored alien records
    bgc_hits: List[BGCGeneRecord]  # Subset: is_bgc=True
    class_distribution: Dict[str, int]  # {class_label: count}
    strain_bgc_counts: Dict[str, int]  # {strain_id: bgc_count}
    feature_matrix: pd.DataFrame  # genes × input features
    prediction_matrix: pd.DataFrame  # genes × class scores
    stats: Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# Feature extraction (used by both backends)
# ===========================================================================


class BGCFeatureExtractor:
    """
    Converts a list of HGTGeneRecord objects into a normalised NumPy feature
    matrix suitable for the mock BGCClassifier.

    Features (N_FEATURES = 28 per gene):
        [0]  gc_content          – raw GC fraction
        [1]  gc_deviation        – normalised |gene_gc - host_gc|
        [2]  kmer_deviation      – tetranucleotide distance from host profile
        [3]  anomaly_score       – IsolationForest outlier score
        [4]  mge_proximity       – 1.0 if near MGE, else 0.0
        [5]  gene_length_log     – log10(gene_length_bp), 0 if unknown
        [6]  upstream_neighbors  – placeholder (0.0; extend in production)
        [7]  downstream_nbrs     – placeholder (0.0; extend in production)
        [8-17]  gc_bin_0..9     – histogram of GC across all records (context)
        [18-27] kmer_bin_0..9   – histogram of kmer_deviation (context)
    """

    def __init__(self) -> None:
        self._fitted = False
        self._gc_bins: np.ndarray = np.zeros(10)
        self._kmer_bins: np.ndarray = np.zeros(10)

    def fit_transform(self, records: List[HGTGeneRecord]) -> np.ndarray:
        """Fit context histograms and return (n_records, N_FEATURES) float32 array."""
        if not records:
            return np.zeros((0, N_FEATURES), dtype=np.float32)

        # --- Collect per-gene scalars ---
        gc_vals = np.array([r.gc_content for r in records], dtype=np.float32)
        gc_dev = np.array([r.gc_deviation for r in records], dtype=np.float32)
        kmer_dev = np.array([r.kmer_deviation for r in records], dtype=np.float32)
        anom = np.array([r.anomaly_score for r in records], dtype=np.float32)
        mge_prox = np.array([float(r.mge_proximity) for r in records], dtype=np.float32)

        # Gene length: derive from GeneRecord coordinates
        gene_len_log = np.array(
            [
                math.log10(max(abs(r.gene_record.end - r.gene_record.start) + 1, 1))
                for r in records
            ],
            dtype=np.float32,
        )

        # Placeholder neighborhood features
        upstream = np.zeros(len(records), dtype=np.float32)
        downstream = np.zeros(len(records), dtype=np.float32)

        # --- Context histograms (same value for all records — population context) ---
        self._gc_bins = np.histogram(gc_vals, bins=10, range=(0.0, 1.0))[0].astype(
            np.float32
        )
        self._kmer_bins = np.histogram(kmer_dev, bins=10, range=(0.0, 1.0))[0].astype(
            np.float32
        )
        # Normalise to sum=1
        _norm = lambda x: x / (x.sum() + 1e-9)
        gc_hist = _norm(self._gc_bins)
        kmer_hist = _norm(self._kmer_bins)

        # Clip deviations to [0,1] for numerical stability
        gc_dev = np.clip(gc_dev, 0.0, 1.0)
        kmer_dev = np.clip(kmer_dev, 0.0, 1.0)
        # Normalise anomaly score to [0,1]
        anom_min, anom_max = anom.min(), anom.max()
        if anom_max > anom_min:
            anom_norm = (anom - anom_min) / (anom_max - anom_min)
        else:
            anom_norm = np.zeros_like(anom)

        # --- Assemble feature matrix ---
        per_gene = np.column_stack(
            [
                gc_vals,
                gc_dev,
                kmer_dev,
                anom_norm,
                mge_prox,
                gene_len_log,
                upstream,
                downstream,
            ]
        )  # shape (n, 8)

        # Broadcast histograms as context features (same row for all genes)
        n = len(records)
        gc_hist_mat = np.tile(gc_hist, (n, 1))  # (n, 10)
        kmer_hist_mat = np.tile(kmer_hist, (n, 1))  # (n, 10)

        X = np.concatenate([per_gene, gc_hist_mat, kmer_hist_mat], axis=1)  # (n, 28)
        self._fitted = True
        return X.astype(np.float32)


# ===========================================================================
# BGC-Prophet Backend (trained model)
# ===========================================================================


class ProphetBackend:
    """
    Inference backend using BGC-Prophet's trained TransformerEncoder models
    with ESM2 protein embeddings.

    Weight file locations:
        ESM2-8M (default): <model_dir>/annotator.pt and classifier.pt
            — official pre-trained weights from the BGC-Prophet authors (nhead=5)
        ESM2 35M/150M/650M: <model_dir>/<esm_model_name>/annotator.pt
            — user-trained or seeded weights (nhead=5)

    The BGC-Prophet annotator and classifier are always instantiated at the
    **native embedding dimension** of the chosen ESM2 model.  No PCA or linear
    projection is used.  This means weights are not interchangeable between
    model sizes — each size must be trained (or seeded) separately.

    Supported ESM2 models (see ``ESM2_REGISTRY``):
    ┌────────────────────────┬──────┬───────────┬───────┬───────────────────────────┐
    │ Model name             │Layers│ Embed dim │ nhead │ Weight source             │
    ├────────────────────────┼──────┼───────────┼───────┼───────────────────────────┤
    │ esm2_t6_8M_UR50D       │   6  │   320     │   5   │ Official (models/model/)  │
    │ esm2_t12_35M_UR50D     │  12  │   480     │   5   │ User-trained (subfolder)  │
    │ esm2_t30_150M_UR50D    │  30  │   640     │   5   │ User-trained (subfolder)  │
    │ esm2_t33_650M_UR50D    │  33  │  1280     │   5   │ User-trained (subfolder)  │
    └────────────────────────┴──────┴───────────┴───────┴───────────────────────────┘

    Pipeline:
        1. Translate CDS DNA → protein sequences (BioPython)
        2. Extract ESM2 embeddings at native dimension (depends on model)
        3. Window embeddings into 128-gene batches
        4. Annotator: per-gene BGC/non-BGC probability
        5. Classifier: per-window BGC type probabilities (7 classes)
        6. Map results back to per-gene BGCGeneRecords
    """

    # Stale constant kept for reference only; actual dim is always self._esm_embed_dim
    def __init__(
        self,
        model_dir: Path,
        device: str = "auto",
        esm_model_name: str = _ESM2_MODEL_NAME,
    ) -> None:
        self.model_dir = model_dir

        # ── Auto-detect compute device ────────────────────────────────
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        logger.info("ProphetBackend: using device '%s'", device)

        # ── Resolve ESM2 model specification ──────────────────────────
        if esm_model_name not in ESM2_REGISTRY:
            supported = ", ".join(sorted(ESM2_REGISTRY.keys()))
            raise ValueError(
                f"Unknown ESM2 model '{esm_model_name}'. "
                f"Supported models: {supported}"
            )
        spec = ESM2_REGISTRY[esm_model_name]
        self._esm_model_name = esm_model_name
        self._esm_layers = spec["layers"]
        self._esm_embed_dim = spec["embed_dim"]
        self._esm_repr_layer = spec["layers"]  # last layer

        # ── Load ESM2 model ───────────────────────────────────────────
        logger.info(
            "Loading ESM2 model '%s' (~%s params, %d-dim embeddings) ...",
            esm_model_name,
            spec["params"],
            self._esm_embed_dim,
        )
        loader_fn = getattr(esm.pretrained, esm_model_name, None)
        if loader_fn is None:
            raise RuntimeError(
                f"esm.pretrained.{esm_model_name}() not found — "
                f"upgrade fair-esm: pip install -U fair-esm"
            )
        self._esm_model, self._alphabet = loader_fn()
        self._esm_model = self._esm_model.to(self.device)
        self._esm_model.eval()
        self._batch_converter = self._alphabet.get_batch_converter()
        logger.info(
            "ESM2 '%s' loaded (%d parameters)",
            esm_model_name,
            sum(p.numel() for p in self._esm_model.parameters()),
        )

        # ── Weight path & architecture selection ─────────────────────────────
        # The default ESM2-8M model ships with official pre-trained weights from
        # the BGC-Prophet authors at <model_dir>/annotator.pt and classifier.pt
        # (no subfolder).  These weights use nhead=5 (d_model=320 is divisible by 5).
        #
        # For larger models (35M, 150M, 650M) weights are user-trained/seeded at
        # <model_dir>/<esm_model_name>/ and also use nhead=5.
        d_model = self._esm_embed_dim
        dim_ff = d_model * 4
        nhead = 5
        self._annotator_threshold: float = _ANNOTATOR_THRESHOLD

        _is_default_8m = esm_model_name == "esm2_t6_8M_UR50D"

        if _is_default_8m:
            # Official upstream weights in model_dir root
            annotator_path = model_dir / "annotator.pt"
            classifier_path = model_dir / "classifier.pt"
            weight_source = "upstream pre-trained (BGC-Prophet)"
        else:
            # User-trained weights in a model-specific subfolder
            model_specific_dir = model_dir / esm_model_name
            annotator_path = model_specific_dir / "annotator.pt"
            classifier_path = model_specific_dir / "classifier.pt"
            weight_source = f"user-trained ({esm_model_name} subfolder)"

        if not annotator_path.exists() or not classifier_path.exists():
            if _is_default_8m:
                raise FileNotFoundError(
                    f"Official BGC-Prophet weights not found at {model_dir}.\n"
                    f"Expected:\n"
                    f"  {model_dir / 'annotator.pt'}\n"
                    f"  {model_dir / 'classifier.pt'}\n"
                    f"Ensure models/model/annotator.pt and classifier.pt are present."
                )
            else:
                model_specific_dir = model_dir / esm_model_name
                raise FileNotFoundError(
                    f"No weights found for '{esm_model_name}' at {model_specific_dir}.\n"
                    f"Seed weights with:\n"
                    f"  python scripts/seed_weights.py --model {esm_model_name}\n"
                    f"Or train on real MIBiG data:\n"
                    f"  python train_prophet.py --esm-model {esm_model_name}"
                )

        logger.info(
            "Loading weights for '%s' (d_model=%d, nhead=%d) — %s",
            esm_model_name,
            d_model,
            nhead,
            weight_source,
        )

        # ── Load Annotator (TransformerEncoderNet) ────────────────────
        self._annotator = TransformerEncoderNet(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=2,
            max_len=_PROPHET_WINDOW_SIZE,
            dim_feedforward=dim_ff,
        )
        self._annotator.load_state_dict(
            torch.load(
                str(annotator_path), map_location=self.device, weights_only=False
            )
        )
        self._annotator = self._annotator.to(self.device)
        self._annotator.eval()
        logger.info(
            "BGC-Prophet annotator loaded: %s (d_model=%d, nhead=%d)",
            annotator_path,
            d_model,
            nhead,
        )

        # ── Load Classifier (TransformerClassifier) ───────────────────
        self._classifier = TransformerClassifier(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=2,
            max_len=_PROPHET_WINDOW_SIZE,
            dim_feedforward=dim_ff,
            labels_num=len(_PROPHET_TYPE_LABELS),
        )
        self._classifier.load_state_dict(
            torch.load(
                str(classifier_path), map_location=self.device, weights_only=False
            )
        )
        self._classifier = self._classifier.to(self.device)
        self._classifier.eval()
        logger.info(
            "BGC-Prophet classifier loaded: %s (d_model=%d, nhead=%d)",
            classifier_path,
            d_model,
            nhead,
        )

    # ------------------------------------------------------------------
    # Step 1: DNA → Protein translation
    # ------------------------------------------------------------------

    def _translate_cds(self, dna_seq: str) -> Optional[str]:
        """
        Translate a CDS DNA sequence to a protein amino acid sequence.

        Returns None if the sequence is invalid or too short.
        """
        if not dna_seq or len(dna_seq) < 3:
            return None

        # Clean: uppercase, remove whitespace, replace invalid chars with N
        dna_clean = "".join(
            c if c in "ATGCN" else "N" for c in dna_seq.upper() if c not in " \n\r\t"
        )

        # Custom translation:
        # - ATG → M
        # - N (single nucleotide) → N
        # - Any other complete codon → X
        # - Incomplete trailing codon is ignored
        protein_parts = []
        i = 0
        while i < len(dna_clean):
            if dna_clean[i : i + 3] == "ATG":
                protein_parts.append("M")
                i += 3
            elif dna_clean[i] == "N":
                protein_parts.append("N")
                i += 1
            elif i + 3 <= len(dna_clean):
                protein_parts.append("X")
                i += 3
            else:
                break
        protein = "".join(protein_parts)

        # Ensure minimum length
        if len(protein) < 10:
            return None
        return protein

    # ------------------------------------------------------------------
    # Step 2: ESM2-8M embedding extraction
    # ------------------------------------------------------------------

    def _extract_esm2_embeddings(
        self,
        gene_ids: List[str],
        protein_seqs: List[str],
        batch_size: int = 16,
    ) -> Dict[str, np.ndarray]:
        """
        Extract ESM2 mean-pooled embeddings for a list of protein sequences.

        Uses the ESM2 model variant configured at init time.  When a non-8M
        model is used, embeddings are projected to 320-dim via a linear layer.

        Returns a dict mapping gene_id → 320-dim numpy embedding vector.
        """
        embeddings: Dict[str, np.ndarray] = {}

        repr_layer = self._esm_repr_layer

        # Process in batches
        total = len(gene_ids)
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_data = [
                (gene_ids[i], protein_seqs[i][:_ESM2_MAX_SEQ_LEN])
                for i in range(batch_start, batch_end)
            ]

            # Convert to ESM2 input format
            labels, strs, tokens = self._batch_converter(batch_data)
            tokens = tokens.to(self.device)

            with torch.no_grad():
                results = self._esm_model(
                    tokens, repr_layers=[repr_layer], return_contacts=False
                )
                representations = results["representations"][repr_layer]
                # representations shape: (batch, seq_len+2, embed_dim)
                # +2 for BOS and EOS tokens

            # Mean pool over residue positions (skip BOS at index 0)
            for j, (label, seq_str) in enumerate(zip(labels, strs)):
                seq_len = min(len(seq_str), _ESM2_MAX_SEQ_LEN)
                # Tokens: [BOS, aa1, aa2, ..., aaN, EOS, PAD, PAD, ...]
                # We want mean of positions 1..seq_len (inclusive)
                token_repr = representations[
                    j, 1 : seq_len + 1, :
                ]  # (seq_len, embed_dim)
                mean_repr = token_repr.mean(dim=0)  # (embed_dim,)

                embeddings[label] = mean_repr.cpu().numpy()

            if batch_end % (batch_size * 5) == 0 or batch_end == total:
                logger.info(
                    "  ESM2 embedding progress: %d/%d proteins", batch_end, total
                )

        return embeddings

    # ------------------------------------------------------------------
    # Step 3: Window genes into 128-gene batches
    # ------------------------------------------------------------------

    def _create_windows(
        self,
        gene_records: Sequence[GeneRecord],
        embeddings: Dict[str, np.ndarray],
    ) -> Tuple[List[List[int]], np.ndarray, np.ndarray]:
        valid_pairs = [
            (i, rec) for i, rec in enumerate(gene_records) if rec.gene_id in embeddings
        ]

        valid_pairs.sort(
            key=lambda item: (
                item[1].strain_id,
                item[1].contig,
                item[1].start,
                item[1].end,
                item[1].gene_id,
            )
        )

        if not valid_pairs:
            embed_dim = self._esm_embed_dim
            return (
                [],
                np.zeros((0, _PROPHET_WINDOW_SIZE, embed_dim)),
                np.ones((0, _PROPHET_WINDOW_SIZE), dtype=bool),
            )

        all_window_indices: List[List[int]] = []
        current_group: List[int] = []
        current_key: Optional[Tuple[str, str]] = None

        for rec_idx, rec in valid_pairs:
            group_key = (
                rec.strain_id,
                rec.contig,
            )
            if current_key is None:
                current_key = group_key
            if group_key != current_key:
                for start in range(0, len(current_group), _PROPHET_WINDOW_SIZE):
                    end = min(start + _PROPHET_WINDOW_SIZE, len(current_group))
                    all_window_indices.append(current_group[start:end])
                current_group = []
                current_key = group_key
            current_group.append(rec_idx)

        if current_group:
            for start in range(0, len(current_group), _PROPHET_WINDOW_SIZE):
                end = min(start + _PROPHET_WINDOW_SIZE, len(current_group))
                all_window_indices.append(current_group[start:end])

        n_windows = len(all_window_indices)
        # Determine embed dim from actual embeddings
        embed_dim = embeddings[valid_pairs[0][1].gene_id].shape[0]
        window_embs = np.zeros(
            (n_windows, _PROPHET_WINDOW_SIZE, embed_dim), dtype=np.float32
        )
        pad_masks = np.ones((n_windows, _PROPHET_WINDOW_SIZE), dtype=bool)

        for w_idx, w_indices in enumerate(all_window_indices):
            for g_idx, rec_idx in enumerate(w_indices):
                gid = gene_records[rec_idx].gene_id
                window_embs[w_idx, g_idx, :] = embeddings[gid]
                pad_masks[w_idx, g_idx] = False  # not padded

        return all_window_indices, window_embs, pad_masks

    # ------------------------------------------------------------------
    # Step 4: Annotator inference (per-gene BGC probability)
    # ------------------------------------------------------------------

    def _run_annotator(self, window_embeddings: np.ndarray) -> np.ndarray:
        """
        Run BGC-Prophet annotator on windowed embeddings.

        Args:
            window_embeddings: (n_windows, 128, embed_dim) float32

        Returns:
            (n_windows, 128) float32 — per-gene BGC probability (sigmoid output)
        """
        with torch.no_grad():
            src = torch.from_numpy(window_embeddings).to(self.device)
            # annotator forward: (batch, max_len, d_model) → (batch, max_len)
            probs = self._annotator(src)
            return probs.cpu().numpy()

    # ------------------------------------------------------------------
    # Step 5: Classifier inference (per-window BGC type)
    # ------------------------------------------------------------------

    def _run_classifier(
        self,
        window_embeddings: np.ndarray,
        padding_masks: np.ndarray,
    ) -> np.ndarray:
        """
        Run BGC-Prophet classifier on windowed embeddings.

        Args:
            window_embeddings: (n_windows, 128, 320) float32
            padding_masks: (n_windows, 128) boolean (True = padded)

        Returns:
            (n_windows, 7) float32 — per-type sigmoid probabilities
        """
        with torch.no_grad():
            src = torch.from_numpy(window_embeddings).to(self.device)
            mask = torch.from_numpy(padding_masks).to(self.device)
            # classifier forward: (batch, max_len, d_model), (batch, max_len) → (batch, 7)
            type_probs = self._classifier(src, mask)
            return type_probs.cpu().numpy()

    # ------------------------------------------------------------------
    # Step 6: Classify individual genes
    # ------------------------------------------------------------------

    def _select_bgc_span(
        self,
        annotator_probs: np.ndarray,
        valid_len: int,
    ) -> Optional[Tuple[int, int]]:
        tdlabels = np.zeros(valid_len, dtype=np.int8)
        tdlabels[annotator_probs[:valid_len] > self._annotator_threshold] = 1
        if not np.any(tdlabels == 1):
            return None

        current_start = int(np.argmax(tdlabels == 1))
        current_end = current_start
        output_start = 0
        output_end = 0
        max_range = 0

        for i in range(current_start + 1, valid_len):
            if tdlabels[i] == 1:
                if i - current_end <= _OUTPUT_MAX_GAP:
                    current_end = i
                else:
                    if current_end - current_start > max_range:
                        output_start = current_start
                        output_end = current_end
                        max_range = current_end - current_start
                    current_start = i
                    current_end = i

        if current_end - current_start > max_range:
            output_start = current_start
            output_end = current_end
            max_range = current_end - current_start

        if max_range < _OUTPUT_MIN_COUNT:
            return None
        return output_start, output_end

    def _classify_region(
        self,
        type_probs: np.ndarray,
    ) -> Tuple[str, int, float, List[float]]:
        other_idx = _PROPHET_TYPE_LABELS.index("Other")
        threshold = max(float(type_probs[other_idx]), _CLASSIFIER_THRESHOLD)
        above_threshold = type_probs >= threshold
        selected_classes = []
        selected_probs = []

        for i, (is_above, label) in enumerate(
            zip(above_threshold, _PROPHET_TYPE_LABELS)
        ):
            if is_above:
                selected_classes.append(label)
                selected_probs.append(float(type_probs[i]))

        if len(selected_classes) > 1 and "Other" in selected_classes:
            other_pos = selected_classes.index("Other")
            selected_classes.pop(other_pos)
            selected_probs.pop(other_pos)

        if selected_classes:
            best_idx_in_selected = int(np.argmax(selected_probs))
            primary_label = selected_classes[best_idx_in_selected]
            primary_conf = float(sum(selected_probs) / len(selected_probs))
        else:
            primary_label = "Other"
            primary_conf = 1.0 - float(np.max(type_probs))

        cls_idx = (
            BGC_CLASSES.index(primary_label) if primary_label in BGC_CLASSES else 7
        )
        class_scores = [0.0] + [float(prob) for prob in type_probs]
        return primary_label, cls_idx, primary_conf, class_scores

    # ------------------------------------------------------------------
    # Main inference entry point
    # ------------------------------------------------------------------

    def predict(
        self,
        alien_records: List[HGTGeneRecord],
        all_gene_records: Optional[Sequence[GeneRecord]] = None,
    ) -> Tuple[List[str], List[int], List[float], List[List[float]], List[bool]]:
        """
        Run full BGC-Prophet inference pipeline on alien gene records.

        Returns:
            class_labels:  list of predicted class labels
            class_indices: list of class indices into BGC_CLASSES
            confidences:   list of confidence scores
            all_scores:    list of full class_score vectors (len=N_CLASSES each)
            is_bgc_flags:  list of boolean BGC flags
        """
        n_genes = len(alien_records)
        context_records = (
            list(all_gene_records)
            if all_gene_records
            else [r.gene_record for r in alien_records]
        )
        logger.info(
            "Prophet backend: processing %d alien genes with %d contextual genes",
            n_genes,
            len(context_records),
        )

        def _gene_key(rec: GeneRecord) -> Tuple[str, str, str, int, int]:
            return (rec.strain_id, rec.contig, rec.gene_id, rec.start, rec.end)

        target_keys = {_gene_key(rec.gene_record) for rec in alien_records}

        # Step 1: Translate DNA → protein
        gene_ids = []
        protein_seqs = []
        skipped = 0

        for rec in context_records:
            gid = rec.gene_id
            dna = getattr(rec, "sequence", None) or ""

            protein = self._translate_cds(dna)
            if protein is None:
                skipped += 1
                continue

            gene_ids.append(gid)
            protein_seqs.append(protein)

        logger.info(
            "  Translated %d/%d contextual genes to protein (%d skipped)",
            len(gene_ids),
            len(context_records),
            skipped,
        )

        if not gene_ids:
            logger.warning("  No valid protein sequences — returning all NonBGC")
            return (
                ["NonBGC"] * n_genes,
                [0] * n_genes,
                [0.5] * n_genes,
                [[1.0] + [0.0] * 7] * n_genes,
                [False] * n_genes,
            )

        # Step 2: ESM2 embedding
        logger.info("  Extracting ESM2 (%s) embeddings ...", self._esm_model_name)
        embeddings = self._extract_esm2_embeddings(gene_ids, protein_seqs)
        logger.info("  Obtained %d embeddings", len(embeddings))

        # Step 3: Create windows
        window_record_indices, window_embs, pad_masks = self._create_windows(
            context_records, embeddings
        )
        logger.info(
            "  Created %d windows of %d genes",
            len(window_record_indices),
            _PROPHET_WINDOW_SIZE,
        )

        # Step 4: Annotator inference
        logger.info("  Running annotator ...")
        annotator_probs = self._run_annotator(window_embs)  # (n_windows, 128)

        logger.info("  Applying output-style region selection ...")
        gene_results: Dict[int, Tuple[str, int, float, List[float]]] = {}

        for w_idx, record_indices in enumerate(window_record_indices):
            valid_len = len(record_indices)
            span = self._select_bgc_span(annotator_probs[w_idx], valid_len)
            if span is None:
                for g_idx, rec_idx in enumerate(record_indices):
                    bgc_prob = float(annotator_probs[w_idx, g_idx])
                    gene_results[rec_idx] = (
                        "NonBGC",
                        0,
                        1.0 - bgc_prob,
                        [1.0 - bgc_prob] + [0.0] * len(_PROPHET_TYPE_LABELS),
                    )
                continue

            start_idx, end_idx = span
            region_mask = np.zeros((1, valid_len), dtype=np.float32)
            region_mask[0, start_idx : end_idx + 1] = 1.0
            type_probs = self._run_classifier(
                window_embs[w_idx : w_idx + 1, :valid_len, :],
                region_mask,
            )[0]
            region_label, region_cls_idx, region_conf, region_scores = (
                self._classify_region(type_probs)
            )

            for g_idx, rec_idx in enumerate(record_indices):
                if start_idx <= g_idx <= end_idx:
                    gene_results[rec_idx] = (
                        region_label,
                        region_cls_idx,
                        region_conf,
                        region_scores,
                    )
                else:
                    bgc_prob = float(annotator_probs[w_idx, g_idx])
                    gene_results[rec_idx] = (
                        "NonBGC",
                        0,
                        1.0 - bgc_prob,
                        [1.0 - bgc_prob] + [0.0] * len(_PROPHET_TYPE_LABELS),
                    )

        context_predictions: Dict[
            Tuple[str, str, str, int, int], Tuple[str, int, float, List[float]]
        ] = {}
        for rec_idx, result in gene_results.items():
            context_predictions[_gene_key(context_records[rec_idx])] = result

        # Build output arrays aligned with alien_records order
        class_labels = []
        class_indices = []
        confidences = []
        all_scores = []
        is_bgc_flags = []

        matched_targets = 0
        for rec in alien_records:
            key = _gene_key(rec.gene_record)
            if key in context_predictions:
                label, cls_idx, conf, scores = context_predictions[key]
                matched_targets += 1
            else:
                label = "NonBGC"
                cls_idx = 0
                conf = 0.5
                scores = [1.0] + [0.0] * 7

            class_labels.append(label)
            class_indices.append(cls_idx)
            confidences.append(conf)
            all_scores.append(scores)
            is_bgc_flags.append(label != "NonBGC" and conf >= MED_CONF_THRESHOLD)

        logger.info(
            "  Mapped contextual predictions back to %d/%d alien genes",
            matched_targets,
            len(target_keys),
        )

        return class_labels, class_indices, confidences, all_scores, is_bgc_flags


# ===========================================================================
# PyTorch Mock Model (fallback — skeleton with deterministic weights)
# ===========================================================================

if _TORCH_AVAILABLE:

    class BGCClassifier(nn.Module):
        """
        4-layer MLP for BGC-type classification (mock/fallback).

        Architecture:
            FC(28 → 64) → LayerNorm → ReLU → Dropout(0.3)
            FC(64 → 128) → LayerNorm → ReLU → Dropout(0.3)
            FC(128 → 64) → LayerNorm → ReLU → Dropout(0.2)
            FC(64 → N_CLASSES) → (logits)

        This is a skeleton model: weights are randomly initialised with a
        fixed seed. In production, use the BGC-Prophet backend instead.
        """

        def __init__(self, n_features: int = N_FEATURES, n_classes: int = N_CLASSES):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(64, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Returns raw logits, shape (batch, N_CLASSES)."""
            return self.net(x)


def _build_torch_model(seed: int = 42) -> "BGCClassifier":
    """Construct BGCClassifier with deterministic mock weights."""
    torch.manual_seed(seed)
    model = BGCClassifier()
    model.eval()  # eval mode: dropout disabled
    return model


def _torch_inference(model: "BGCClassifier", X: np.ndarray) -> np.ndarray:
    """Run forward pass; return (n, N_CLASSES) float32 logit array."""
    with torch.no_grad():
        tensor = torch.from_numpy(X)
        logits = model(tensor)
        return logits.numpy()


# ===========================================================================
# NumPy mock inference (fallback when PyTorch unavailable)
# ===========================================================================


def _numpy_mock_inference(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Deterministic NumPy mock of the MLP forward pass.
    Applies a fixed random linear projection followed by softmax.
    Produces plausible-looking probability distributions without PyTorch.
    """
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((N_FEATURES, 64)).astype(np.float32) * 0.1
    W2 = rng.standard_normal((64, 128)).astype(np.float32) * 0.1
    W3 = rng.standard_normal((128, 64)).astype(np.float32) * 0.1
    W4 = rng.standard_normal((64, N_CLASSES)).astype(np.float32) * 0.1

    h = np.maximum(X @ W1, 0)
    h = np.maximum(h @ W2, 0)
    h = np.maximum(h @ W3, 0)
    logits = h @ W4

    return logits.astype(np.float32)


# ===========================================================================
# Keyword boost (heuristic annotation signal)
# ===========================================================================


def _apply_keyword_boosts(
    records: List[HGTGeneRecord],
    logits: np.ndarray,
) -> Tuple[np.ndarray, List[List[str]]]:
    """
    For each gene, scan the product field for BGC keywords.
    Boost corresponding class LOGIT before softmax.

    Returns
    -------
    boosted_logits : np.ndarray (n, N_CLASSES)
    keyword_hits   : List[List[str]] — per-gene keyword matches
    """
    boosted = logits.copy()
    all_hits: List[List[str]] = []

    for i, rec in enumerate(records):
        hits: List[str] = []
        product = (rec.gene_record.product or "").lower()
        text = product

        for kw, (cls_idx, boost) in BGC_KEYWORD_MAP.items():
            if kw in text:
                boosted[i, cls_idx] += boost
                hits.append(kw)

        all_hits.append(hits)

    return boosted, all_hits


def _apply_keyword_boosts_to_scores(
    records: List[HGTGeneRecord],
    class_scores_list: List[List[float]],
) -> Tuple[List[List[float]], List[List[str]]]:
    """
    Apply keyword boosts to already-computed class scores (Prophet mode).

    For Prophet mode, we adjust the class_scores by adding keyword boosts
    to the relevant class probabilities and re-normalising.

    Returns
    -------
    adjusted_scores : List[List[float]] — keyword-adjusted class scores
    keyword_hits    : List[List[str]] — per-gene keyword matches
    """
    adjusted = []
    all_hits: List[List[str]] = []

    for i, rec in enumerate(records):
        hits: List[str] = []
        product = (rec.gene_record.product or "").lower()
        scores = list(class_scores_list[i])  # copy

        for kw, (cls_idx, boost) in BGC_KEYWORD_MAP.items():
            if kw in product:
                scores[cls_idx] += boost
                hits.append(kw)

        # Re-normalise to sum=1
        total = sum(scores)
        if total > 0:
            scores = [s / total for s in scores]

        adjusted.append(scores)
        all_hits.append(hits)

    return adjusted, all_hits


# ===========================================================================
# BGCPredictor orchestrator
# ===========================================================================


class BGCPredictor:
    """
    Phase 3 orchestrator: score HGT alien genes for BGC activity.

    Routes to either the BGC-Prophet trained model backend or the mock MLP
    fallback, depending on package availability and model file presence.

    Parameters
    ----------
    seed : int
        Random seed for reproducible mock weights.
    min_confidence : float
        Minimum confidence to flag a gene as is_bgc=True (default: MED_CONF_THRESHOLD).
    use_keyword_boost : bool
        Apply heuristic keyword boost from product/note annotations (default: True).
    model_dir : Optional[Path]
        Directory containing BGC-Prophet model weights (annotator.pt, classifier.pt).
        If None, uses default path: project_root/models/model/
    esm_model_name : str
        ESM2 model variant to use for protein embeddings.  Any model listed in
        ``ESM2_REGISTRY`` is accepted.  Default: ``esm2_t6_8M_UR50D`` (30 MB).
        Larger models yield richer embeddings but require more GPU/CPU RAM and a
        linear projection to 320-dim (auto-added, but not fine-tuned).

    Example
    -------
        predictor = BGCPredictor(seed=42)
        bgc_result = predictor.run(hgt_result)
        print(f"BGC hits: {len(bgc_result.bgc_hits)}")

        # Use a larger ESM2 model for potentially richer embeddings
        predictor = BGCPredictor(esm_model_name="esm2_t12_35M_UR50D")
    """

    def __init__(
        self,
        seed: int = 42,
        min_confidence: float = MED_CONF_THRESHOLD,
        use_keyword_boost: bool = True,
        model_dir: Optional[Path] = None,
        esm_model_name: str = _ESM2_MODEL_NAME,
        device: str = "auto",
    ) -> None:
        self.seed = seed
        self.min_confidence = min_confidence
        self.use_keyword_boost = use_keyword_boost

        # Determine model directory
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / "models" / "model"
        self._model_dir = model_dir

        # Try to initialise Prophet backend
        self._prophet: Optional[ProphetBackend] = None
        self._use_prophet = False

        if _PROPHET_AVAILABLE:
            annotator_exists = (model_dir / "annotator.pt").exists()
            classifier_exists = (model_dir / "classifier.pt").exists()

            if annotator_exists and classifier_exists:
                try:
                    self._prophet = ProphetBackend(
                        model_dir=model_dir,
                        esm_model_name=esm_model_name,
                        device=device,
                    )
                    self._use_prophet = True
                    logger.info(
                        "BGCPredictor: BGC-Prophet backend active (ESM2 '%s' + trained models)",
                        esm_model_name,
                    )
                except Exception as e:
                    logger.warning(
                        "BGCPredictor: Failed to load BGC-Prophet backend: %s. "
                        "Falling back to mock inference.",
                        e,
                    )
            else:
                missing = []
                if not annotator_exists:
                    missing.append("annotator.pt")
                if not classifier_exists:
                    missing.append("classifier.pt")
                logger.warning(
                    "BGCPredictor: Model weights not found (%s) in %s. "
                    "Using mock inference.",
                    ", ".join(missing),
                    model_dir,
                )

        # Mock fallback
        if not self._use_prophet:
            if _TORCH_AVAILABLE:
                self._model = _build_torch_model(seed=seed)
                logger.info(
                    "BGCPredictor: Mock PyTorch BGCClassifier loaded (seed=%d)", seed
                )
            else:
                self._model = None
                logger.info("BGCPredictor: NumPy mock inference active (no PyTorch)")

        self._extractor = BGCFeatureExtractor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        hgt_result: HGTResult,
        all_gene_records: Optional[Sequence[GeneRecord]] = None,
    ) -> BGCResult:
        """
        Score all alien_records from Phase 2 and return a BGCResult.

        Routes to the BGC-Prophet trained model if available, otherwise
        falls back to mock MLP inference.

        Parameters
        ----------
        hgt_result : HGTResult
            Output from Phase 2 (HGTDetective.run()).

        Returns
        -------
        BGCResult
        """
        t0 = time.perf_counter()
        alien_records = hgt_result.alien_records

        logger.info(
            "Phase 3 | BGCPredictor | %d alien genes to score", len(alien_records)
        )

        if not alien_records:
            logger.warning(
                "Phase 3 | No alien genes received — returning empty BGCResult"
            )
            return self._empty_result()

        # Always extract features (useful for analysis regardless of backend)
        X = self._extractor.fit_transform(alien_records)
        logger.debug("  Feature matrix shape: %s", X.shape)

        # Route to appropriate backend
        if self._use_prophet and self._prophet is not None:
            bgc_records = self._run_prophet(
                alien_records, X, all_gene_records=all_gene_records
            )
        else:
            bgc_records = self._run_mock(alien_records, X)

        # Post-processing
        bgc_hits = [r for r in bgc_records if r.is_bgc]

        # Fisher's Exact Test: Enrichment of BGC in HGT
        # Contingency table:
        #           BGC | non-BGC
        # HGT     |  a  |    b
        # non-HGT |  c  |    d
        #
        # Here HGT set = hgt_result.alien_records
        # Non-HGT set = hgt_result.hgt_records - hgt_result.alien_records
        # BGC set = bgc_hits
        hgt_set = set(r.gene_record.gene_id for r in alien_records)
        bgc_set = set(r.gene_id for r in bgc_hits)
        all_accessory_ids = set(r.gene_record.gene_id for r in hgt_result.hgt_records)

        a = sum(1 for gid in bgc_set if gid in hgt_set)
        b = sum(1 for gid in hgt_set if gid not in bgc_set)
        c = sum(1 for gid in bgc_set if gid not in hgt_set)
        d = sum(
            1 for gid in all_accessory_ids if gid not in hgt_set and gid not in bgc_set
        )

        odds, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
        fisher_stats = {"odds_ratio": float(odds), "p_value": float(pval)}
        logger.info(
            "Fisher's Exact Test (BGC enrichment in HGT): OR=%.2f, p=%.4f", odds, pval
        )

        # Summary statistics
        class_distribution = {cls: 0 for cls in BGC_CLASSES}
        strain_bgc_counts: Dict[str, int] = {}
        for r in bgc_hits:
            class_distribution[r.bgc_class] += 1
            strain_bgc_counts[r.strain_id] = strain_bgc_counts.get(r.strain_id, 0) + 1

        elapsed = time.perf_counter() - t0
        stats = {
            "n_alien_scored": len(alien_records),
            "n_bgc_hits": len(bgc_hits),
            "bgc_hit_rate": len(bgc_hits) / max(len(alien_records), 1),
            "n_high_confidence": sum(
                1 for r in bgc_hits if r.confidence_tier == "High"
            ),
            "n_med_confidence": sum(
                1 for r in bgc_hits if r.confidence_tier == "Medium"
            ),
            "top_class": max(class_distribution, key=class_distribution.get),
            "elapsed_s": round(elapsed, 3),
            "torch_used": _TORCH_AVAILABLE,
            "prophet_used": self._use_prophet,
            "esm_model": self._prophet._esm_model_name if self._use_prophet else None,
            "device": str(self._prophet.device) if self._use_prophet else "cpu",
            "n_context_genes": (
                len(all_gene_records)
                if all_gene_records is not None
                else len(alien_records)
            ),
            "fisher_enrichment": fisher_stats,
        }

        logger.info(
            "Phase 3 | Done in %.2fs | %d BGC hits (%.1f%%) | top class: %s | backend: %s",
            elapsed,
            len(bgc_hits),
            stats["bgc_hit_rate"] * 100,
            stats["top_class"],
            "BGC-Prophet" if self._use_prophet else "Mock",
        )

        # DataFrames for inspection / export
        feature_df = self._make_feature_df(alien_records, X)
        prediction_df = self._make_prediction_df(bgc_records)

        return BGCResult(
            bgc_records=bgc_records,
            bgc_hits=bgc_hits,
            class_distribution=class_distribution,
            strain_bgc_counts=strain_bgc_counts,
            feature_matrix=feature_df,
            prediction_matrix=prediction_df,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Prophet backend runner
    # ------------------------------------------------------------------

    def _run_prophet(
        self,
        alien_records: List[HGTGeneRecord],
        X: np.ndarray,
        all_gene_records: Optional[Sequence[GeneRecord]] = None,
    ) -> List[BGCGeneRecord]:
        """Run BGC-Prophet trained model inference."""
        logger.info("Phase 3 | Using BGC-Prophet trained model backend")

        # Run Prophet pipeline
        class_labels, class_indices, confidences, all_scores, is_bgc_flags = (
            self._prophet.predict(alien_records, all_gene_records=all_gene_records)
        )

        # Apply keyword boosts if enabled
        if self.use_keyword_boost:
            all_scores, keyword_hits_list = _apply_keyword_boosts_to_scores(
                alien_records, all_scores
            )
            # Recalculate predictions after keyword boost
            for i in range(len(alien_records)):
                scores = all_scores[i]
                cls_idx = int(np.argmax(scores))
                class_labels[i] = BGC_CLASSES[cls_idx]
                class_indices[i] = cls_idx
                confidences[i] = scores[cls_idx]
                is_bgc_flags[i] = (
                    class_labels[i] != "NonBGC"
                    and confidences[i] >= self.min_confidence
                )
        else:
            keyword_hits_list = [[] for _ in alien_records]

        # Assemble BGCGeneRecord objects
        bgc_records: List[BGCGeneRecord] = []
        for i, rec in enumerate(alien_records):
            conf = confidences[i]

            if conf >= HIGH_CONF_THRESHOLD:
                tier = "High"
            elif conf >= MED_CONF_THRESHOLD:
                tier = "Medium"
            else:
                tier = "Low"

            bgc_records.append(
                BGCGeneRecord(
                    hgt_record=rec,
                    bgc_class=class_labels[i],
                    bgc_class_idx=class_indices[i],
                    confidence=conf,
                    class_scores=all_scores[i],
                    is_bgc=is_bgc_flags[i],
                    confidence_tier=tier,
                    keyword_hits=keyword_hits_list[i],
                )
            )

        return bgc_records

    # ------------------------------------------------------------------
    # Mock backend runner (fallback)
    # ------------------------------------------------------------------

    def _run_mock(
        self,
        alien_records: List[HGTGeneRecord],
        X: np.ndarray,
    ) -> List[BGCGeneRecord]:
        """Run mock MLP inference (fallback when Prophet unavailable)."""
        logger.info("Phase 3 | Using mock MLP backend (no trained model)")

        # Model inference → raw logits
        if _TORCH_AVAILABLE and hasattr(self, "_model") and self._model is not None:
            logits = _torch_inference(self._model, X)
        else:
            logits = _numpy_mock_inference(X, seed=self.seed)

        # Optional keyword boost (applied to logits, before softmax)
        if self.use_keyword_boost:
            logits, keyword_hits_list = _apply_keyword_boosts(alien_records, logits)
        else:
            keyword_hits_list = [[] for _ in alien_records]

        # Softmax → probabilities (AFTER keyword boost)
        logits -= logits.max(axis=1, keepdims=True)  # numerical stability
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)

        # For Mock backend, we must also ensure keyword_hits is available to BGCGeneRecord
        # but the existing loop below uses keyword_hits_list[i] correctly.

        # Assemble BGCGeneRecord objects
        bgc_records: List[BGCGeneRecord] = []
        for i, (rec, kw_hits) in enumerate(zip(alien_records, keyword_hits_list)):
            class_scores = probs[i].tolist()
            cls_idx = int(np.argmax(probs[i]))
            cls_label = BGC_CLASSES[cls_idx]
            conf = float(probs[i, cls_idx])

            is_bgc = cls_label != "NonBGC" and conf >= self.min_confidence

            if conf >= HIGH_CONF_THRESHOLD:
                tier = "High"
            elif conf >= MED_CONF_THRESHOLD:
                tier = "Medium"
            else:
                tier = "Low"

            bgc_records.append(
                BGCGeneRecord(
                    hgt_record=rec,
                    bgc_class=cls_label,
                    bgc_class_idx=cls_idx,
                    confidence=conf,
                    class_scores=class_scores,
                    is_bgc=is_bgc,
                    confidence_tier=tier,
                    keyword_hits=kw_hits,
                )
            )

        return bgc_records

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _empty_result(self) -> BGCResult:
        """Return an empty BGCResult when no alien genes exist."""
        return BGCResult(
            bgc_records=[],
            bgc_hits=[],
            class_distribution={cls: 0 for cls in BGC_CLASSES},
            strain_bgc_counts={},
            feature_matrix=pd.DataFrame(),
            prediction_matrix=pd.DataFrame(),
            stats={
                "n_alien_scored": 0,
                "n_bgc_hits": 0,
                "fisher_enrichment": {"odds_ratio": 1.0, "p_value": 1.0},
            },
        )

    def _make_feature_df(
        self,
        records: List[HGTGeneRecord],
        X: np.ndarray,
    ) -> pd.DataFrame:
        """Return feature matrix as labelled DataFrame."""
        col_names = (
            [
                "gc_content",
                "gc_deviation",
                "kmer_deviation",
                "anomaly_score",
                "mge_proximity",
                "gene_length_log",
                "upstream_nbrs",
                "downstream_nbrs",
            ]
            + [f"gc_bin_{i}" for i in range(10)]
            + [f"kmer_bin_{i}" for i in range(10)]
        )
        df = pd.DataFrame(X, columns=col_names)
        df.insert(0, "gene_id", [r.gene_record.gene_id for r in records])
        df.insert(1, "strain_id", [r.gene_record.strain_id for r in records])
        return df

    def _make_prediction_df(self, bgc_records: List[BGCGeneRecord]) -> pd.DataFrame:
        """Return prediction matrix as labelled DataFrame."""
        rows = []
        for r in bgc_records:
            row = {
                "gene_id": r.gene_id,
                "strain_id": r.strain_id,
                "bgc_class": r.bgc_class,
                "confidence": round(r.confidence, 4),
                "confidence_tier": r.confidence_tier,
                "is_bgc": r.is_bgc,
                "keyword_hits": "|".join(r.keyword_hits),
            }
            for cls, score in zip(BGC_CLASSES, r.class_scores):
                row[f"score_{cls}"] = round(score, 4)
            rows.append(row)
        return pd.DataFrame(rows)
