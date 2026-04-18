"""Factory for loading pre-computed BulkRNABert embeddings for downstream tasks.

The :func:`load_tcga_cancer_classification_5cohort` factory assembles an
:class:`~pyhealth.datasets.InMemorySampleDataset` whose samples carry
per-patient BulkRNABert encoder outputs (shape ``(embed_dim,)``) together
with a TCGA cancer-type label. It is the PyHealth-native entry point for
the "pattern 2" workflow, where pre-training embeddings are saved once and
only a classifier head is trained on them.

Inputs expected on disk:

* ``embeddings_path`` — ``.npy`` file produced by
  ``examples/bulk_rna_bert_extract_embeddings.py``. Row ``i`` of this file
  must correspond to row ``i`` of ``identifier_csv`` (i.e. the preprocessed
  TCGA CSV used for pre-training).
* ``identifier_csv`` — the same ``tcga_preprocessed.csv`` used during
  pre-training. Only the ``identifier`` column is read here.
* ``mapping_csv`` — ``tcga_file_mapping.csv`` from the TCGA GDC metadata
  dump. The factory joins ``identifier == file_name.split(".")[0]`` and
  filters to ``sample_type == "Primary Tumor"`` and ``project`` in the
  five target cohorts.

All 5-cohort samples are returned in a single dataset; splitting into
train / val / test is the caller's responsibility (see
:func:`stratified_split_indices`).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from pyhealth.tasks.tcga_cancer_classification_5cohort import (
    LABEL_MAP,
    TCGACancerClassification5Cohort,
)

from .sample_dataset import InMemorySampleDataset, create_sample_dataset

logger = logging.getLogger(__name__)


def _build_identifier_to_label(mapping_csv: str | Path) -> dict[str, int]:
    """Parse ``tcga_file_mapping.csv`` and return ``identifier -> int label``.

    The identifier is defined as ``file_name.split(".")[0]`` to match how
    the preprocessed CSV is keyed during pre-training. Only rows whose
    ``project`` is in :data:`LABEL_MAP` and whose ``sample_type`` equals
    ``"Primary Tumor"`` are retained.
    """
    identifier_to_label: dict[str, int] = {}
    with open(mapping_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["project"] in LABEL_MAP and row["sample_type"] == "Primary Tumor":
                key = row["file_name"].split(".")[0]
                identifier_to_label[key] = LABEL_MAP[row["project"]]
    return identifier_to_label


def _select_rows(
    embeddings: np.ndarray,
    identifier_csv: str | Path,
    identifier_to_label: dict[str, int],
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Slice ``embeddings`` to rows whose identifier is in the label map.

    Returns the selected embedding matrix, the integer labels, and the
    identifier strings (for traceability). The row order in the returned
    arrays matches their first-encountered order in ``identifier_csv``.
    """
    with open(identifier_csv) as f:
        header = f.readline().rstrip("\n").split(",")
        if header[-1] != "identifier":
            raise ValueError(
                f"{identifier_csv}: last column must be 'identifier', got "
                f"{header[-1]!r}"
            )
        indices: List[int] = []
        labels: List[int] = []
        identifiers: List[str] = []
        for row_idx, line in enumerate(f):
            identifier = line.rstrip("\n").rsplit(",", 1)[-1]
            label = identifier_to_label.get(identifier)
            if label is not None:
                indices.append(row_idx)
                labels.append(label)
                identifiers.append(identifier)

    if not indices:
        raise ValueError(
            "No rows in the identifier CSV matched any label in "
            "tcga_file_mapping.csv. Check that the two files cover the "
            "same cohort."
        )

    if max(indices) >= embeddings.shape[0]:
        raise ValueError(
            f"identifier CSV has row index {max(indices)} but embeddings "
            f"has only {embeddings.shape[0]} rows — mismatched files?"
        )

    selected = embeddings[np.asarray(indices)]
    return selected, labels, identifiers


def load_tcga_cancer_classification_5cohort(
    embeddings_path: str | Path,
    identifier_csv: str | Path,
    mapping_csv: str | Path,
    dataset_name: str = "TCGA_BulkRNABert_Embeddings",
) -> InMemorySampleDataset:
    """Build an in-memory SampleDataset of pre-computed BulkRNABert embeddings.

    Args:
        embeddings_path: Path to the ``.npy`` file holding all TCGA
            embeddings in the same row order as ``identifier_csv``. Shape
            ``(n_total_samples, embed_dim)``.
        identifier_csv: ``tcga_preprocessed.csv`` used during pre-training.
            Its last column must be ``identifier``.
        mapping_csv: ``tcga_file_mapping.csv`` from the TCGA GDC metadata
            dump.
        dataset_name: Name attached to the returned dataset.

    Returns:
        An :class:`InMemorySampleDataset` whose samples are dicts
        ``{"patient_id": identifier, "embedding": np.ndarray,
        "label": int}`` with schema defined by
        :class:`~pyhealth.tasks.TCGACancerClassification5Cohort`.
    """
    embeddings = np.load(embeddings_path).astype(np.float32)
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2-D (n_samples, embed_dim), got shape "
            f"{embeddings.shape}"
        )
    logger.info(
        "Loaded BulkRNABert embeddings from %s (shape=%s)",
        embeddings_path,
        embeddings.shape,
    )

    identifier_to_label = _build_identifier_to_label(mapping_csv)
    logger.info(
        "Built identifier->label map from %s (%d entries)",
        mapping_csv,
        len(identifier_to_label),
    )

    selected_embeddings, labels, identifiers = _select_rows(
        embeddings, identifier_csv, identifier_to_label
    )
    logger.info(
        "Selected %d samples across %d cohorts",
        len(labels),
        len(set(labels)),
    )

    task = TCGACancerClassification5Cohort()
    samples = [
        {
            "patient_id": identifier,
            "embedding": selected_embeddings[i],
            "label": int(labels[i]),
        }
        for i, identifier in enumerate(identifiers)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        dataset_name=dataset_name,
        task_name=task.task_name,
        in_memory=True,
    )


def stratified_split_indices(
    labels: List[int] | np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(train_idx, test_idx)`` stratified by label.

    PyHealth's built-in ``split_by_sample`` does not preserve class
    proportions; this helper mirrors the per-class shuffle used in the
    reference BulkRNABert downstream pipeline.

    Args:
        labels: Integer labels of shape ``(n_samples,)``.
        test_ratio: Fraction of each class routed to the test split.
        seed: Seed for the NumPy generator used in the per-class shuffle.

    Returns:
        Two arrays of integer indices into the original sample order.
    """
    labels_arr = np.asarray(labels)
    rng = np.random.default_rng(seed)

    train_idx: List[int] = []
    test_idx: List[int] = []
    for lbl in np.unique(labels_arr):
        idx = np.where(labels_arr == lbl)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    train_arr = np.array(train_idx)
    test_arr = np.array(test_idx)
    rng.shuffle(train_arr)
    rng.shuffle(test_arr)
    return train_arr, test_arr


__all__ = [
    "load_tcga_cancer_classification_5cohort",
    "stratified_split_indices",
]
