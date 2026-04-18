"""5-cohort TCGA cancer-type classification task for BulkRNABert embeddings.

This task defines the input / output schema used by
:func:`pyhealth.datasets.load_tcga_cancer_classification_5cohort` and the
:class:`~pyhealth.models.BulkRNABertClassifier` downstream head. Samples
consist of a pre-computed ``(embed_dim,)`` BulkRNABert encoder output and
an integer cancer-type label in ``{0, 1, 2, 3, 4}`` corresponding to the
five cohorts (BLCA, BRCA, GBM+LGG, LUAD, UCEC) used in the reference
experiments.

Unlike most PyHealth tasks, ``__call__`` is a no-op here: the TCGA inputs
live as wide CSV / NPY matrices rather than per-patient event streams, so
the dataset factory constructs sample dicts directly and ``set_task`` is
not part of the pipeline. ``input_schema`` / ``output_schema`` remain
authoritative so :class:`~pyhealth.datasets.SampleBuilder` and the model
can query them as usual.
"""

from __future__ import annotations

from typing import Dict, List

from .base_task import BaseTask


LABEL_MAP: Dict[str, int] = {
    "TCGA-BLCA": 0,
    "TCGA-BRCA": 1,
    "TCGA-GBM": 2,
    "TCGA-LGG": 2,
    "TCGA-LUAD": 3,
    "TCGA-UCEC": 4,
}

COHORT_NAMES: List[str] = ["BLCA", "BRCA", "GBMLGG", "LUAD", "UCEC"]


class TCGACancerClassification5Cohort(BaseTask):
    """PyHealth task for 5-way TCGA cancer-type classification.

    Attributes:
        task_name: Identifier string used by
            :class:`~pyhealth.datasets.SampleDataset`.
        input_schema: ``{"embedding": "tensor"}`` — a pre-computed
            BulkRNABert encoder output of shape ``(embed_dim,)``.
        output_schema: ``{"label": "multiclass"}`` — integer label in
            ``[0, 5)``.
    """

    task_name: str = "TCGACancerClassification5Cohort"
    input_schema: Dict[str, str] = {"embedding": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __call__(self, patient) -> List[Dict]:  # pragma: no cover - unused
        raise NotImplementedError(
            "TCGACancerClassification5Cohort does not use the patient-level "
            "set_task pipeline. Use "
            "pyhealth.datasets.load_tcga_cancer_classification_5cohort to "
            "construct an InMemorySampleDataset directly."
        )


__all__ = [
    "TCGACancerClassification5Cohort",
    "LABEL_MAP",
    "COHORT_NAMES",
]
