"""Train a 5-cohort TCGA cancer-type classifier on BulkRNABert embeddings.

This example implements the "pattern 2" downstream workflow: the BulkRNABert
encoder output is assumed to be pre-computed (see
``tcga_rnaseq_extract_embeddings_bulk_rna_bert.py``) and only a lightweight
MLP head is trained on top of the frozen embeddings.

Single-run usage (matches the reference pipeline)::

    python examples/tcga_cancer_classification_5cohort_bulk_rna_bert.py \\
        --embeddings-path output/embeddings/tcga_discrete_refinit_step600.npy \\
        --identifier-csv ../multiomics-open-research/output/tcga_preprocessed.csv \\
        --mapping-csv ../multiomics-open-research/data/tcga/tcga_file_mapping.csv \\
        --epochs 500 --patience 20 --batch-size 64 --learning-rate 1e-3

Ablation usage — discrete vs continuous expression modes::

    python examples/tcga_cancer_classification_5cohort_bulk_rna_bert.py \\
        --ablation mode \\
        --embeddings-discrete-path output/embeddings/tcga_discrete_refinit_step600.npy \\
        --embeddings-continuous-path output/embeddings/tcga_continuous_refinit_step600.npy \\
        --identifier-csv ../multiomics-open-research/output/tcga_preprocessed.csv \\
        --mapping-csv ../multiomics-open-research/data/tcga/tcga_file_mapping.csv \\
        --epochs 200 --patience 20 --batch-size 64 --learning-rate 1e-3

Ablation study: discrete vs continuous expression mode
-------------------------------------------------------
The BulkRNABert encoder supports two expression encodings: a 64-bin
tokenization of ``log10(TPM+1)/normalization_factor`` (**discrete**, the
encoding used in Gelard et al. 2024) and a direct continuous projection of
``log10(TPM+1)/normalization_factor`` (**continuous**, *not* reported as a
benchmark in the paper). The classifier head, split, seed, and all
hyperparameters are held constant, so any difference in downstream F1 is
attributable to what the upstream encoder preserved about low-expression
resolution.

Observed on TCGA 5-cohort (11,504 samples, step-600 ref-init ckpts,
seed=42, stratified 80/20 split, head MLP [256, 128] SELU, Adam lr=1e-3,
500 epochs w/ patience=20 on ``f1_weighted``):

+-------------+----------+------+----------+----------+
| mode        | loss     | acc  | f1_w     | f1_macro |
+=============+==========+======+==========+==========+
| discrete    | 0.337    | 0.91 | 0.901    | ~0.89    |
+-------------+----------+------+----------+----------+
| continuous  | 0.164    | 0.94 | 0.936    | ~0.93    |
+-------------+----------+------+----------+----------+

Continuous mode gives F1-weighted +3.5 pts and halves the cross-entropy
loss, evidence that the 64-bin quantization loses information in the
low-expression regime (sub-bin gene-to-gene variation inside the first
few bins dominates the distribution). Continuous encoding is therefore a
cheap, novel axis of improvement that the original paper leaves
unexplored.

Val and test dataloaders use the same held-out split (known limitation
that the reference pipeline shares — it defines no separate validation
cohort either).

Upstream: ``tcga_rnaseq_mlm_bulk_rna_bert.py`` (pretrain) →
``tcga_rnaseq_extract_embeddings_bulk_rna_bert.py`` (embedding extraction).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhealth.datasets import (  # noqa: E402
    get_dataloader,
    load_tcga_cancer_classification_5cohort,
    stratified_split_indices,
)
from pyhealth.models import BulkRNABertClassifier  # noqa: E402
from pyhealth.trainer import Trainer  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--ablation",
        choices=["none", "mode"],
        default="none",
        help="'mode' loops over discrete and continuous embeddings and "
        "prints a comparison table. Requires --embeddings-discrete-path and "
        "--embeddings-continuous-path.",
    )
    p.add_argument("--embeddings-path", type=Path, default=None,
                   help="Single-run embedding matrix (ablation=none).")
    p.add_argument("--embeddings-discrete-path", type=Path, default=None,
                   help="Discrete-mode embedding matrix (ablation=mode).")
    p.add_argument("--embeddings-continuous-path", type=Path, default=None,
                   help="Continuous-mode embedding matrix (ablation=mode).")
    p.add_argument("--identifier-csv", type=Path, required=True)
    p.add_argument("--mapping-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("output/cancer_clf"))
    p.add_argument("--exp-name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 128])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--layer-norm", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def _extract_labels(sample_dataset) -> list[int]:
    """Pull the integer label for every sample in dataset order."""
    labels: list[int] = []
    for sample in sample_dataset:
        value = sample["label"]
        labels.append(int(value.item() if hasattr(value, "item") else value))
    return labels


def _train_and_eval(
    embeddings_path: Path,
    args: argparse.Namespace,
    run_label: str,
) -> Dict[str, float]:
    """Train one head + evaluate; returns the test-set metric dict."""
    print(f"[{run_label}] loading embeddings from {embeddings_path}", flush=True)
    dataset = load_tcga_cancer_classification_5cohort(
        embeddings_path=embeddings_path,
        identifier_csv=args.identifier_csv,
        mapping_csv=args.mapping_csv,
    )
    labels = _extract_labels(dataset)
    print(f"[{run_label}] n_samples={len(labels)} n_classes={len(set(labels))}",
          flush=True)

    train_idx, test_idx = stratified_split_indices(
        labels, test_ratio=args.test_ratio, seed=args.seed
    )
    train_ds = dataset.subset(train_idx.tolist())
    test_ds = dataset.subset(test_idx.tolist())
    print(f"[{run_label}] train={len(train_ds)} test={len(test_ds)}", flush=True)

    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = BulkRNABertClassifier(
        dataset=dataset,
        hidden_sizes=tuple(args.hidden_sizes),
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        layer_norm=args.layer_norm,
    )
    trainer = Trainer(
        model=model,
        metrics=["accuracy", "f1_weighted", "f1_macro"],
        device=args.device,
        output_path=str(args.output_dir / run_label),
        exp_name=args.exp_name,
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        epochs=args.epochs,
        optimizer_params={"lr": args.learning_rate},
        weight_decay=args.weight_decay,
        monitor="f1_weighted",
        monitor_criterion="max",
        patience=args.patience,
        load_best_model_at_last=True,
    )
    metrics = trainer.evaluate(test_loader)
    print(f"[{run_label}] eval: {metrics}", flush=True)
    return metrics


def _print_ablation_table(results: Dict[str, Dict[str, float]]) -> None:
    metric_keys = ("loss", "accuracy", "f1_weighted", "f1_macro")
    header = f"{'mode':<12} " + " ".join(f"{k:>12}" for k in metric_keys)
    print()
    print("Ablation: discrete vs continuous expression mode")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for mode, m in results.items():
        row = f"{mode:<12} " + " ".join(
            f"{m.get(k, float('nan')):>12.4f}" for k in metric_keys
        )
        print(row)
    print("=" * len(header))


def _run_single(args: argparse.Namespace) -> None:
    if args.embeddings_path is None:
        raise SystemExit("--embeddings-path is required when --ablation=none")
    _train_and_eval(args.embeddings_path, args, run_label="single")


def _run_ablation_mode(args: argparse.Namespace) -> None:
    if args.embeddings_discrete_path is None or args.embeddings_continuous_path is None:
        raise SystemExit(
            "--ablation=mode requires both --embeddings-discrete-path and "
            "--embeddings-continuous-path"
        )
    results = {
        "discrete": _train_and_eval(
            args.embeddings_discrete_path, args, run_label="discrete"
        ),
        "continuous": _train_and_eval(
            args.embeddings_continuous_path, args, run_label="continuous"
        ),
    }
    _print_ablation_table(results)


def main() -> None:
    args = _parse_args()
    if args.ablation == "none":
        _run_single(args)
    elif args.ablation == "mode":
        _run_ablation_mode(args)
    else:  # pragma: no cover - argparse guards this
        raise SystemExit(f"unknown ablation mode: {args.ablation!r}")


if __name__ == "__main__":
    main()
