"""Train a 5-cohort TCGA cancer-type classifier on BulkRNABert embeddings.

This example implements the "pattern 2" downstream workflow: the BulkRNABert
encoder output is assumed to be pre-computed (see
``tcga_rnaseq_extract_embeddings_bulk_rna_bert.py``) and only a lightweight MLP head is
trained on top of the frozen embeddings.

Usage example:

    python examples/tcga_cancer_classification_5cohort_bulk_rna_bert.py \\
        --embeddings-path output/embeddings/tcga_discrete_refinit_step600.npy \\
        --identifier-csv ../multiomics-open-research/output/tcga_preprocessed.csv \\
        --mapping-csv ../multiomics-open-research/data/tcga/tcga_file_mapping.csv \\
        --epochs 500 --patience 20 --batch-size 64 --learning-rate 1e-3

Val and test dataloaders use the same held-out split (documented known
limitation — the reference pipeline does not define a separate validation
cohort either).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    p.add_argument("--embeddings-path", type=Path, required=True)
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


def main() -> None:
    args = _parse_args()

    print(f"[data] loading 5-cohort embeddings from {args.embeddings_path}", flush=True)
    dataset = load_tcga_cancer_classification_5cohort(
        embeddings_path=args.embeddings_path,
        identifier_csv=args.identifier_csv,
        mapping_csv=args.mapping_csv,
    )
    labels = _extract_labels(dataset)
    print(f"[data] n_samples={len(labels)} n_classes={len(set(labels))}", flush=True)

    train_idx, test_idx = stratified_split_indices(
        labels, test_ratio=args.test_ratio, seed=args.seed
    )
    train_ds = dataset.subset(train_idx.tolist())
    test_ds = dataset.subset(test_idx.tolist())
    print(
        f"[split] train={len(train_ds)} test={len(test_ds)} (val reuses test)",
        flush=True,
    )

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
        output_path=str(args.output_dir),
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

    print("[eval] final test metrics:", trainer.evaluate(test_loader), flush=True)


if __name__ == "__main__":
    main()
