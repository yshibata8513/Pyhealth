"""Pre-train BulkRNABert on a bulk RNA-seq expression CSV (e.g. TCGA).

Usage example (discrete MLM, paper config):

    python examples/bulk_rna_bert_pretrain.py \\
        --csv-path ../multiomics-open-research/output/tcga_preprocessed.csv \\
        --output-dir output/bulk_rna_bert_pretrain_tcga_discrete \\
        --mode discrete --micro-batch-size 2 --accumulation-steps 80 \\
        --max-steps 600 --learning-rate 1e-4 --save-every 50 --log-every 10 --seed 42

The script implements the same training recipe as the reference repository's
``pretrain_bulk_rna_bert.py`` (micro_batch=2, accumulation=80, Adam, lr=1e-4),
with bf16 autocast and FlashAttention-2 forced via SDPA for speed on modern
GPUs (Blackwell / Ampere).

Checkpoints are written to ``<output-dir>/step_{N}/`` as ``params.pt`` +
``config.json`` every ``--save-every`` effective steps. Receiving SIGTERM (for
example from the companion GPU temperature watchdog) triggers a graceful stop
that writes one last checkpoint before exiting.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

# Make PyHealth importable when running from a source checkout.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhealth.models import (  # noqa: E402
    BulkRNABert,
    BulkRNABertConfig,
    load_expression_csv,
)


# ---------------------------------------------------------------------------
# Graceful-shutdown flag (flipped by SIGTERM / SIGINT)
# ---------------------------------------------------------------------------

_STOP_REQUESTED = False


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
        print(
            f"[signal] received signal {signum}; will stop after current step",
            flush=True,
        )

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--csv-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--mode", choices=["discrete", "continuous"], default="discrete"
    )
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--accumulation-steps", type=int, default=80)
    p.add_argument("--max-steps", type=int, default=600,
                   help="Stop after this many effective (post-accumulation) steps.")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=50,
                   help="Effective-step interval for checkpoint saving.")
    p.add_argument("--log-every", type=int, default=10,
                   help="Effective-step interval for log lines.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bins", type=int, default=64)
    p.add_argument(
        "--already-log-normalized", action="store_true",
        help="Set if the CSV values are already log10(TPM+1) rather than raw TPM."
    )
    p.add_argument(
        "--autocast-dtype", choices=["bfloat16", "float16", "none"], default="bfloat16",
        help="Mixed-precision dtype for autocast inside forward. 'none' disables autocast."
    )
    p.add_argument(
        "--no-force-flash-attention", action="store_true",
        help="Disable the explicit FLASH_ATTENTION SDPA backend (default: on).",
    )
    p.add_argument(
        "--init-gene-embedding-from", type=Path, default=None,
        help="Path to a multiomics-open-research params.joblib. If given, "
             "copies gene_embedding.embeddings and the (200->embed_dim) linear "
             "projection into model.gene_embedding before training starts. "
             "Only the 2 gene-embedding tensors are used; attention and LM "
             "head stay at fresh initialization.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data(args: argparse.Namespace) -> Tuple[torch.Tensor, List[str]]:
    print(f"[data] loading {args.csv_path} (mode={args.mode}) ...", flush=True)
    t0 = time.time()
    data, gene_names = load_expression_csv(
        args.csv_path,
        mode=args.mode,
        n_bins=args.n_bins,
        already_log_normalized=args.already_log_normalized,
    )
    print(
        f"[data] loaded tensor {tuple(data.shape)} dtype={data.dtype} "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )
    return data, gene_names


def _build_model(
    args: argparse.Namespace, n_genes: int
) -> Tuple[BulkRNABert, BulkRNABertConfig]:
    cfg = BulkRNABertConfig(
        n_genes=n_genes,
        n_bins=args.n_bins,
        expression_mode=args.mode,
        continuous_hidden_dim=256 if args.mode == "continuous" else None,
        autocast_dtype=None if args.autocast_dtype == "none" else args.autocast_dtype,
    )
    model = BulkRNABert(dataset=None, config=cfg, feature_key="expression")
    return model, cfg


def _load_reference_gene_embedding(model: BulkRNABert, path: Path) -> None:
    """Copy gene_embedding tensors from a JAX/Haiku ``params.joblib``.

    Only two modules are copied — ``bulk_bert/~/gene_embedding.embeddings``
    and ``bulk_bert/~/linear.{w, b}`` — matching the reference pipeline's
    ``init_params_with_gene_embedding`` helper. Attention and LM-head weights
    keep their fresh initialization so the model still learns TCGA-specific
    expression patterns from scratch.
    """
    import joblib  # optional dep; keep lazy

    gm = model.gene_embedding
    if gm is None:
        raise ValueError(
            "model has no gene_embedding module (config.use_gene_embedding=False)"
        )
    if gm.proj is None:
        raise ValueError(
            "cannot load reference weights: model.gene_embedding has no proj "
            "layer (init_gene_embed_dim == embed_dim). Set init_gene_embed_dim=200 "
            "and embed_dim=256 to match the reference checkpoint."
        )

    params = joblib.load(path)
    ge = params["bulk_bert/~/gene_embedding"]["embeddings"]
    lw = params["bulk_bert/~/linear"]["w"]
    lb = params["bulk_bert/~/linear"]["b"]

    # JAX linear stores (in, out); torch nn.Linear weight is (out, in).
    lw_t = lw.T.copy()  # .copy() makes it contiguous after transpose

    if gm.embed.weight.shape != ge.shape:
        raise ValueError(
            f"gene_embedding shape mismatch: model {tuple(gm.embed.weight.shape)} "
            f"vs ref {ge.shape}"
        )
    if gm.proj.weight.shape != lw_t.shape:
        raise ValueError(
            f"gene_embedding proj.weight shape mismatch: model "
            f"{tuple(gm.proj.weight.shape)} vs ref {lw_t.shape}"
        )
    if gm.proj.bias.shape != lb.shape:
        raise ValueError(
            f"gene_embedding proj.bias shape mismatch: model "
            f"{tuple(gm.proj.bias.shape)} vs ref {lb.shape}"
        )

    with torch.no_grad():
        gm.embed.weight.copy_(torch.from_numpy(ge))
        gm.proj.weight.copy_(torch.from_numpy(lw_t))
        gm.proj.bias.copy_(torch.from_numpy(lb))
    print(f"[init] loaded gene_embedding from {path}", flush=True)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _save_checkpoint(
    model: BulkRNABert,
    cfg: BulkRNABertConfig,
    step: int,
    output_dir: Path,
) -> Path:
    ckpt_dir = output_dir / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "params.pt")
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    return ckpt_dir


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _iter_micro_batches(
    data: torch.Tensor, micro_batch_size: int, generator: torch.Generator
):
    """Yield micro-batch tensors, reshuffling every full pass."""
    n = data.shape[0]
    while True:
        perm = torch.randperm(n, generator=generator)
        for i in range(0, n - micro_batch_size + 1, micro_batch_size):
            idx = perm[i : i + micro_batch_size]
            yield data[idx]


def _train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Save a run manifest so downstream analysis can recover the command line.
    with open(args.output_dir / "run_args.json", "w") as f:
        json.dump(
            {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            f,
            indent=2,
        )

    data, gene_names = _load_data(args)
    n_samples, n_genes = data.shape
    with open(args.output_dir / "gene_names.json", "w") as f:
        json.dump(gene_names, f)

    model, cfg = _build_model(args, n_genes=n_genes)
    if args.init_gene_embedding_from is not None:
        _load_reference_gene_embedding(model, args.init_gene_embedding_from)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(
        f"[model] n_genes={cfg.n_genes} mode={cfg.expression_mode} "
        f"params={sum(p.numel() for p in model.parameters()):,} device={device} "
        f"autocast={cfg.autocast_dtype}",
        flush=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()

    generator = torch.Generator().manual_seed(args.seed)
    micro_iter = _iter_micro_batches(data, args.micro_batch_size, generator)

    from contextlib import nullcontext
    # Attention-backend context: FlashAttention only (fastest path on modern
    # GPUs when autocast is bf16/fp16).
    if args.no_force_flash_attention or not torch.cuda.is_available():
        attn_ctx = nullcontext()
    else:
        attn_ctx = sdpa_kernel([SDPBackend.FLASH_ATTENTION])

    steps_per_epoch = n_samples // (args.micro_batch_size * args.accumulation_steps)
    print(
        f"[train] micro_batch={args.micro_batch_size} accum={args.accumulation_steps} "
        f"effective_batch={args.micro_batch_size * args.accumulation_steps} "
        f"eff_steps_per_epoch~{steps_per_epoch} max_steps={args.max_steps}",
        flush=True,
    )

    eff_step = 0
    micro_accum_loss = 0.0
    t_start = time.time()
    t_last_log = t_start

    with attn_ctx:
        while eff_step < args.max_steps and not _STOP_REQUESTED:
            optimizer.zero_grad(set_to_none=True)
            total_loss_value = 0.0
            for _ in range(args.accumulation_steps):
                batch = next(micro_iter)
                out = model(expression=batch)
                loss = out["loss"] / args.accumulation_steps
                loss.backward()
                total_loss_value += float(loss.detach()) * args.accumulation_steps
            optimizer.step()
            eff_step += 1
            avg_micro_loss = total_loss_value / args.accumulation_steps
            micro_accum_loss += avg_micro_loss

            if eff_step % args.log_every == 0:
                elapsed = time.time() - t_start
                since_last = time.time() - t_last_log
                t_last_log = time.time()
                mean_loss = micro_accum_loss / args.log_every
                micro_accum_loss = 0.0
                peak_mem = (
                    torch.cuda.max_memory_allocated() / (1024 ** 3)
                    if torch.cuda.is_available()
                    else 0.0
                )
                print(
                    f"[step {eff_step:4d}/{args.max_steps}] "
                    f"loss={mean_loss:.4f} "
                    f"elapsed={elapsed:.1f}s (+{since_last:.1f}s) "
                    f"peak_mem={peak_mem:.2f}GB",
                    flush=True,
                )

            if eff_step % args.save_every == 0:
                path = _save_checkpoint(model, cfg, eff_step, args.output_dir)
                print(f"[ckpt] saved {path}", flush=True)

        # Final safety save if a graceful stop happened off-boundary.
        if _STOP_REQUESTED and eff_step % args.save_every != 0:
            path = _save_checkpoint(model, cfg, eff_step, args.output_dir)
            print(f"[ckpt] (graceful stop) saved {path}", flush=True)

    elapsed = time.time() - t_start
    print(
        f"[done] stopped at step {eff_step} after {elapsed:.1f}s "
        f"(stop_requested={_STOP_REQUESTED})",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    _install_signal_handlers()
    _train(args)


if __name__ == "__main__":
    main()
