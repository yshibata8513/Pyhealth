"""Unit tests for :class:`pyhealth.models.BulkRNABert`."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pyhealth.models import (
    BulkRNABert,
    BulkRNABertConfig,
    bin_expression_values,
    load_expression_csv,
)


def _small_config(expression_mode: str = "discrete") -> BulkRNABertConfig:
    return BulkRNABertConfig(
        n_genes=32,
        n_bins=8,
        embed_dim=16,
        num_layers=2,
        num_heads=4,
        ffn_embed_dim=32,
        init_gene_embed_dim=16,
        expression_mode=expression_mode,
        continuous_hidden_dim=16 if expression_mode == "continuous" else None,
    )


class TestBulkRNABertDiscrete(unittest.TestCase):
    def setUp(self):
        self.cfg = _small_config("discrete")
        self.model = BulkRNABert(
            dataset=None, config=self.cfg, feature_key="expression"
        )

    def test_mode(self):
        self.assertEqual(self.model.mode, "multiclass")

    def test_forward_train_shapes(self):
        self.model.train()
        tokens = torch.randint(0, self.cfg.n_bins, (4, self.cfg.n_genes))
        out = self.model(expression=tokens)
        self.assertIn("loss", out)
        self.assertEqual(out["loss"].dim(), 0)
        self.assertEqual(
            out["logits"].shape, (4, self.cfg.n_genes, self.cfg.n_bins)
        )
        # y_prob / y_true gathered at masked positions
        self.assertEqual(out["y_prob"].dim(), 2)
        self.assertEqual(out["y_prob"].shape[1], self.cfg.n_bins)
        self.assertEqual(out["y_true"].dim(), 1)
        self.assertEqual(out["y_prob"].shape[0], out["y_true"].shape[0])

    def test_backward(self):
        self.model.train()
        tokens = torch.randint(0, self.cfg.n_bins, (4, self.cfg.n_genes))
        out = self.model(expression=tokens)
        out["loss"].backward()
        grads = [
            p.grad for p in self.model.parameters() if p.requires_grad
        ]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))

    def test_eval_no_mask(self):
        self.model.eval()
        tokens = torch.randint(0, self.cfg.n_bins, (2, self.cfg.n_genes))
        out = self.model(expression=tokens)
        self.assertEqual(float(out["loss"]), 0.0)
        self.assertEqual(
            out["y_prob"].shape, (2 * self.cfg.n_genes, self.cfg.n_bins)
        )

    def test_masking_ratio(self):
        """Empirical mask ratio should be close to mlm_probability."""
        torch.manual_seed(0)
        self.model.train()
        big_tokens = torch.zeros(
            (8, self.cfg.n_genes), dtype=torch.long
        )
        _, mask_positions = self.model._apply_mask_discrete(big_tokens)
        ratio = mask_positions.float().mean().item()
        self.assertAlmostEqual(ratio, self.cfg.mlm_probability, delta=0.05)

    def test_sequence_length_mismatch(self):
        self.model.train()
        tokens = torch.randint(0, self.cfg.n_bins, (2, self.cfg.n_genes - 1))
        with self.assertRaises(ValueError):
            self.model(expression=tokens)


class TestBulkRNABertContinuous(unittest.TestCase):
    def setUp(self):
        self.cfg = _small_config("continuous")
        self.model = BulkRNABert(
            dataset=None, config=self.cfg, feature_key="expression"
        )

    def test_mode(self):
        self.assertEqual(self.model.mode, "regression")

    def test_forward_train_shapes(self):
        self.model.train()
        values = torch.rand(4, self.cfg.n_genes) * 5.0
        out = self.model(expression=values)
        self.assertIn("loss", out)
        self.assertEqual(out["loss"].dim(), 0)
        self.assertEqual(out["predictions"].shape, (4, self.cfg.n_genes))
        self.assertEqual(out["y_prob"].dim(), 1)
        self.assertEqual(out["y_prob"].shape, out["y_true"].shape)

    def test_backward(self):
        self.model.train()
        values = torch.rand(4, self.cfg.n_genes) * 5.0
        out = self.model(expression=values)
        out["loss"].backward()
        grads = [
            p.grad for p in self.model.parameters() if p.requires_grad
        ]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))

    def test_eval_no_mask(self):
        self.model.eval()
        values = torch.rand(2, self.cfg.n_genes)
        out = self.model(expression=values)
        self.assertEqual(float(out["loss"]), 0.0)
        self.assertEqual(out["predictions"].shape, (2, self.cfg.n_genes))


class TestTrainingLoop(unittest.TestCase):
    """Integration test: a few manual optimizer steps should decrease loss."""

    def _run_loop(self, expression_mode: str):
        torch.manual_seed(0)
        cfg = _small_config(expression_mode)
        model = BulkRNABert(
            dataset=None, config=cfg, feature_key="expression"
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        model.train()

        if expression_mode == "discrete":
            data = torch.randint(0, cfg.n_bins, (16, cfg.n_genes))
        else:
            data = torch.rand(16, cfg.n_genes) * 5.0

        first_losses, last_losses = [], []
        for step in range(20):
            out = model(expression=data)
            optim.zero_grad()
            out["loss"].backward()
            optim.step()
            if step < 3:
                first_losses.append(float(out["loss"]))
            if step >= 17:
                last_losses.append(float(out["loss"]))
        # loss should decrease on this tiny fixed batch
        self.assertLess(
            sum(last_losses) / len(last_losses),
            sum(first_losses) / len(first_losses),
        )

    def test_discrete_loop(self):
        self._run_loop("discrete")

    def test_continuous_loop(self):
        self._run_loop("continuous")


class TestBinExpressionValues(unittest.TestCase):
    def test_shape_and_range(self):
        vals = np.random.rand(4, 20) * 5.0
        bins = bin_expression_values(vals, n_bins=16)
        self.assertEqual(bins.shape, (4, 20))
        self.assertTrue(bins.min() >= 0)
        self.assertTrue(bins.max() < 16)

    def test_zero_values_go_to_bin_zero(self):
        vals = np.zeros((2, 5))
        bins = bin_expression_values(vals, n_bins=8)
        self.assertTrue(torch.equal(bins, torch.zeros(2, 5, dtype=torch.long)))

    def test_raw_tpm_path(self):
        tpm = np.array([[0.0, 1.0, 10.0, 100.0]])
        bins_log = bin_expression_values(
            np.log10(tpm + 1.0), n_bins=8, already_log_normalized=True
        )
        bins_raw = bin_expression_values(
            tpm, n_bins=8, already_log_normalized=False
        )
        self.assertTrue(torch.equal(bins_log, bins_raw))

    def test_monotonic(self):
        vals = np.linspace(0.0, 5.5, 50).reshape(1, -1)
        bins = bin_expression_values(vals, n_bins=16).squeeze(0)
        # non-decreasing
        self.assertTrue(torch.all(bins[1:] >= bins[:-1]))


class TestAutocastDtypeConfig(unittest.TestCase):
    def test_invalid_value_rejected(self):
        with self.assertRaises(ValueError):
            BulkRNABertConfig(
                n_genes=8, n_bins=4, embed_dim=8, num_layers=1,
                num_heads=2, ffn_embed_dim=16, init_gene_embed_dim=8,
                autocast_dtype="float64",
            )

    def test_cpu_autocast_is_noop(self):
        """autocast_dtype is ignored on CPU — forward still works normally."""
        cfg = BulkRNABertConfig(
            n_genes=8, n_bins=4, embed_dim=8, num_layers=1,
            num_heads=2, ffn_embed_dim=16, init_gene_embed_dim=8,
            autocast_dtype="bfloat16",
        )
        model = BulkRNABert(dataset=None, config=cfg, feature_key="expression")
        self.assertFalse(model._autocast_enabled())  # CPU -> disabled
        tokens = torch.randint(0, cfg.n_bins, (2, cfg.n_genes))
        model.train()
        out = model(expression=tokens)
        self.assertIn("loss", out)


class TestLoadExpressionCSV(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "expr.csv"
        self.n_genes = 10
        self.n_samples = 3
        rng = np.random.default_rng(0)
        tpm = rng.uniform(0.0, 100.0, size=(self.n_samples, self.n_genes))
        cols = [f"ENSG{i:011d}" for i in range(self.n_genes)]
        df = pd.DataFrame(tpm, columns=cols)
        df["identifier"] = [f"sample-{i}" for i in range(self.n_samples)]
        df.to_csv(self.path, index=False)
        self.cols = cols
        self.tpm = tpm

    def tearDown(self):
        self.tmp.cleanup()

    def test_continuous_shape_and_log10(self):
        tensor, genes = load_expression_csv(self.path, mode="continuous")
        self.assertEqual(genes, self.cols)
        self.assertEqual(tensor.shape, (self.n_samples, self.n_genes))
        self.assertEqual(tensor.dtype, torch.float32)
        # Continuous mode applies the same normalization_factor as discrete
        # tokenization so the model sees inputs on a consistent scale; see
        # reference dataloader.py :: load_continuous.
        norm = 5.547176906585117
        expected = (np.log10(self.tpm + 1.0) / norm).astype(np.float32)
        np.testing.assert_allclose(tensor.numpy(), expected, rtol=1e-5)

    def test_discrete_returns_long_tokens(self):
        tokens, genes = load_expression_csv(
            self.path, mode="discrete", n_bins=64
        )
        self.assertEqual(genes, self.cols)
        self.assertEqual(tokens.shape, (self.n_samples, self.n_genes))
        self.assertEqual(tokens.dtype, torch.long)
        self.assertGreaterEqual(tokens.min().item(), 0)
        self.assertLess(tokens.max().item(), 64)

    def test_identifier_column_dropped(self):
        _, genes = load_expression_csv(self.path, mode="continuous")
        self.assertNotIn("identifier", genes)

    def test_non_numeric_residual_raises(self):
        bad = Path(self.tmp.name) / "bad.csv"
        df = pd.DataFrame(
            {"g0": [1.0, 2.0], "g1": [3.0, 4.0], "cohort": ["A", "B"]}
        )
        df.to_csv(bad, index=False)
        with self.assertRaises(ValueError):
            load_expression_csv(bad, mode="continuous", drop_columns=())

    def test_already_log_normalized_skips_log(self):
        logged = np.log10(self.tpm + 1.0)
        df = pd.DataFrame(logged, columns=self.cols)
        logged_path = Path(self.tmp.name) / "logged.csv"
        df.to_csv(logged_path, index=False)
        tensor, _ = load_expression_csv(
            logged_path, mode="continuous", already_log_normalized=True
        )
        norm = 5.547176906585117
        np.testing.assert_allclose(
            tensor.numpy(), (logged / norm).astype(np.float32), rtol=1e-5
        )

    def test_feeds_into_model(self):
        cfg = BulkRNABertConfig(
            n_genes=self.n_genes,
            n_bins=8,
            embed_dim=16,
            num_layers=1,
            num_heads=4,
            ffn_embed_dim=32,
            init_gene_embed_dim=16,
            expression_mode="continuous",
            continuous_hidden_dim=16,
        )
        model = BulkRNABert(dataset=None, config=cfg, feature_key="expression")
        tensor, _ = load_expression_csv(self.path, mode="continuous")
        model.train()
        out = model(expression=tensor)
        self.assertEqual(out["predictions"].shape, tensor.shape)


if __name__ == "__main__":
    unittest.main()
