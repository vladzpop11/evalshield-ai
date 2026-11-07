"""EvalShield-AI: ML Drift & Metrics Degradation Tests.

Validates statistical drift detection mechanisms:
1. Kolmogorov-Smirnov (KS) 2-Sample Test
2. Population Stability Index (PSI) calculation
3. Production metric thresholds (Accuracy, F1-macro)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from src.ml.inference import get_inference_engine


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between baseline and sample."""
    # Compute quantiles based on expected distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_bounds = np.percentile(expected, percentiles)
    # Ensure distinct boundaries
    bucket_bounds[0] = -np.inf
    bucket_bounds[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bucket_bounds)
    actual_counts, _ = np.histogram(actual, bins=bucket_bounds)

    # Convert to proportions with smoothing for zero-count bins
    expected_pct = np.clip(expected_counts / len(expected), 1e-4, 1.0)
    actual_pct = np.clip(actual_counts / len(actual), 1e-4, 1.0)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


@pytest.fixture(scope="module")
def artifacts_dir():
    """Retrieve artifacts directory."""
    engine = get_inference_engine()
    return engine.artifacts_dir


@pytest.mark.ml_quality
class TestMetricsAndDrift:
    """Validate drift alarms and performance governance criteria."""

    def test_model_meets_minimum_performance_benchmarks(self, artifacts_dir):
        """Ensure the exported model meets quality gates: Accuracy >= 0.85, F1 >= 0.80."""
        meta_file = artifacts_dir / "metadata.json"
        assert meta_file.exists(), "metadata.json not found"

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        metrics = meta["metrics"]
        assert (
            metrics["accuracy"] >= 0.85
        ), f"Accuracy {metrics['accuracy']:.4f} is below 0.85 quality gate"
        assert (
            metrics["f1_macro"] >= 0.80
        ), f"Macro F1 {metrics['f1_macro']:.4f} is below 0.80 quality gate"

    def test_ks_test_detects_distribution_drift(self):
        """Kolmogorov-Smirnov test must flag shifted feature distributions."""
        rng = np.random.RandomState(42)
        baseline = rng.normal(loc=0.3, scale=0.1, size=500)
        drifted = rng.normal(loc=0.6, scale=0.15, size=500)  # Significant shift

        stat, p_value = ks_2samp(baseline, drifted)
        assert p_value < 0.01, f"Expected drift alarm, got p-value {p_value}"

    def test_ks_test_passes_on_identically_distributed_data(self):
        """Kolmogorov-Smirnov test should NOT flag identically distributed fresh samples."""
        rng = np.random.RandomState(101)
        baseline = rng.beta(0.5, 4.0, size=600)
        fresh_in_domain = rng.beta(0.5, 4.0, size=600)

        stat, p_value = ks_2samp(baseline, fresh_in_domain)
        assert p_value > 0.05, f"False drift alarm triggered, p-value: {p_value}"

    def test_psi_stability_thresholds(self):
        """Verify PSI correctly identifies stable vs drifting populations."""
        rng = np.random.RandomState(999)
        baseline = rng.normal(loc=5.0, scale=1.5, size=1000)

        # 1. Stable sample from same generator
        sample_stable = rng.normal(loc=5.0, scale=1.5, size=1000)
        psi_stable = calculate_psi(baseline, sample_stable)
        assert psi_stable < 0.10, f"Stable population produced high PSI: {psi_stable}"

        # 2. Heavily drifted sample
        sample_drifted = rng.normal(loc=7.5, scale=2.5, size=1000)
        psi_drifted = calculate_psi(baseline, sample_drifted)
        assert psi_drifted > 0.25, f"Drifted population produced low PSI: {psi_drifted}"
