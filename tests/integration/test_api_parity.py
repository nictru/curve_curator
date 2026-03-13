"""test_api_parity.py
End-to-end quality test: verify that run_pipeline_api (PyTorch backend)
finds curve fits of equal quality to the original scipy fitting.

The key insight is that both backends minimise the same OLS objective over the
same non-convex 4PL landscape, so they may converge to *different* local minima
while still achieving equally good fits.  We therefore compare **fit quality**
(residual SSE per curve) rather than raw parameter values.

A PyTorch fit is considered equally good if its SSE is within a small tolerance
of the scipy SSE (or better).  We also verify that summary metrics (R², AUC,
regulation labels) are well-formed.
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from curve_curator import run_pipeline_api
from curve_curator import data_parser, quantification, thresholding, toml_parser

# ---------------------------------------------------------------------------
# Shared synthetic dataset
# ---------------------------------------------------------------------------
N_CURVES = 50
N_DOSES = 6
RNG = np.random.default_rng(42)

TRUE_PEC50 = RNG.uniform(5.0, 9.0, N_CURVES)
TRUE_SLOPE = RNG.uniform(0.5, 3.0, N_CURVES)
TRUE_FRONT = RNG.uniform(0.9, 1.1, N_CURVES)
TRUE_BACK = RNG.uniform(0.0, 0.3, N_CURVES)

DOSES_UM = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
DOSE_SCALE = 1e-6  # µM → M


def _4pl(x: np.ndarray, pec50: float, slope: float, front: float, back: float) -> np.ndarray:
    return (front - back) / (1 + 10 ** (slope * (x + pec50))) + back


def _make_synthetic_tsv(path: Path) -> None:
    """Write a TSV with 1 synthetic control + N_DOSES dose viability columns."""
    log_doses = np.log10(DOSES_UM * DOSE_SCALE)  # log10(M) from -9 to -4
    n_exp = 1 + N_DOSES
    col_names = ["Name"] + [f"Raw {i}" for i in range(n_exp)]
    rows = []
    for i in range(N_CURVES):
        y = _4pl(log_doses, TRUE_PEC50[i], TRUE_SLOPE[i], TRUE_FRONT[i], TRUE_BACK[i])
        y += RNG.normal(0, 0.01, N_DOSES)
        y = np.clip(y, 1e-4, 1e6)
        rows.append([f"CVCL_{i:04d}|DRUG{i:04d}", 1.0] + y.tolist())
    pd.DataFrame(rows, columns=col_names).to_csv(path, sep="\t", index=False)


def _make_config(tsv_path: Path, output_path: Path) -> dict:
    n_exp = 1 + N_DOSES
    return {
        "Meta": {
            "id": tsv_path.name,
            "description": "parity_test",
            "condition": "",
            "treatment_time": "72 h",
        },
        "Experiment": {
            "experiments": list(range(n_exp)),
            "control_experiment": [0],
            "doses": [0.0] + list(DOSES_UM),
            "dose_scale": str(DOSE_SCALE),
            "measurement_type": "OTHER",
            "data_type": "OTHER",
            "search_engine": "OTHER",
            "search_engine_version": "0",
        },
        "Paths": {
            "input_file": str(tsv_path),
            "curves_file": str(output_path / "curves.tsv"),
            "normalization_file": str(output_path / "norm.txt"),
        },
        "F Statistic": {"alpha": 0.05, "fc_lim": 0.5},
        "__file__": {"Path": str(output_path / "config.toml")},
    }


@pytest.fixture(scope="module")
def synthetic_data(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("parity")
    tsv_path = tmp / "data.tsv"
    _make_synthetic_tsv(tsv_path)
    cfg = toml_parser.set_default_values(_make_config(tsv_path, tmp))
    return {"config": cfg}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_scipy(config: dict) -> pd.DataFrame:
    os.environ.setdefault("TQDM_DISABLE", "1")
    with contextlib.redirect_stdout(io.StringIO()):
        data = data_parser.load(config)
        data = quantification.run_pipeline(data, config=config)
        data = thresholding.apply_significance_thresholds(data, config=config)
    return data


def _compute_sse(result: pd.DataFrame, config: dict) -> np.ndarray:
    """Recompute per-curve OLS SSE from fitted parameters and observed ratios.

    Uses dose-sorted Ratio columns (same as the fitters), returns (N,) array.
    """
    from curve_curator import toolbox as tool

    experiments = np.array(config["Experiment"]["experiments"])
    drug_concs = np.array(config["Experiment"]["doses"])
    drug_scale = float(config["Experiment"]["dose_scale"])
    control_mask = drug_concs != 0.0

    drug_log_concs = tool.build_drug_log_concentrations(drug_concs[control_mask], drug_scale)
    sorted_idx = np.argsort(drug_log_concs)
    drug_log_concs_sorted = drug_log_concs[sorted_idx]
    cols_ratio_dose = tool.build_col_names("Ratio {}", experiments[control_mask][sorted_idx])

    x = drug_log_concs_sorted           # (D,)
    y_obs = result[cols_ratio_dose].to_numpy()  # (N, D)

    pec50 = result["pEC50"].to_numpy()
    slope = result["Curve Slope"].to_numpy()
    front = result["Curve Front"].to_numpy()
    back = result["Curve Back"].to_numpy()

    y_pred = (
        (front - back)[:, None]
        / (1 + 10 ** (slope[:, None] * (x[None, :] + pec50[:, None])))
        + back[:, None]
    )
    return np.nansum((y_pred - y_obs) ** 2, axis=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestApiParity:
    """PyTorch backend must find fits of equal or better quality than scipy."""

    @pytest.fixture(autouse=True)
    def load_results(self, synthetic_data: dict) -> None:
        self.config = synthetic_data["config"]
        self.scipy_result = _run_scipy(self.config)
        self.torch_result = run_pipeline_api(self.config, device="cpu")
        self.scipy_sse = _compute_sse(self.scipy_result, self.config)
        self.torch_sse = _compute_sse(self.torch_result, self.config)
        self.sse_ratio = self.torch_sse / (self.scipy_sse + 1e-12)

    def test_row_count_matches(self) -> None:
        assert len(self.torch_result) == len(self.scipy_result)

    def test_torch_sse_not_worse_than_scipy(self) -> None:
        """PyTorch SSE must not be meaningfully larger than scipy SSE.

        Both optimizers minimise the same OLS objective — the PyTorch result
        is acceptable when its residual is not meaningfully larger.  We use an
        *absolute* excess rather than a pure ratio, because both SSEs can be
        near-machine-zero for easy curves, making the ratio unstable.

        A PyTorch fit is flagged as clearly worse only when it has both a
        larger ratio *and* an absolute excess above 1e-4 (i.e. the difference
        is numerically meaningful, not just floating-point noise).
        """
        abs_excess = self.torch_sse - self.scipy_sse  # negative = torch is better
        pct_clearly_worse = (
            (self.sse_ratio > 1.5) & (abs_excess > 1e-4)
        ).mean() * 100
        assert pct_clearly_worse < 5, (
            f"{pct_clearly_worse:.1f}% of curves have meaningfully worse SSE than scipy"
        )

    def test_torch_sse_median_at_parity_with_scipy(self) -> None:
        """Median SSE ratio ≈ 1 means PyTorch finds equally good fits on average."""
        median_ratio = float(np.median(self.sse_ratio))
        assert median_ratio < 1.05, f"Median SSE ratio torch/scipy: {median_ratio:.4f}"

    def test_r2_at_parity_with_scipy(self) -> None:
        """PyTorch R² should not be meaningfully lower than scipy R²."""
        r2_diff = (self.scipy_result["Curve R2"] - self.torch_result["Curve R2"])
        # Positive means scipy is better; allow median degradation < 0.01
        assert float(r2_diff.median()) < 0.01, (
            f"Median R² degradation (scipy − torch): {r2_diff.median():.4f}"
        )

    def test_regulation_column_present(self) -> None:
        assert "Curve Regulation" in self.torch_result.columns

    def test_curve_auc_is_finite(self) -> None:
        assert self.torch_result["Curve AUC"].notna().all()

    def test_r2_is_non_negative(self) -> None:
        assert (self.torch_result["Curve R2"] >= 0).all()

    def test_pec50_in_valid_range(self) -> None:
        log_doses = np.log10(DOSES_UM * DOSE_SCALE)
        pec50_lo = -log_doses.max() - 2.0
        pec50_hi = -log_doses.min() + 2.0
        assert (self.torch_result["pEC50"] >= pec50_lo).all()
        assert (self.torch_result["pEC50"] <= pec50_hi).all()

    @pytest.mark.skipif(
        not __import__("torch").cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_sse_matches_cpu(self, synthetic_data: dict) -> None:
        """CUDA and CPU runs should find the same SSE (same deterministic starts)."""
        cuda_result = run_pipeline_api(synthetic_data["config"], device="cuda")
        cuda_sse = _compute_sse(cuda_result, synthetic_data["config"])
        ratio = cuda_sse / (self.torch_sse + 1e-12)
        assert float(np.median(ratio)) < 1.05, (
            f"Median CUDA/CPU SSE ratio: {np.median(ratio):.4f}"
        )
