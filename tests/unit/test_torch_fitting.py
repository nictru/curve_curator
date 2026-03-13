"""test_torch_fitting.py
Numerical equivalence tests for the PyTorch 4PL fitting backend.

Uses the same synthetic data as ``TestFitting`` in ``test_model_logistic.py``
to verify that the PyTorch LBFGS backend recovers known ground-truth parameters.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from curve_curator.torch_fitting import _sigmoid_4pl, batch_fit_4pl, _SLOPE_LO, _SLOPE_HI


# ---------------------------------------------------------------------------
# Helpers matching test_model_logistic.py TestFitting setup
# ---------------------------------------------------------------------------
_TRUE_PARAMS = {"pec50": 7.0, "slope": 1.0, "front": 1.0, "back": 0.1}

# Doses: 6 concentrations in µM (0.001 to 100), scaled to M (multiply by 1e-6)
# log10(dose * dose_scale) = log10(dose * 1e-6) = log10(dose) + log10(1e-6)
# At dose=1µM: log10(1e-6) = -6
# At dose=1nM: log10(1e-9) = -9
DOSES_UM = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
DOSE_SCALE = 1e-6
_X = np.log10(DOSES_UM * DOSE_SCALE)  # log10(M) from -9 to -4, sorted ascending

_Y_NOISELESS = (
    (_TRUE_PARAMS["front"] - _TRUE_PARAMS["back"])
    / (1 + 10 ** (_TRUE_PARAMS["slope"] * (_X + _TRUE_PARAMS["pec50"])))
    + _TRUE_PARAMS["back"]
)


def _make_config(doses_real: np.ndarray) -> dict:
    """Minimal CurveCurator config for the given real-space dose array."""
    n = len(doses_real)
    ctrl_idx = 0
    return {
        "Experiment": {
            "experiments": list(range(n + 1)),  # +1 for synthetic control
            "control_experiment": [ctrl_idx],
            "doses": [0.0] + list(doses_real),
            "dose_scale": str(DOSE_SCALE),
        },
        "Processing": {
            "available_cores": 1,
            "max_missing": 0,
            "imputation": False,
            "imputation_pct": 0.005,
            "max_imputation": 0,
            "normalization": False,
            "ratio_range": None,
        },
        "Curve Fit": {
            "type": "OLS",
            "speed": "standard",
            "weights": None,
            "interpolation": False,
            "control_fold_change": False,
            "slope": None,
            "front": None,
            "back": None,
            "max_iterations": 1000,
        },
        "F Statistic": {
            "optimized_dofs": True,
            "scale": 1.0,
            "loc": 0.12,
        },
        "Paths": {"input_file": "/tmp/test.tsv", "output": "/tmp"},
    }


def _make_df(y_values: np.ndarray, config: dict) -> pd.DataFrame:
    """Build the preprocessed DataFrame that batch_fit_4pl expects."""
    experiments = np.array(config["Experiment"]["experiments"])
    control_experiments = np.array(config["Experiment"]["control_experiment"])
    drug_concs = np.array(config["Experiment"]["doses"])
    control_mask = drug_concs != 0.0
    n_dose = control_mask.sum()

    # Build Ratio columns for dose points (control already divided out = y_values)
    # and also add Signal Quality
    ratio_dose_cols = [f"Ratio {i}" for i in experiments[control_mask]]
    ratio_ctrl_cols = [f"Ratio {i}" for i in control_experiments]
    raw_ctrl_cols = [f"Raw {i}" for i in control_experiments]

    rows = []
    for y in y_values:
        row: dict = {"Name": "CVCL_0000|100"}
        for col, val in zip(ratio_dose_cols, y):
            row[col] = val
        for col in ratio_ctrl_cols:
            row[col] = 1.0
        for col in raw_ctrl_cols:
            row[col] = 1.0
        rows.append(row)

    df = pd.DataFrame(rows)
    df["Signal Quality"] = np.log2(1.0)
    return df


# ---------------------------------------------------------------------------
# Core 4PL function tests
# ---------------------------------------------------------------------------


class TestSigmoid4pl:
    def test_at_inflection_point(self) -> None:
        """At x=-pec50, y should be (front+back)/2."""
        pec50 = torch.tensor([7.0])
        slope = torch.tensor([1.0])
        front = torch.tensor([1.0])
        back = torch.tensor([0.1])
        x = -pec50[:, None]  # (1,1)
        y = _sigmoid_4pl(x, pec50, slope, front, back)
        expected = (front + back) / 2
        assert torch.allclose(y.squeeze(), expected.squeeze(), atol=1e-6)

    def test_shape(self) -> None:
        B, D = 10, 7
        x = torch.zeros(B, D)
        params = torch.ones(B)
        y = _sigmoid_4pl(x, params, params, params * 1.0, params * 0.1)
        assert y.shape == (B, D)

    def test_matches_numpy_formula(self) -> None:
        """Verify PyTorch matches the original numpy formula element-wise."""
        y_torch = _sigmoid_4pl(
            torch.tensor([_X], dtype=torch.float64),
            torch.tensor([7.0]),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            torch.tensor([0.1]),
        )
        assert np.allclose(y_torch.squeeze().numpy(), _Y_NOISELESS, atol=1e-8)


# ---------------------------------------------------------------------------
# Fitting accuracy tests
# ---------------------------------------------------------------------------


class TestBatchFit4pl:
    """Verify parameter recovery on noiseless synthetic 4PL data."""

    @pytest.fixture()
    def config(self) -> dict:
        return _make_config(DOSES_UM)

    @pytest.fixture()
    def df_single(self, config: dict) -> pd.DataFrame:
        return _make_df(_Y_NOISELESS[np.newaxis, :], config)

    def test_pec50_recovery_cpu(self, df_single: pd.DataFrame, config: dict) -> None:
        result = batch_fit_4pl(df_single, config, device="cpu")
        assert abs(result["pEC50"].iloc[0] - _TRUE_PARAMS["pec50"]) < 0.1

    def test_slope_recovery_cpu(self, df_single: pd.DataFrame, config: dict) -> None:
        """Slope should be in the valid range [0.01, 10] after fitting."""
        result = batch_fit_4pl(df_single, config, device="cpu")
        assert _SLOPE_LO <= result["Curve Slope"].iloc[0] <= _SLOPE_HI

    def test_front_recovery_cpu(self, df_single: pd.DataFrame, config: dict) -> None:
        result = batch_fit_4pl(df_single, config, device="cpu")
        assert abs(result["Curve Front"].iloc[0] - _TRUE_PARAMS["front"]) < 0.3

    def test_back_recovery_cpu(self, df_single: pd.DataFrame, config: dict) -> None:
        result = batch_fit_4pl(df_single, config, device="cpu")
        assert abs(result["Curve Back"].iloc[0] - _TRUE_PARAMS["back"]) < 0.3

    def test_output_columns_present(self, df_single: pd.DataFrame, config: dict) -> None:
        result = batch_fit_4pl(df_single, config, device="cpu")
        expected_cols = [
            "pEC50", "Curve Slope", "Curve Front", "Curve Back",
            "Curve Fold Change", "Curve AUC", "Curve RMSE", "Curve R2",
            "Curve F_Value", "Curve P_Value", "Curve Log P_Value",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_batch_recovery(self, config: dict) -> None:
        """Multiple curves in a single batch should all converge."""
        # 5 copies of the same noiseless curve
        y_batch = np.tile(_Y_NOISELESS, (5, 1))
        df = _make_df(y_batch, config)
        result = batch_fit_4pl(df, config, device="cpu")
        assert len(result) == 5
        assert all(abs(result["pEC50"] - _TRUE_PARAMS["pec50"]) < 0.1)

    def test_r2_close_to_one_on_noiseless_data(
        self, df_single: pd.DataFrame, config: dict
    ) -> None:
        result = batch_fit_4pl(df_single, config, device="cpu")
        assert result["Curve R2"].iloc[0] > 0.9

    def test_f_value_is_positive(self, df_single: pd.DataFrame, config: dict) -> None:
        result = batch_fit_4pl(df_single, config, device="cpu")
        assert result["Curve F_Value"].iloc[0] > 0

    def test_p_value_is_significant_on_noiseless_data(
        self, df_single: pd.DataFrame, config: dict
    ) -> None:
        result = batch_fit_4pl(df_single, config, device="cpu")
        assert result["Curve P_Value"].iloc[0] < 0.01

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pec50_recovery_cuda(self, df_single: pd.DataFrame, config: dict) -> None:
        result = batch_fit_4pl(df_single, config, device="cuda")
        assert abs(result["pEC50"].iloc[0] - _TRUE_PARAMS["pec50"]) < 0.1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_and_cpu_give_same_result(
        self, df_single: pd.DataFrame, config: dict
    ) -> None:
        result_cpu = batch_fit_4pl(df_single, config, device="cpu")
        result_cuda = batch_fit_4pl(df_single, config, device="cuda")
        assert abs(result_cpu["pEC50"].iloc[0] - result_cuda["pEC50"].iloc[0]) < 0.05

    def test_cpu_fallback_on_unavailable_device(
        self, df_single: pd.DataFrame, config: dict
    ) -> None:
        """Requesting a non-existent device should silently fall back to CPU."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = batch_fit_4pl(df_single, config, device="cuda:99")
        assert "pEC50" in result.columns
