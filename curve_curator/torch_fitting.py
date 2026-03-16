"""torch_fitting.py
Batched 4-parameter log-logistic (4PL) curve fitting using PyTorch.

Replaces the per-row scipy-based fitting in quantification.run_pipeline with a
vectorised Adam optimisation pass that runs transparently on CPU or GPU.

The public entry point is ``batch_fit_4pl(df, config, device)``, which accepts
the preprocessed DataFrame produced by ``quantification._preprocess`` and
returns a DataFrame augmented with exactly the same fitted-curve columns that
``add_logistic_model`` previously produced.

Mathematical conventions follow models.py exactly:
  y = (front - back) / (1 + 10^(slope * (x + pec50))) + back

where x is log10(concentration) and pec50 = -log10(EC50).

Optimizer choice — Adam instead of L-BFGS
------------------------------------------
L-BFGS uses a *shared* Hessian history (s/y pairs) for the entire flattened
parameter vector [pec50_0, ..., pec50_{BN}, slope_0, ...].  When B curves are
batched, the aggregate curvature information from all curves is conflated.  The
strong Wolfe line search then picks a single step size that satisfies the Wolfe
conditions for the aggregate loss but overshoots individual curves, causing most
pEC50 values to collapse to the upper boundary.  Empirically this failure mode
activates at B ≥ 5.

Adam maintains *per-parameter independent* first/second moment estimates
(m_i, v_i).  The effective step for parameter θ_j (belonging to curve j)
depends only on the gradient history of θ_j — there is no cross-curve
curvature contamination, even when all BN parameters share one flat tensor.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from scipy import stats

from . import toolbox as tool

if TYPE_CHECKING:
    from torch import Tensor

# ---------------------------------------------------------------------------
# Global constants (match models.py)
# ---------------------------------------------------------------------------
_PEC50_DELTA = 2.0  # pEC50 search range beyond observed dose range
_SLOPE_INIT = 10.0  # initial slope guess (max of SLOPE_LIMITS)
_SLOPE_LO = 0.01
_SLOPE_HI = 10.0
_Y_LO = 1e-4
_Y_HI = 1e6
_LOG10 = math.log(10.0)

# Adam hyper-parameters
_ADAM_ITERATIONS = 1000
_ADAM_LR = 0.1
_ADAM_ETA_MIN = 1e-5


# ---------------------------------------------------------------------------
# 4PL model (vectorised over a batch dimension)
# ---------------------------------------------------------------------------


def _sigmoid_4pl(x: Tensor, pec50: Tensor, slope: Tensor, front: Tensor, back: Tensor) -> Tensor:
    """Batched 4-parameter log-logistic.

    Parameters
    ----------
    x:      (B, D) — log10 drug concentrations (B curves, D dose points)
    pec50:  (B,)
    slope:  (B,)
    front:  (B,)
    back:   (B,)

    Returns
    -------
    y:      (B, D)
    """
    exponent = slope[:, None] * (x + pec50[:, None])
    return (front - back)[:, None] / (1.0 + torch.pow(torch.full_like(exponent, 10.0), exponent)) + back[:, None]


# ---------------------------------------------------------------------------
# AUC via analytical antiderivative (matches models.py line 873)
# ---------------------------------------------------------------------------


def _primitive(x: Tensor, pec50: Tensor, slope: Tensor, front: Tensor, back: Tensor) -> Tensor:
    """Antiderivative F(x) of the 4PL function (no constant).

    F(x) = front*x - (front-back)/slope * log10(1 + 10^(slope*(x+pec50)))
    """
    log10_term = torch.log10(1.0 + torch.pow(torch.full_like(x, 10.0), slope[:, None] * (x + pec50[:, None])))
    return front[:, None] * x - (front - back)[:, None] / slope[:, None] * log10_term


def _auc(x0: Tensor, x1: Tensor, pec50: Tensor, slope: Tensor, front: Tensor, back: Tensor) -> Tensor:
    """Normalised AUC over [x0, x1] (scalar per batch element).

    Matches LogisticModel.calculate_auc with intercept=1.
    """
    f1 = _primitive(x1[:, None], pec50, slope, front, back).squeeze(1)
    f0 = _primitive(x0[:, None], pec50, slope, front, back).squeeze(1)
    curve_area = f1 - f0
    reference_area = x1 - x0  # intercept=1
    return curve_area / reference_area


# ---------------------------------------------------------------------------
# Vectorised initial guesses (mirrors alternative_guesses in models.py)
# ---------------------------------------------------------------------------


def _make_initial_guesses(x_cpu: np.ndarray, y: Tensor, pec50_lo: float, pec50_hi: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build the single best initial guess for each curve by OLS cost.

    Returns four tensors of shape (B,): best_pec50, best_slope, best_front, best_back.
    Used internally by ``_all_initial_guesses`` and as a standalone helper.
    """
    pec50_cands, front_cands, back_cands = _all_initial_guesses(x_cpu, y, pec50_lo, pec50_hi)
    n_cand, B = pec50_cands.shape
    D = y.shape[1]
    device = y.device
    dtype = y.dtype

    x_t = torch.tensor(x_cpu, dtype=dtype, device=device)
    slope_cands = torch.full((n_cand, B), _SLOPE_INIT, device=device, dtype=dtype)

    x_exp = x_t[None, None, :].expand(n_cand, B, D)
    pec50_exp = pec50_cands[:, :, None].expand(n_cand, B, D)
    slope_exp = slope_cands[:, :, None].expand(n_cand, B, D)
    front_exp = front_cands[:, :, None].expand(n_cand, B, D)
    back_exp = back_cands[:, :, None].expand(n_cand, B, D)

    exponent = slope_exp * (x_exp + pec50_exp)
    y_pred = (front_exp - back_exp) / (1.0 + torch.pow(torch.full_like(exponent, 10.0), exponent)) + back_exp
    y_obs_exp = y[None, :, :].expand(n_cand, B, D)
    nan_mask = torch.isfinite(y_obs_exp)
    sq_err = ((y_pred - y_obs_exp) ** 2) * nan_mask
    sse = sq_err.sum(dim=2)  # (n_cand, B)

    best_idx = sse.argmin(dim=0)  # (B,)
    best_pec50 = pec50_cands.gather(0, best_idx[None, :]).squeeze(0)
    best_front = front_cands.gather(0, best_idx[None, :]).squeeze(0)
    best_back = back_cands.gather(0, best_idx[None, :]).squeeze(0)
    best_slope = slope_cands.gather(0, best_idx[None, :]).squeeze(0)
    return best_pec50, best_slope, best_front, best_back


def _all_initial_guesses(x_cpu: np.ndarray, y: Tensor, pec50_lo: float, pec50_hi: float) -> tuple[Tensor, Tensor, Tensor]:
    """Build *all* candidate (pec50, front, back) starting guesses.

    Returns three tensors of shape (n_cand, B) — one row per candidate.
    Mirrors ``alternative_guesses`` in models.py without any reduction.
    """
    B, D = y.shape
    device = y.device
    dtype = y.dtype

    y_mean = y.nanmean(dim=1)  # (B,)

    candidates_pec50: list[Tensor] = []
    candidates_front: list[Tensor] = []
    candidates_back: list[Tensor] = []

    # null guess
    candidates_pec50.append(-torch.full((B,), float(np.median(x_cpu)), device=device, dtype=dtype))
    candidates_front.append(y_mean)
    candidates_back.append(y_mean)

    # first outside guess
    pec50_first_outside = -(x_cpu[0] - (x_cpu[1] - x_cpu[0]) / 2.0)
    candidates_pec50.append(torch.full((B,), pec50_first_outside, device=device, dtype=dtype))
    candidates_front.append(torch.ones(B, device=device, dtype=dtype))
    candidates_back.append(y_mean)

    # piecewise guesses for each split position
    for n in range(1, D):
        pec50_guess = -(float(x_cpu[n - 1]) + float(x_cpu[n])) / 2.0
        front_guess = y[:, :n].nanmean(dim=1)
        back_guess = y[:, n:].nanmean(dim=1)
        candidates_pec50.append(torch.full((B,), pec50_guess, device=device, dtype=dtype))
        candidates_front.append(front_guess)
        candidates_back.append(back_guess)

    # last outside guess
    pec50_last_outside = -(x_cpu[-1] + (x_cpu[-1] - x_cpu[-2]) / 2.0)
    candidates_pec50.append(torch.full((B,), pec50_last_outside, device=device, dtype=dtype))
    candidates_front.append(y_mean)
    candidates_back.append(torch.ones(B, device=device, dtype=dtype))

    # Stack and clamp: shape (n_cand, B)
    pec50_cands = torch.clamp(torch.stack(candidates_pec50, dim=0), pec50_lo, pec50_hi)
    front_cands = torch.clamp(torch.stack(candidates_front, dim=0), _Y_LO, _Y_HI)
    back_cands = torch.clamp(torch.stack(candidates_back, dim=0), _Y_LO, _Y_HI)
    return pec50_cands, front_cands, back_cands


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------


def batch_fit_4pl(
    df: pd.DataFrame,
    config: dict,
    *,
    device: str | torch.device = "cpu",
    gpu_chunk_size: int = 50_000,
) -> pd.DataFrame:
    """Fit 4PL curves to all rows in *df* using batched Adam on *device*.

    *df* must already be preprocessed (ratio columns computed, doses sorted
    low-to-high).  This matches the output of ``quantification._preprocess``.

    Parameters
    ----------
    df:
        Preprocessed DataFrame with ``Ratio {i}`` columns and ``Signal Quality``.
    config:
        CurveCurator config dict (after ``toml_parser.set_default_values``).
    device:
        PyTorch device string or object, e.g. ``"cpu"``, ``"cuda"``,
        ``"cuda:0"``, ``"mps"``.  Falls back to CPU if the requested device
        is unavailable.
    gpu_chunk_size:
        Maximum number of curves per GPU sub-batch.  When ``B > gpu_chunk_size``
        on a non-CPU device, the DataFrame is split into sequential sub-batches
        of at most *gpu_chunk_size* rows, each processed independently, and the
        results are concatenated.  This bounds peak VRAM regardless of group
        size.  Has no effect on CPU.  Default: 50,000.

    Returns
    -------
    pd.DataFrame
        *df* augmented with the same columns that ``add_logistic_model``
        previously produced (``pEC50``, ``Curve Slope``, ``Curve Front``,
        ``Curve Back``, ``Curve Fold Change``, ``Curve AUC``, ``Curve RMSE``,
        ``Curve R2``, ``pEC50 Error``, ``Curve Slope Error``,
        ``Curve Front Error``, ``Curve Back Error``, ``Null Model``,
        ``Null RMSE``, ``Curve F_Value``, ``Curve P_Value``,
        ``Curve Log P_Value``).
    """
    # ------------------------------------------------------------------
    # Resolve device (fall back to CPU if CUDA/MPS not available)
    # "auto" selects CUDA > MPS > CPU in priority order.
    # ------------------------------------------------------------------
    import warnings

    if device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        try:
            dev = torch.device(device)
            if dev.type == "cuda" and not torch.cuda.is_available():
                warnings.warn("CUDA requested but not available — falling back to CPU.", RuntimeWarning, stacklevel=2)
                dev = torch.device("cpu")
            elif dev.type == "cuda" and torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
                gpu_idx = dev.index if dev.index is not None else 0
                if gpu_idx >= n_gpus:
                    warnings.warn(
                        f"CUDA device {device} out of range (have {n_gpus} GPU(s)) — falling back to CPU.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    dev = torch.device("cpu")
            elif dev.type == "mps" and not torch.backends.mps.is_available():
                warnings.warn("MPS requested but not available — falling back to CPU.", RuntimeWarning, stacklevel=2)
                dev = torch.device("cpu")
        except Exception:
            dev = torch.device("cpu")

    B = len(df)

    # Sub-batching: split oversized groups into sequential chunks so that
    # peak memory per chunk stays bounded regardless of group size.
    # Applied on both GPU (to bound VRAM) and CPU (to avoid single huge
    # tensor allocations that can crash the process).
    if B > gpu_chunk_size:
        chunks = [
            df.iloc[i : i + gpu_chunk_size]
            for i in range(0, B, gpu_chunk_size)
        ]
        return pd.concat(
            [_batch_fit_4pl_inner(chunk, config, dev=dev) for chunk in chunks]
        )

    return _batch_fit_4pl_inner(df, config, dev=dev)


def _batch_fit_4pl_inner(df: pd.DataFrame, config: dict, *, dev: torch.device) -> pd.DataFrame:
    """Core 4PL fitting logic for a single chunk on a resolved device.

    Called by ``batch_fit_4pl`` — either directly (small batch) or once per
    sub-batch (large GPU group).  VRAM is explicitly freed at the end of each
    call so sequential sub-batches do not accumulate GPU memory.
    """
    import gc

    dtype = torch.float64

    # ------------------------------------------------------------------
    # Build dose vector (log10 space, sorted, excluding control at dose=0)
    # ------------------------------------------------------------------
    experiments = np.array(config["Experiment"]["experiments"])
    control_experiments = np.array(config["Experiment"]["control_experiment"])
    drug_concs = np.array(config["Experiment"]["doses"])
    drug_scale = float(config["Experiment"]["dose_scale"])
    control_mask = drug_concs != 0.0

    drug_log_concs = tool.build_drug_log_concentrations(drug_concs[control_mask], drug_scale)
    sorted_dose_idx = np.argsort(drug_log_concs)
    drug_log_concs_sorted = drug_log_concs[sorted_dose_idx]
    cols_ratio = tool.build_col_names("Ratio {}", experiments)
    col_ratio_control = tool.build_col_names("Ratio {}", control_experiments)
    cols_ratio_dose = tool.build_col_names("Ratio {}", experiments[control_mask][sorted_dose_idx])

    # ------------------------------------------------------------------
    # Bounds on pEC50
    # ------------------------------------------------------------------
    pec50_lo = float(-drug_log_concs_sorted.max() - _PEC50_DELTA)
    pec50_hi = float(-drug_log_concs_sorted.min() + _PEC50_DELTA)

    # ------------------------------------------------------------------
    # Extract ratio matrix — shape (B, D) where D = n dose points
    # ------------------------------------------------------------------
    y_np = df[cols_ratio_dose].to_numpy(dtype=np.float64)  # (B, D)
    B, D = y_np.shape

    if B == 0:
        return _add_nan_cols(df)

    y_obs = torch.tensor(y_np, dtype=dtype, device=dev)
    nan_mask = torch.isfinite(y_obs)  # (B, D) True where valid

    x_np = drug_log_concs_sorted  # (D,)
    x = torch.tensor(x_np, dtype=dtype, device=dev)  # (D,)

    # ------------------------------------------------------------------
    # Control ratio (prepend synthetic control at x=-inf → y=1 for fold change)
    # Build full x and y including the synthetic control for F-stat computation
    # We prepend x_ctrl = -inf-equivalent (very negative) and y_ctrl = 1
    # ------------------------------------------------------------------
    y_ctrl = torch.ones(B, 1, dtype=dtype, device=dev)
    x_ctrl_val = float(drug_log_concs_sorted.min() - 3.0)  # 3 log-units below lowest dose
    x_full = torch.cat([torch.tensor([x_ctrl_val], dtype=dtype, device=dev), x], dim=0)  # (D+1,)
    y_full_obs = torch.cat([y_ctrl, y_obs], dim=1)  # (B, D+1)
    nan_mask_full = torch.isfinite(y_full_obs)  # (B, D+1)

    # ------------------------------------------------------------------
    # Multi-start: generate ALL initial guesses × slope seeds (mirrors
    # extensively_fit_guesses_ols from models.py).
    # We expand the batch to (B × n_starts) where n_starts = n_cand × n_slopes,
    # run one vectorised Adam pass, then reduce by selecting the start with the
    # lowest final OLS cost per original curve.
    # ------------------------------------------------------------------
    # Slope seeds (same as extensively_fit_guesses_ols default)
    slope_seeds = [_SLOPE_LO, 1.0, _SLOPE_HI]  # 3 seeds
    n_slopes = len(slope_seeds)

    # Get all candidate (pec50, front, back) guesses: each (n_cand, B)
    pec50_all, front_all, back_all = _all_initial_guesses(x_np, y_obs, pec50_lo, pec50_hi)
    n_cand = pec50_all.shape[0]

    # Expand for each slope seed: shape (n_cand × n_slopes, B)
    pec50_starts = pec50_all.repeat_interleave(n_slopes, dim=0)  # (n_cand*n_slopes, B)
    front_starts = front_all.repeat_interleave(n_slopes, dim=0)
    back_starts = back_all.repeat_interleave(n_slopes, dim=0)
    slope_seed_t = torch.tensor(slope_seeds, dtype=dtype, device=dev)  # (n_slopes,)
    slope_starts = slope_seed_t.repeat(n_cand).unsqueeze(1).expand(-1, B)  # (n_cand*n_slopes, B)

    n_starts = pec50_starts.shape[0]  # n_cand × n_slopes

    # Flatten to a single mega-batch of size (B × n_starts) for one Adam pass
    # Layout: [curve_0_start_0, curve_1_start_0, ..., curve_B_start_0, curve_0_start_1, ...]
    # We index as [start_i * B + curve_j]
    BN = B * n_starts  # total parameter vectors

    # Shape (BN, D): interleave so that indices [0, B, 2B, ...] are start-0 for each curve
    # Actually keep layout as (n_starts * B,) for simplicity
    y_obs_exp = y_obs.repeat(n_starts, 1)        # (BN, D)
    nan_mask_exp = nan_mask.repeat(n_starts, 1)  # (BN, D)
    x_exp = x[None, :].expand(BN, D)             # (BN, D)

    # Full data (dose + synthetic control) for best-start selection.
    # The synthetic control (y=1.0 at x_ctrl) is included so that degenerate
    # solutions with large front values that fit the dose data but extrapolate
    # poorly to the control point are penalised during start selection.
    y_full_exp = y_full_obs.repeat(n_starts, 1)           # (BN, D+1)
    nan_mask_full_exp = nan_mask_full.repeat(n_starts, 1)  # (BN, D+1)
    x_full_exp = x_full[None, :].expand(BN, D + 1)        # (BN, D+1)

    pec50_ms = pec50_starts.reshape(-1).clone().requires_grad_(True)  # (BN,)
    slope_ms = slope_starts.reshape(-1).clone().requires_grad_(True)
    front_ms = front_starts.reshape(-1).clone().requires_grad_(True)
    back_ms = back_starts.reshape(-1).clone().requires_grad_(True)

    # Adam with cosine LR annealing.
    #
    # L-BFGS cannot be used here: it maintains a *shared* Hessian history for
    # the entire flat parameter vector, which conflates curvature from all B
    # curves and causes the line search to collapse pEC50 values to the upper
    # boundary for B ≥ 5 (empirically confirmed).  Adam's per-parameter m/v
    # estimates are fully independent, so there is no cross-curve contamination.
    optimizer_ms = torch.optim.Adam(
        [pec50_ms, slope_ms, front_ms, back_ms], lr=_ADAM_LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_ms, T_max=_ADAM_ITERATIONS, eta_min=_ADAM_ETA_MIN
    )
    for _ in range(_ADAM_ITERATIONS):
        optimizer_ms.zero_grad()
        y_pred = _sigmoid_4pl(x_exp, pec50_ms, slope_ms, front_ms, back_ms)
        residuals = (y_pred - y_obs_exp) * nan_mask_exp
        loss = (residuals ** 2).sum()
        loss.backward()
        optimizer_ms.step()
        scheduler.step()
        # Project parameters back into bounds after each step to prevent the
        # optimiser from wandering into regions where the sigmoid is numerically
        # degenerate (e.g. pEC50 >> pec50_hi → flat curve).
        with torch.no_grad():
            pec50_ms.data.clamp_(pec50_lo, pec50_hi)
            slope_ms.data.clamp_(_SLOPE_LO, _SLOPE_HI)
            front_ms.data.clamp_(_Y_LO, _Y_HI)
            back_ms.data.clamp_(_Y_LO, _Y_HI)

    # ------------------------------------------------------------------
    # Select best start per original curve by final OLS cost over full data
    # (dose points + synthetic control).  Using full-data SSE prevents
    # degenerate solutions that fit doses well but extrapolate poorly to the
    # control (e.g. front >> 1) from being selected as the best start.
    # ------------------------------------------------------------------
    with torch.no_grad():
        y_pred_full_all = _sigmoid_4pl(x_full_exp, pec50_ms, slope_ms, front_ms, back_ms)
        sse_full_all = ((y_pred_full_all - y_full_exp) ** 2 * nan_mask_full_exp).sum(dim=1)  # (BN,)
        # Reshape to (n_starts, B) and argmin over starts
        sse_mat = sse_full_all.reshape(n_starts, B)  # (n_starts, B)
        best_start = sse_mat.argmin(dim=0)            # (B,) indices in [0, n_starts)

        # Gather best parameters per curve
        idx = (best_start * B + torch.arange(B, device=dev))  # flat indices
        pec50_f = pec50_ms.data[idx]
        slope_f = slope_ms.data[idx]
        front_f = front_ms.data[idx]
        back_f = back_ms.data[idx]

    # ------------------------------------------------------------------
    # Derived statistics (all on device, moved to CPU for DataFrame)
    # ------------------------------------------------------------------
    with torch.no_grad():
        # Predictions over full range (including control)
        y_pred_full = _sigmoid_4pl(x_full[None, :].expand(B, D + 1), pec50_f, slope_f, front_f, back_f)

        # n_valid per curve (for DOFs)
        n_valid = nan_mask_full.sum(dim=1).float()  # (B,)

        # M1 residuals & SSE
        m1_resid = (y_pred_full - y_full_obs) * nan_mask_full
        m1_sse = (m1_resid ** 2).sum(dim=1)  # (B,)

        # Null model: mean of y (controls contribute y=1 with weight 1)
        y_for_mean = y_full_obs.clone()
        y_for_mean[~nan_mask_full] = float("nan")
        y_mean_null = y_for_mean.nanmean(dim=1)  # (B,)
        m0_sse = ((y_full_obs - y_mean_null[:, None]) ** 2 * nan_mask_full).sum(dim=1)  # (B,)
        m0_sse = m0_sse + 1e-20

        # RMSE M1 (NaN where n_valid == 0)
        n_params = torch.tensor(4.0, dtype=dtype, device=dev)
        rmse = torch.sqrt(m1_sse / (n_valid + 1e-20))

        # R²
        ss_tot = ((y_full_obs - y_mean_null[:, None]) ** 2 * nan_mask_full).sum(dim=1)
        r2 = 1.0 - m1_sse / (ss_tot + 1e-20)
        r2 = r2.clamp(max=1.0)

        # Null RMSE
        null_rmse = torch.sqrt(m0_sse / n_valid.clamp(min=1))

        # Null model intercept
        null_intercept = y_mean_null

        # Fold change: log2(y at max dose / y at min dose), not-to-control
        y_at_max = _sigmoid_4pl(x[-1:][None, :].expand(B, 1), pec50_f, slope_f, front_f, back_f).squeeze(1)
        y_at_min = _sigmoid_4pl(x[:1][None, :].expand(B, 1), pec50_f, slope_f, front_f, back_f).squeeze(1)
        fold_change = torch.log2(y_at_max) - torch.log2(y_at_min)

        # AUC
        x0_batch = x[0].expand(B)
        x1_batch = x[-1].expand(B)
        auc = _auc(x0_batch, x1_batch, pec50_f, slope_f, front_f, back_f)

        # F-statistic (matches fit_model in quantification.py exactly)
        m1_sse_s = m1_sse + 1e-20
        n_int = n_valid  # total data points including synthetic control
        f_stat = (m0_sse - m1_sse_s) / m1_sse_s * (n_int / n_params)
        f_stat = f_stat.clamp(min=0.0)

    # Move all results to CPU numpy arrays before releasing device tensors.
    def _np(t: Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    pec50_np = _np(pec50_f)
    slope_np = _np(slope_f)
    front_np = _np(front_f)
    back_np = _np(back_f)
    fold_change_np = _np(fold_change)
    auc_np = _np(auc)
    rmse_np = _np(rmse)
    r2_np = _np(r2)
    null_intercept_np = _np(null_intercept)
    null_rmse_np = _np(null_rmse)
    f_stat_np = _np(f_stat)
    n_valid_np = _np(n_valid).astype(int)

    # Release device tensors and flush allocator cache.
    del optimizer_ms, scheduler
    del (
        y_obs, nan_mask, x, y_ctrl, x_full, y_full_obs, nan_mask_full,
        pec50_all, front_all, back_all,
        pec50_starts, front_starts, back_starts, slope_starts,
        y_obs_exp, nan_mask_exp, x_exp,
        y_full_exp, nan_mask_full_exp, x_full_exp,
        pec50_ms, slope_ms, front_ms, back_ms,
        y_pred_full_all, sse_full_all, sse_mat, best_start, idx,
        pec50_f, slope_f, front_f, back_f,
        y_pred_full, n_valid, m1_resid, m1_sse,
        y_for_mean, y_mean_null, m0_sse,
        rmse, r2, null_rmse, null_intercept,
        y_at_max, y_at_min, fold_change, auc, f_stat,
    )
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
    elif dev.type == "mps":
        torch.mps.empty_cache()

    # ------------------------------------------------------------------
    # p-values via scipy (CPU, vectorised)
    # ------------------------------------------------------------------
    f_stat_params = config["F Statistic"]
    scale = f_stat_params["scale"]
    loc = f_stat_params["loc"]
    optimized_dofs = f_stat_params.get("optimized_dofs", True)

    # DOFs — replicate LogisticModel.get_dofs(n, optimized=True)
    def _get_dofs(n: int) -> tuple[float, float]:
        if optimized_dofs:
            def _low_n_slope_adjustment(n_val: int) -> float:
                return 1.0 / ((n_val - 4) ** 4 / n_val + 4)
            slope_dof = 0.8
            dfn = 5.0
            dfd = (slope_dof - _low_n_slope_adjustment(n)) * (n - 2.5)
            return dfn, dfd
        dfn = 3.0  # n_parameter - 1 = 4 - 1
        dfd = float(n - 4)  # n - n_parameter
        return dfn, dfd

    p_values = np.empty(B)
    for i in range(B):
        n_i = int(n_valid_np[i])
        if n_i <= 4:
            p_values[i] = np.nan
            f_stat_np[i] = np.nan
        else:
            dfn_i, dfd_i = _get_dofs(n_i)
            # Override with config if present
            dfn_i = f_stat_params.get("dfn", dfn_i)
            dfd_i = f_stat_params.get("dfd", dfd_i)
            p_values[i] = stats.f.sf(float(f_stat_np[i]), dfn=dfn_i, dfd=dfd_i, scale=scale, loc=loc)

    # Parameter errors — not available analytically from first-order optimizers, set to NaN
    # (matches the case when the covariance matrix cannot be computed)
    pec50_err = np.full(B, np.nan)
    slope_err = np.full(B, np.nan)
    front_err = np.full(B, np.nan)
    back_err = np.full(B, np.nan)

    # ------------------------------------------------------------------
    # Assemble output columns (same as add_logistic_model)
    # ------------------------------------------------------------------
    fit_cols = [
        "pEC50", "Curve Slope", "Curve Front", "Curve Back",
        "Curve Fold Change", "Curve AUC", "Curve RMSE", "Curve R2",
        "pEC50 Error", "Curve Slope Error", "Curve Front Error", "Curve Back Error",
        "Null Model", "Null RMSE", "Curve F_Value", "Curve P_Value",
    ]
    fit_data = np.column_stack([
        pec50_np, slope_np, front_np, back_np,
        fold_change_np, auc_np, rmse_np, r2_np,
        pec50_err, slope_err, front_err, back_err,
        null_intercept_np, null_rmse_np, f_stat_np, p_values,
    ])
    df = df.copy()
    df[fit_cols] = pd.DataFrame(fit_data, columns=fit_cols, index=df.index)
    df["Curve Log P_Value"] = -np.log10(df["Curve P_Value"])
    return df


def _add_nan_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add NaN-filled fit columns to an empty DataFrame."""
    fit_cols = [
        "pEC50", "Curve Slope", "Curve Front", "Curve Back",
        "Curve Fold Change", "Curve AUC", "Curve RMSE", "Curve R2",
        "pEC50 Error", "Curve Slope Error", "Curve Front Error", "Curve Back Error",
        "Null Model", "Null RMSE", "Curve F_Value", "Curve P_Value", "Curve Log P_Value",
    ]
    df = df.copy()
    for col in fit_cols:
        df[col] = np.nan
    return df
