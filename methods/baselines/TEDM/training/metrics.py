import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# ------------------------------
# MSE and MAE
# -------------------------------
def mse_and_mae(reals, samples):
    if samples.ndim > 3:
        samples = samples.mean(axis=-1)
    mse = mean_squared_error(reals.reshape(-1), samples.reshape(-1))
    mae = mean_absolute_error(reals.reshape(-1), samples.reshape(-1))
    return mse, mae

# -----------------------------
# CRPS from samples (multivariate)
# -----------------------------


def crps(y_true, samples):
    """
    Multivariate, sample-based CRPS estimator.

    Parameters
    ----------
    y_true : array-like, shape (B, N, F)
        Observed values.
    samples : array-like, shape (B, N, F, S)  (S along draw_axis)
        Predictive samples per timestep and feature.

    Returns
    -------
    crps : float or ndarray shape (N, F)
    """
    S = samples.shape[-1]
    y_true = y_true[..., None]

    # E|X - y|
    term1 = np.mean(np.abs(samples - y_true), axis=-1)  # (N, F)

    # 0.5 * E|X - X'| via sorting trick along draw axis
    X_sorted = np.sort(samples, axis=-1)
    i = np.arange(1, S + 1, dtype=samples.dtype)
    weights = (2 * i - (S + 1)) / (S ** 2)   # shape (S,)
    term2 = np.sum(X_sorted * weights, axis=-1)  # (N, F)

    crps = term1 - term2
    return np.mean(crps)

# -----------------------------
# QICE (Quantile Interval Calibration Error)
# -----------------------------
def qice(y_true, samples):
    """
    Multivariate Quantile Interval Calibration Error (QICE) matching the paper.

    Definition (per a chosen set of internal quantile levels tau_1<...<tau_K):
      - Build M=K+1 intervals using the *predicted* boundaries Q_{τ_k}(x_i,f):
          (-inf, Q_{tau_1}], (Q_{tau_1}, Q_{tau_2}], ..., (Q_{tau_K}, +inf)
      - Let r_m be the empirical fraction of items whose y falls in interval m.
      - Expected mass for interval m is p_m = tau_m - tau_{m-1}
        with tau_0=0, tau_M=1.
      - QICE = mean_m | r_m - p_m |. (Smaller is better; 0 = perfectly calibrated.)

    Parameters
    ----------
    y_true : ndarray, shape (B, N, F)
        Observed values for B windows, N timesteps, and F features.
    samples : ndarray, shape (B, N, F, S)
        Predictive samples per timestep and feature.


    Returns
    -------
    qice : float
        Scalar QICE averaged over all bins.
    """
    # Standard decile setup: 9 internal cut points -> 10 equal-mass bins.
    qs = np.arange(0.1, 1.0, 0.1)
    q_pred = np.quantile(samples, qs, axis=-1)
    q_pred = np.moveaxis(q_pred, 0, -1)

    Q = np.asarray(q_pred)

    B, N, F, K = Q.shape
    M = K + 1

    # Expected mass per interval from the tau grid: p_m = τ_m - τ_{m-1}
    tau_bounds = np.concatenate(([0.0], qs, [1.0]))  # length M+1
    p = np.diff(tau_bounds)  # (M,)

    # Build per-item interval boundaries using *predicted* quantiles
    # Intervals are [lower, upper) except the last is [last, +inf) — no double counting.
    lower = np.concatenate([np.full((B, N, F, 1), -np.inf), Q], axis=-1)  # (B,N,F,M)
    upper = np.concatenate([Q, np.full((B, N, F, 1),  np.inf)], axis=-1)  # (B,N,F,M)

    inside = (y_true[..., None] > lower) & (y_true[..., None] <= upper)  # (B,N,F,M), exactly one True per (B,N,F)

    r = inside.mean(axis=(0, 1, 2))            # (M,)
    qice = np.mean(np.abs(r - p))           # scalar
    return qice
