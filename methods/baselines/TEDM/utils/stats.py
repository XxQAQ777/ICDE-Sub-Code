import torch
import torch.nn.functional as F


def cumulative_mean(x: torch.Tensor, window_size=None, ema_beta=None):
    """
    Cumulative mean along the last axis.

    Args:
        x (torch.Tensor): shape (B, F, S)

    Returns:
        mean (torch.Tensor): shape (B, F, S)
    """
    n = torch.arange(1, x.size(-1) + 1, device=x.device, dtype=x.dtype)
    n_shape = [1] * (x.ndim - 1) + [-1]
    n = n.view(n_shape)  # Shape (..., N)
    mean = torch.cumsum(x, dim=-1) / n
    return mean


def cumulative_var(
    x: torch.Tensor,
    window_size=None,
    ema_beta=None,
    eps: float = 1e-8,
):
    """
    Cumulative variance along the last axis.

    Args:
        x (torch.Tensor): shape (B, F, S)
        eps (float): small constant for numerical stability

    Returns:
        var (torch.Tensor): shape (B, F, S)
    """
    n = torch.arange(1, x.size(-1) + 1, device=x.device, dtype=x.dtype)
    n_shape = [1] * (x.ndim - 1) + [-1]
    n = n.view(n_shape)  # Shape (..., N)
    mean = torch.cumsum(x, dim=-1) / n
    mean_sq = torch.cumsum(x ** 2, dim=-1) / n
    var = mean_sq - mean ** 2 + eps
    # Apply Bessel's correction
    var = n / torch.clamp(n - 1, min=1) * var
    return var


# --- helper: causal recursive EMA along time (vectorized over B,F) ---
def _ema_recursive(x: torch.Tensor, beta: float) -> torch.Tensor:
    """
    y[t] = beta * y[t-1] + (1-beta) * x[t], with y[0] = x[0]
    beta in (0,1); higher beta => smoother.
    """
    if not (0.0 < beta < 1.0):
        raise ValueError("ema_beta must be in (0,1).")
    B, F, S = x.shape
    y = torch.empty_like(x)
    y[:, :, 0] = x[:, :, 0]
    one_minus = 1.0 - beta
    for t in range(1, S):  # tiny loop over time; fully vectorized over (B,F)
        y[:, :, t] = beta * y[:, :, t - 1] + one_minus * x[:, :, t]
    return y


# --- shared window-bounds utility ---
def _bounds_lengths(S: int, w: int, device):
    t = torch.arange(S, device=device)
    left = (t - (w - 1) // 2).clamp(min=0)
    right = (t + w // 2).clamp(max=S - 1)
    right1 = right + 1
    lengths = (right - left + 1)
    return left, right1, lengths


def sliding_mean(
    x: torch.Tensor,
    window_size: int,
    ema_beta=None,
) -> torch.Tensor:
    """
    Sliding mean with shrinking windows at the edges (no padding).
    Fully vectorized via prefix sums.

    Args:
        x (torch.Tensor): shape (B, F, S)
        window_size (int): window length >= 1

    Returns:
        mean (torch.Tensor): shape (B, F, S)
    """
    B, F_, S = x.shape
    device, dtype = x.device, x.dtype

    # window bounds
    left, right1, lengths = _bounds_lengths(S, window_size, device)

    x_flat = x.reshape(B * F_, S)
    pref = F.pad(torch.cumsum(x_flat, dim=1), (1, 0))  # (BF, S+1)

    BF = B * F_
    l_idx = left.view(1, -1).expand(BF, -1)
    r1_idx = right1.view(1, -1).expand(BF, -1)

    sum_x = torch.gather(pref, 1, r1_idx) - torch.gather(pref, 1, l_idx)
    lengths = lengths.to(dtype=dtype).view(1, -1).expand(BF, -1)

    mean = (sum_x / lengths).reshape(B, F_, S)
    if ema_beta is not None:
        mean = _ema_recursive(mean, ema_beta)
    return mean


def sliding_var(
    x: torch.Tensor,
    window_size: int,
    eps: float = 1e-8,
    ema_beta=None,
) -> torch.Tensor:
    """
    Sliding variance with shrinking windows at the edges (no padding).
    Fully vectorized via prefix sums.

    Args:
        x (torch.Tensor): shape (B, F, S)
        window_size (int): window length >= 1
        eps (float): small constant for numerical stability

    Returns:
        var (torch.Tensor): shape (B, F, S)
    """
    B, F_, S = x.shape
    device, dtype = x.device, x.dtype

    # window bounds
    left, right1, lengths = _bounds_lengths(S, window_size, device)
    BF = B * F_

    x_flat = x.reshape(B * F_, S)
    x2_flat = x_flat ** 2

    pref = F.pad(torch.cumsum(x_flat, dim=1), (1, 0))  # (BF, S+1)
    pref2 = F.pad(torch.cumsum(x2_flat, dim=1), (1, 0))   # (BF, S+1)

    l_idx = left.view(1, -1).expand(BF, -1)
    r1_idx = right1.view(1, -1).expand(BF, -1)

    sum_x = torch.gather(pref, 1, r1_idx) - torch.gather(pref, 1, l_idx)
    sum_x2 = torch.gather(pref2, 1, r1_idx) - torch.gather(pref2, 1, l_idx)

    lengths = lengths.to(dtype=dtype).view(1, -1).expand(BF, -1)

    mean_flat = sum_x / lengths
    ex2_flat = sum_x2 / lengths
    var = torch.clamp(ex2_flat - mean_flat ** 2, min=0.0).reshape(B, F_, S)

    if ema_beta is not None:
        var = _ema_recursive(var, ema_beta)
    return var
