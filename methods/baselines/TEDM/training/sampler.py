from abc import ABC, abstractmethod
import numpy as np
import torch


class Sampler(ABC):

    def __init__(
        self,
        model,
        noise_sch,
        horizon=96,
        seq_len=96,
        past_k=2,
        padding=1,
        backward=True,
        deterministic=True,
        clamp=True,
        n_tracks=1,
        batch_size=32,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.noise_sch = noise_sch
        self.horizon = horizon
        self.seq_len = seq_len
        self.past_k = past_k
        self.padding = padding
        self.clamp = clamp
        self.n_tracks = n_tracks
        self.backward = backward
        self.deterministic = deterministic
        self.batch_size = batch_size

    @abstractmethod
    def sample(self, x, x_cond=None, *args):
        pass

    @torch.no_grad()
    def rollout(
        self,
        x,
        prepend_mode='last',
        trace_hook=None,
        max_steps=None,
    ):
        if prepend_mode not in ('last', 'first'):
            raise ValueError(
                f"Unknown prepend_mode='{prepend_mode}'. Use 'last' or 'first'."
            )
        x = x.transpose(2, 1)
        x_cond = x[..., :-(self.past_k + self.padding)]
        x_cur = x[..., self.past_k:]
        # If probabilistic preds with SDE
        if isinstance(self, TEDM) and not self.deterministic:
            x_cur = x_cur.repeat(self.n_tracks, 1, 1)
            x_cond = x_cond.repeat(self.n_tracks, 1, 1)

        rollout_steps = self.horizon
        if max_steps is not None:
            rollout_steps = max(0, min(self.horizon, int(max_steps)))

        if trace_hook is not None:
            init_window = x_cur[..., self.padding:] if isinstance(self, TEDM) else x_cur
            trace_hook(
                phase='init',
                step=0,
                x_window=init_window,
                x_cur=x_cur,
                x_cond=x_cond,
                x_next=None,
            )

        shift = self.past_k + 1  # step offset between cond and cur windows
        for step in range(rollout_steps):
            x_next, _ = self.sample(x_cur, x_cond)
            if self.clamp and isinstance(self, TEDM):
                x_min = x_cur.amin(dim=-1, keepdim=True)
                x_max = x_cur.amax(dim=-1, keepdim=True)
                x_next = torch.clamp(x_next, min=x_min, max=x_max)
            # Stride the condition subsequence and update x_cur
            x_cond = torch.cat(
                [x_cond[..., 1:], x_cur[..., -shift].unsqueeze(-1)], dim=-1
            )
            if isinstance(self, TEDM):
                prefix = x_cur[..., -1:] if prepend_mode == 'last' else x_cur[..., :1]
                x_cur = torch.cat([prefix, x_next], dim=-1)
            else:
                overlap = 0.5 * (
                    x_cur[..., self.padding + 1:] + x_next[..., :-1]
                )
                x_cur = torch.cat([overlap, x_next[..., -1:]], dim=-1)
            if trace_hook is not None:
                trace_hook(
                    phase='step',
                    step=step + 1,
                    x_window=x_next,
                    x_cur=x_cur,
                    x_cond=x_cond,
                    x_next=x_next,
                )

        x_cur = x_cur[..., 1:] if self.padding else x_cur
        if isinstance(self, TEDM) and not self.deterministic:
            x_cur = x_cur.reshape(-1, *x_cur.shape[-2:], self.n_tracks)
        return x_cur.transpose(2, 1)


class DDIM(Sampler):

    def __init__(self, model, noise_sch, **kwargs):
        super().__init__(model, noise_sch, **kwargs)

    @torch.no_grad()
    def sample(self, x_cur, x_cond, *args):
        latents = torch.randn_like(x_cond)
        t_steps = self.noise_sch.time_discretization()

        x_next = latents * t_steps[0]
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            x_cur = x_next
            # Euler step.
            denoised = self.model(x_cur, x_cond, t_cur)
            d_cur = (x_cur - denoised) / t_cur
            x_next = x_cur + (t_next - t_cur) * d_cur

        return x_next, None


class EDM(Sampler):

    def __init__(self, model, noise_sch, **kwargs):
        super().__init__(model, noise_sch, **kwargs)

    @torch.no_grad()
    def sample(self, x_cur, x_cond, *args):
        latents = torch.randn_like(x_cond)
        t_steps = self.noise_sch.time_discretization()
        num_steps = len(t_steps) - 1

        S_churn, S_min, S_max, S_noise = (
            self.noise_sch.S_churn, self.noise_sch.S_min,
            self.noise_sch.S_max, self.noise_sch.S_noise
        )
        x_next = latents * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            noise_delta = (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise
            noise_step = noise_delta * torch.randn_like(x_cur)
            x_hat = x_cur + noise_step

            # Euler step.
            denoised = self.model(x_next, x_cond, t_hat)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.model(x_next, x_cond, t_next)
                d_prime = (x_next - denoised) / t_next
                correction = 0.5 * d_cur + 0.5 * d_prime
                x_next = x_hat + (t_next - t_hat) * correction

        return x_next, None


class TEDM(Sampler):

    def __init__(self, model, noise_sch, **kwargs):
        super().__init__(model, noise_sch, **kwargs)
        self._eps = 1e-12

    def _robust_var_ratio(self, cum_var, tol=1e-3):
        high_variance = torch.logical_and(
            cum_var[..., 1:] > tol, cum_var[..., 0:-1] > tol
        )
        ratio = cum_var[..., 1:] / cum_var[..., 0:-1]
        return torch.where(high_variance, ratio, 1.0)

    def _log_scale_deltas(self, scale):
        ratio = (
            scale[..., 1:] / torch.clamp(scale[..., :-1], min=self._eps)
        ).abs().log()
        if self.padding == 0:
            return ratio
        return ratio[..., self.padding - 1:]

    @torch.no_grad()
    def sample(self, x_cur, x_cond, *args):
        scale = self.noise_sch.scale(x_cur)
        # Diagonal case: Σ_t is represented through its diagonal variance,
        # so Σ_t = var_t / s_t².
        cum_var = self.noise_sch.var(x_cur) / scale ** 2
        x_t = x_cur[..., self.padding:]
        s_t = scale[..., self.padding:]
        cum_std = cum_var[..., self.padding:].sqrt()
        # var_ratio = var_t / var_{t-1}.
        var_ratio = self._robust_var_ratio(cum_var)
        # Δ log √Σ_t = log(√(var_t / var_{t-1}))
        #           = 1/2 log(var_t / var_{t-1}).
        delta_log_var = var_ratio.sqrt().log()
        if self.padding:
            delta_log_var = delta_log_var[..., self.padding - 1:]
        # Δ log s_t = log(s_t / s_{t-1}).
        delta_log_scale = self._log_scale_deltas(scale)
        # In the diagonal case, (ΔΣ_t)^(1/2) is represented by √|Δvar_t|.
        delta_var = (cum_var[..., 1:] - cum_var[..., :-1]).abs().sqrt()
        if self.padding:
            delta_var = delta_var[..., self.padding - 1:]
        sign_t_axis = 1 if self.backward else -1
        Dx = self.model(x_t / s_t, x_cond, cum_std)
        # D(x_t / s_t; Σ_t) - x_t / s_t.
        score_residual = Dx - x_t / s_t
        # Δ log √Σ_t · [D(x_t / s_t; Σ_t) - x_t / s_t] / s_t.
        sigma_deriv_x_score = delta_log_var * score_residual / s_t
        # With Euler-Maruyama over one rollout step Δt,
        # Δx_t^noise = G_t √Δt ε = s_t (Σ̇_t Δt)^(1/2) ε.
        # Since ΔΣ_t ≈ Σ̇_t Δt, this becomes
        # Δx_t^noise ≈ s_t (ΔΣ_t)^(1/2) ε.
        dx_t_noise = s_t.abs() * delta_var * torch.randn_like(x_t)
        # s_t² Δ log √Σ_t · [D(x_t / s_t; Σ_t) - x_t / s_t] / s_t
        # = s_t Δ log √Σ_t · [D(x_t / s_t; Σ_t) - x_t / s_t].
        common_term = (s_t ** 2) * sigma_deriv_x_score
        if self.deterministic:
            # Δx_t =
            #   sign · s_t Δ log √Σ_t · [D(x_t / s_t; Σ_t) - x_t / s_t]
            #   - sign · Δ log s_t · x_t.
            dx_t = sign_t_axis * common_term
            dx_t += -sign_t_axis * delta_log_scale * x_t
        else:
            # Δx_t =
            #   s_t Δ log √Σ_t · [D(x_t / s_t; Σ_t) - x_t / s_t]
            #   + sign · Δ log √Σ_t · [D(x_t / s_t; Σ_t) - x_t / s_t] / s_t
            #   + |s_t| · √|ΔΣ_t| · ε.
            dx_t = common_term + sign_t_axis * sigma_deriv_x_score + dx_t_noise
        x_next = x_t + dx_t
        return x_next, None
