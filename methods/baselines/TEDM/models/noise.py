from abc import ABC, abstractmethod
from functools import partial
import copy
import numpy as np
import torch
from utils import stats


class NoiseSchedule(ABC):

    def __init__(self, padding=0, training=True, **kwargs):
        super().__init__()
        self.padding = padding
        self.training = training
        self.stats = None
        self.sliding_window = None
        self.ema_beta = None
        self.var_min, self.var_max = None, None
        self.scale_min, self.scale_max = None, None
        self.sigma_data = 1.0

        for k, v in kwargs.items():
            setattr(self, k, v)

    def to(self, device):
        self.device = device

    @abstractmethod
    def sigma(self, y=None, t=None):
        pass

    @abstractmethod
    def time_discretization(self, num_steps=18):
        pass

    @abstractmethod
    def loss_weight(self):
        pass

    @torch.no_grad()
    def __call__(self, y, t=None):
        sigma = self.sigma(y, t=t)
        weight = self.loss_weight(sigma)
        # Remove padding if applicable
        if self.padding:
            y = y[..., self.padding:]
        return y, sigma, weight

    def eval(self):
        cloned_self = copy.copy(self)
        cloned_self.training = False
        return cloned_self


class iDDPM(NoiseSchedule):

    def __init__(self, padding, **kwargs):
        super().__init__(
            padding, M=10000, C1=0.001, C2=0.008, j0=8,
            beta_d=19.9, beta_min=0.1, epsilon_s=1e-3, epsilon_t=1e-5, **kwargs
        )
        # Define u discretization
        u = torch.zeros(self.M + 1)

        def alpha_bar(j):
            return (
                (0.5 * np.pi * j / self.M / (self.C2 + 1)).sin() ** 2
            )

        for j in torch.arange(self.M, 0, -1):  # M, ..., 1
            ratio = (alpha_bar(j - 1) / alpha_bar(j)).clip(min=self.C1)
            u[j - 1] = (
                ((u[j] ** 2 + 1) / ratio - 1).sqrt()
            )
        self.u = u

    def sigma(self, y=None, t=None):  # VP sigma
        """Variance-preserving schedule"""
        if t is None:
            rnd_u = torch.rand([y.shape[0], 1, 1], device=self.device)
            t = 1 + rnd_u * (self.epsilon_t - 1)
        else:
            t = torch.as_tensor(t)
        sigma = (
            (0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1
        ).sqrt()
        return sigma

    def sigma_inv(self, sigma):  # time from VP sigma
        sigma = torch.as_tensor(sigma)
        discriminant = (
            self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()
        )
        return (discriminant.sqrt() - self.beta_min) / self.beta_d

    def round_sigma(self, sigma, return_index=False):
        self.u = self.u.to(self.device)
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(
            sigma.reshape(1, -1, 1), self.u.reshape(1, -1, 1)
        ).argmin(2)
        result = (
            index if return_index else self.u[index.flatten()].to(sigma.dtype)
        )
        return result.reshape(sigma.shape)

    def scale(self, y):  # VP scale
        return 1 / (self.sigma(y) ** 2 + 1).sqrt()

    def loss_weight(self, sigma):  # VP loss weight
        return 1 / sigma ** 2

    def time_discretization(self, num_steps=18):  # DDIM (not VP) sampler
        self.u = self.u.to(self.device)
        sigma_min = self.sigma(y=torch.as_tensor([0]), t=self.epsilon_s).to(
            self.device
        )
        sigma_max = self.sigma(y=torch.as_tensor([0]), t=1).to(self.device)
        steps = torch.arange(
            num_steps, dtype=torch.float32, device=self.device
        )
        u_filtered = self.u[
            torch.logical_and(self.u >= sigma_min, self.u <= sigma_max)
        ]
        sigma_steps = u_filtered[
            ((len(u_filtered) - 1) / (num_steps - 1) * steps)
            .round()
            .to(torch.int64)
        ]
        return self.sigma_inv(self.round_sigma(sigma_steps))


class EDM(NoiseSchedule):

    def __init__(self, padding, **kwargs):
        super().__init__(
            padding, P_mean=-1.2, P_std=1.2,
            sigma_min=0.01, sigma_max=88, rho=7,
            S_churn=2.5, S_min=0.75, S_max=80, S_noise=1.05, **kwargs
        )

    def sigma(self, y, t=None):
        rnd_normal = torch.randn([y.shape[0], 1, 1], device=self.device)
        return (rnd_normal * self.P_std + self.P_mean).exp()

    def loss_weight(self, sigma):
        return (
            (sigma ** 2 + self.sigma_data ** 2)
            / (sigma * self.sigma_data) ** 2
        )

    def time_discretization(self, num_steps=18):
        steps = torch.arange(
            num_steps, dtype=torch.float32, device=self.device
        )
        sigma_min, sigma_max, rho = self.sigma_min, self.sigma_max, self.rho
        u = steps / (num_steps - 1)  # u in [0, 1]
        sigma_steps = (
            sigma_max ** (1 / rho)
            + u * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        # t_N = 0
        t_steps = torch.cat([
            sigma_steps,
            torch.zeros_like(sigma_steps[:1])
        ])
        return t_steps


class GenCast(EDM):

    def __init__(self, padding, **kwargs):
        super().__init__(padding, **kwargs)

    def sigma(self, y, t=None):
        rnd_u = torch.rand([y.shape[0], 1, 1], device=self.device)
        sigma_min, sigma_max, rho = self.sigma_min, self.sigma_max, self.rho
        return (
            sigma_max ** (1 / rho)
            + rnd_u * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho


class TEDM(NoiseSchedule):

    def __init__(self, padding, **kwargs):
        super().__init__(padding, **kwargs)
        if self.stats == 'cumulative':
            mean_fn = stats.cumulative_mean
            var_fn = stats.cumulative_var
        elif self.stats == 'sliding':
            mean_fn = stats.sliding_mean
            var_fn = stats.sliding_var
        else:
            raise ValueError(f"Unknown stats mode: {self.stats}")
        self._mean = partial(
            mean_fn,
            window_size=self.sliding_window,
            ema_beta=self.ema_beta,
        )
        self._var = partial(
            var_fn,
            window_size=self.sliding_window,
            ema_beta=self.ema_beta,
        )
        self.scale = partial(self.scale, from_data=kwargs['scale_from_data'])

    def var(self, y):
        var = self._var(y)
        var = torch.clamp(var, min=self.var_min, max=self.var_max)
        if self.training:
            return var[..., 1:]  # remove padding
        return var

    def scale(self, y, from_data=False):
        if not from_data:
            return torch.ones_like(y)
        mean = self._mean(y)
        return torch.clamp(
            mean / mean[..., :1],
            min=self.scale_min,
            max=self.scale_max
        )

    def sigma(self, y, t=None):
        s_t = self.scale(y)[..., 1:]
        return (self.var(y) / s_t**2).sqrt()

    def sample(self, eps, sigma):
        return sigma * eps

    def loss_weight(self, sigma):
        return (
            (sigma ** 2 + self.sigma_data ** 2)
            / (sigma * self.sigma_data) ** 2
        )

    def time_discretization(self, num_steps=18):
        raise NotImplementedError("TEDM does not use time_discretization")
