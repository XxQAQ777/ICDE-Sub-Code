import torch
import utils
from omegaconf import OmegaConf
import torch.nn as nn
from . import networks, noise, precond


class Model(nn.Module):

    def __init__(self, config: OmegaConf):
        super().__init__()

        # Instantiate denoiser network and noise schedule
        model = utils.instantiate_from_config(
            config.model.denoiser, from_module=networks
        )
        self.noise_sch = utils.instantiate_from_config(
            config.model.noise_schedule,
            config.dataloader.params.padding,
            from_module=noise
        )
        # Wrap model in preconditioner
        precondition_model = utils.instantiate_from_config(
            config.model.preconditioning,
            model, self.noise_sch,
            from_module=precond
        )
        self.device = torch.device(
            f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu'
        )
        self.base: nn.Module = precondition_model.to(self.device)
        self.noise_sch.to(self.device)

        # Keep config
        self.config = config

    def data_from_window(self, x):
        seq_len = self.config.dataloader.params.window
        past_k = self.config.dataloader.params.past_k
        x_cond = x[..., :seq_len]
        x_cur = x[..., past_k:]
        return x_cond, x_cur

    def _set_visuals(self, y, noisy_y, denoised_y):
        self.seq, self.noisy_seq, self.denoised_seq = y, noisy_y, denoised_y

    def get_visuals(self):
        return self.seq, self.noisy_seq, self.denoised_seq

    def loss(self, x):
        x_cond, y = self.data_from_window(x.transpose(2, 1))
        y, sigma, weight = self.noise_sch(y)
        noise = sigma * torch.randn_like(y)
        denoised_y = self.base(y + noise, x_cond, sigma)
        loss = weight * ((denoised_y - y)**2)
        self._set_visuals(y, y + noise, denoised_y)
        return loss.mean()

    def forward(self, x):
        return self.loss(x)
