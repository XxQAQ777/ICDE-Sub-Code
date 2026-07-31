import torch
import torch.nn as nn
from .noise import NoiseSchedule


class Preconditioner(nn.Module):

    def __init__(self, model: nn.Module, noise_sch: NoiseSchedule, **kwargs):
        super().__init__()
        self.model = model
        self.noise_sch = noise_sch

    def input_transform(self, *args, **kwargs):
        raise NotImplementedError

    def output_transform(self, output, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args):
        # Apply input transformation (should be differentiable)
        new_args = self.input_transform(*args)
        # Forward pass through the wrapped model
        output = self.model(*new_args)
        # Apply output transformation (should be differentiable)
        output = self.output_transform(output, *args)
        return output


class iDDPM(Preconditioner):

    def input_transform(self, noisy_y, x_cond, sigma):
        sigma = sigma.reshape(-1, 1, 1)
        c_in = 1 / (1 + sigma ** 2).sqrt()  # same as noise scale at that sigma
        c_noise = self.noise_sch.M - 1 - \
            self.noise_sch.round_sigma(sigma, return_index=True)
        return c_in * noisy_y, x_cond, c_noise.flatten()

    def output_transform(self, output, noisy_y, x_cond, sigma):
        c_skip = 1
        c_out = -sigma
        return c_skip * noisy_y + c_out * output


class EDM(Preconditioner):

    def input_transform(self, noisy_y, x_cond, sigma):
        sigma = sigma.reshape(-1, 1, 1)
        sigma_data = self.noise_sch.sigma_data
        c_in = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        return c_in * noisy_y, x_cond, c_noise.flatten()

    def output_transform(self, output, noisy_y, x_cond, sigma):
        sigma_data = self.noise_sch.sigma_data
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
        return c_skip * noisy_y + c_out * output


class TEDM(EDM):

    def input_transform(self, noisy_y, x_cond, sigma):
        sigma_data = self.noise_sch.sigma_data
        c_in = (
            self.noise_sch.scale(noisy_y).abs()
            / (sigma_data ** 2 + sigma ** 2).sqrt()
        )
        return c_in * noisy_y, x_cond, sigma
