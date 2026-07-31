from scipy.signal.windows import kaiser
import numpy as np
import torch
import torch.nn.functional as F

# This file adapts the 2D filters from the original implementation [1] to 1D filters for time series data.
# The sinc filter is used for anti-aliasing during downsampling and upsampling operations.
# ------------------------------------------
# [1] https://github.com/MDFahimAnjum/AliasFree-Diffusion-Models-PyTorch/blob/main/modules/filtrs.py


def sinc_filter_1d(size=6, beta=14):
    # Create a sinc filter
    sinc_filter_1d = np.sinc(np.linspace(-size / 2, size / 2, size))
    window = kaiser(size, beta)
    sinc_filter_1d = sinc_filter_1d * window
    # Normalize the kernel
    sinc_filter_1d = sinc_filter_1d / np.sum(sinc_filter_1d)
    return torch.tensor(sinc_filter_1d, dtype=torch.float32)


def custom_downsample(x, sinc_filter, factor=2):
    # Apply the Jinc filter before downsampling
    sinc_filter = sinc_filter[None, None, :].to(
        x.device)  # Shape (1, 1, filter_size)
    sinc_filter = sinc_filter.repeat(
        x.size(1), 1, 1)  # Match number of channels
    x = F.conv1d(x, sinc_filter, padding='same', groups=x.size(1))
    x = x[:, :, ::factor]
    return x


def custom_upsample(x, sinc_filter, factor=2):
    # Upsample using zero padding followed by applying the sinc filter
    # Get the original dimensions
    batch_size, channels, window = x.shape

    # Create a new tensor with double the sequence size
    upsampled = torch.zeros(
        batch_size, channels, window * factor, device=x.device, dtype=x.dtype
    )
    # Assign the original values to the correct positions
    upsampled[:, :, ::factor] = x
    x = upsampled
    # Apply the sinc filter (low-pass filter)
    sinc_filter = sinc_filter[None, None, :].to(
        x.device)  # Shape (1, 1, filter_size)
    sinc_filter = sinc_filter.repeat(
        x.size(1), 1, 1)  # Match number of channels
    x = F.conv1d(x, sinc_filter, padding='same', groups=x.size(1))
    return x
