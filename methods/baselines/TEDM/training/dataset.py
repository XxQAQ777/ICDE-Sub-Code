import numpy as np
from torch.utils.data import Dataset
import torch

import pandas as pd
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        path,
        window=96,
        horizon=96,
        past_k=2,
        padding=0,
        period='train',
        train=0.7,
        test=0.2,
    ):
        super().__init__()
        assert period in ['train', 'test', 'val'], \
            'period must be train or test or validation.'
        self.window = window
        self.horizon = horizon
        data = self.read_data(path)
        self.feat_size = data.shape[-1]

        # 1. Split RAW data FIRST (before any transformation)
        train_raw, val_raw, test_raw = self.split_raw(
            data, train=train, test=test
        )

        # 2. Fit scaler on TRAINING data only
        scaler = StandardScaler().fit(train_raw)
        self.scaler = scaler
        train_raw = scaler.transform(train_raw)
        val_raw = scaler.transform(val_raw)
        test_raw = scaler.transform(test_raw)

        # 3. Create windows from each split separately
        enlarge_for_inference = int(period != 'train') * self.horizon
        enlarge_for_cond = past_k + padding
        _window = self.window + enlarge_for_cond + enlarge_for_inference

        train_data = self._create_windows(train_raw, _window)
        val_data = self._create_windows(val_raw, _window)
        test_data = self._create_windows(test_raw, _window)

        if period == 'train':
            self.samples = train_data
        elif period == 'test':
            self.samples = test_data
        elif period == 'val':
            self.samples = val_data
        else:
            raise ValueError(
                "Invalid period. Must be 'train', 'test', or 'val'.")

        self.num_samples = self.samples.shape[0] if self.samples.size > 0 else 0

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv"""
        df = pd.read_csv(filepath, header=0)
        columns = df.columns.str.lower()

        if 'date' in columns:
            date = np.where([c == 'date' for c in columns])[0].item()
            df.drop(df.columns[date], axis=1, inplace=True)

        return df.values

    @staticmethod
    def split_raw(data, train=0.7, test=0.2):
        """Split raw time series BEFORE windowing to prevent leakage."""
        size = data.shape[0]

        train_num = int(np.ceil(size * train))
        test_num = int(np.ceil(size * test))

        train_data = data[:train_num, :]
        val_data = data[train_num:-test_num, :]
        test_data = data[-test_num:, :]

        return train_data, val_data, test_data

    def _create_windows(self, data, window_size):
        """Create sliding windows from a single split."""
        num_seqs = max(data.shape[0] - window_size + 1, 0)
        if num_seqs == 0:
            return np.zeros((0, window_size, self.feat_size))

        x = np.zeros((num_seqs, window_size, self.feat_size))

        for i in range(num_seqs):
            window = slice(i, i + window_size)
            x[i, :, :] = data[window, :]

        return x

    def __getitem__(self, ind):
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.num_samples
