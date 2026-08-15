import numpy as np
from torch.utils.data import Dataset, DataLoader

from .utils import log_string


class TrafficIndexedDataset(Dataset):
    """Dataset using the benchmark's explicit [x_start, y_start, y_end] index."""

    def __init__(self, data_path, index_path, history_length, pred_length, tod, dow, split):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")

        self.history_length = history_length
        self.pred_length = pred_length
        self.tod = tod
        self.dow = dow

        with np.load(data_path) as data_file:
            raw = data_file["data"][..., :1].astype(np.float32)

        with np.load(index_path) as index_file:
            if not {"train", "val", "test"}.issubset(index_file.files):
                raise ValueError(f"{index_path} must contain train/val/test")
            train_index = index_file["train"].astype(np.int64)
            self.indices = index_file[split].astype(np.int64)

        if self.indices.ndim != 2 or self.indices.shape[1] != 3:
            raise ValueError(f"Invalid index shape: {self.indices.shape}")
        if np.any(self.indices[:, 1] - self.indices[:, 0] != history_length):
            raise ValueError("Index history length does not match input_len")
        if np.any(self.indices[:, 2] - self.indices[:, 1] != pred_length):
            raise ValueError("Index prediction length does not match output_len")
        if self.indices[:, 0].min() < 0 or self.indices[:, 2].max() > raw.shape[0]:
            raise ValueError("Index range exceeds traffic data length")

        # Fit normalization only on the temporal extent reached by train samples.
        train_end = int(train_index[:, 2].max())
        train_raw = raw[:train_end]
        self.mean = float(train_raw.mean())
        self.std = float(train_raw.std())
        if self.std <= 0:
            raise ValueError("Training data has zero standard deviation")

        self.data_x = (raw - self.mean) / self.std
        self.data_y = raw
        self.num_nodes = raw.shape[1]

    def _timestamps(self, start, end):
        ticks = np.arange(start, end, dtype=np.int64)
        stamps = np.empty((len(ticks), self.num_nodes, 2), dtype=np.int64)
        stamps[..., 0] = (ticks % self.tod)[:, None]
        stamps[..., 1] = ((ticks // self.tod) % self.dow)[:, None]
        return stamps

    def __getitem__(self, item):
        x_start, y_start, y_end = self.indices[item]
        return (
            self.data_x[x_start:y_start],
            self.data_y[y_start:y_end],
            self._timestamps(x_start, y_start),
            self._timestamps(y_start, y_end),
        )

    def __len__(self):
        return len(self.indices)

    def inverse_transform(self, values):
        return values * self.std + self.mean


def data_provider(
    num_workers,
    batch_size,
    data_path,
    index_path,
    history_length,
    pred_length,
    tod,
    dow,
    split,
    log,
):
    dataset = TrafficIndexedDataset(
        data_path=data_path,
        index_path=index_path,
        history_length=history_length,
        pred_length=pred_length,
        tod=tod,
        dow=dow,
        split=split,
    )
    log_string(log, f"{split}: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        drop_last=False,
    )
    return dataset, loader