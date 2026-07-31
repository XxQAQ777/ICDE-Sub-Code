import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _load_npz_array(path):
    z = np.load(path, allow_pickle=True)
    key = "data" if "data" in z.files else z.files[0]
    data = z[key]
    if data.ndim == 3:
        data = data[..., 0]
    return data.astype(np.float32)


def _pick_index(index_npz, mode):
    keys = {k.lower(): k for k in index_npz.files}
    aliases = {
        "train": ["train", "train_index", "train_indices"],
        "valid": ["valid", "val", "valid_index", "val_index"],
        "test": ["test", "test_index", "test_indices"],
    }
    for name in aliases[mode]:
        if name in keys:
            return np.asarray(index_npz[keys[name]])
    raise KeyError(f"Cannot find {mode} split in index file. keys={index_npz.files}")


class TrafficForecastingDataset(Dataset):
    def __init__(self, data_path, index_path, mode="train", history_length=None, pred_length=None):
        self.raw_data = _load_npz_array(data_path)
        index_npz = np.load(index_path, allow_pickle=True)
        mode = "valid" if mode == "valid" else mode
        self.indices = _pick_index(index_npz, mode).astype(np.int64)

        if self.indices.ndim == 1:
            self.indices = self.indices.reshape(-1, 3)
        if self.indices.shape[1] < 3:
            if history_length is None or pred_length is None:
                raise ValueError("Index has <3 columns; provide history_length and pred_length.")
            starts = self.indices[:, 0]
            self.indices = np.stack([starts, starts + history_length, starts + history_length + pred_length], axis=1)

        if history_length is not None:
            self.indices[:, 1] = self.indices[:, 0] + history_length
        if pred_length is not None:
            self.indices[:, 2] = self.indices[:, 1] + pred_length

        train_indices = _pick_index(index_npz, "train").astype(np.int64)
        if train_indices.ndim == 1:
            train_indices = train_indices.reshape(-1, 3)
        train_start = int(train_indices[:, 0].min())
        train_end = int(train_indices[:, 2].max())

        train_data = self.raw_data[train_start:train_end]
        self.mean_data = train_data.mean(axis=0).astype(np.float32)
        self.std_data = train_data.std(axis=0).astype(np.float32)
        self.std_data[self.std_data < 1e-6] = 1.0

        self.main_data = (self.raw_data - self.mean_data) / self.std_data
        self.mask_data = np.ones_like(self.main_data, dtype=np.float32)

        self.history_length = int(self.indices[0, 1] - self.indices[0, 0])
        self.pred_length = int(self.indices[0, 2] - self.indices[0, 1])
        self.seq_length = self.history_length + self.pred_length
        self.target_dim = self.main_data.shape[1]

    def __getitem__(self, orgindex):
        start, mid, end = self.indices[orgindex, :3]
        observed = self.main_data[start:end]
        observed_mask = self.mask_data[start:end]
        target_mask = observed_mask.copy()
        target_mask[mid - start:] = 0.0

        return {
            "observed_data": observed,
            "observed_mask": observed_mask,
            "gt_mask": target_mask,
            "timepoints": np.arange(end - start, dtype=np.float32),
            "feature_id": np.arange(self.target_dim, dtype=np.float32),
        }

    def __len__(self):
        return len(self.indices)


def get_dataloader(data_path, index_path, device, batch_size=4, history_length=None, pred_length=None):
    train_dataset = TrafficForecastingDataset(data_path, index_path, "train", history_length, pred_length)
    valid_dataset = TrafficForecastingDataset(data_path, index_path, "valid", history_length, pred_length)
    test_dataset = TrafficForecastingDataset(data_path, index_path, "test", history_length, pred_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    scaler = torch.from_numpy(train_dataset.std_data).to(device).float()
    mean_scaler = torch.from_numpy(train_dataset.mean_data).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler, train_dataset.target_dim
