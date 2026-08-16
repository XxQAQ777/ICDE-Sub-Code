#!/usr/bin/env python3
"""Calibrated probabilistic scoring for deterministic 144-to-144 baselines.

For a point-forecast model, a node-specific Gaussian predictive distribution
is calibrated from *validation* residual RMS at horizon 144.  Ten seeded
draws from that distribution give an empirical CRPS on the fixed test split.
The WD column deliberately follows the legacy HimNet/STGCN implementation:
the pooled scipy 1-Wasserstein distance between all point forecasts and all
observations at horizon 144.  It is therefore not pointwise MAE.

This is an explicit calibration protocol, not a claim that a deterministic
upstream model natively produces uncertainty samples.
"""

import argparse
import csv
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
TIMING_PATH = Path(__file__).with_name("benchmark_inference_time.py")
METHODS = ("GWNET", "STUM", "STID", "UniST", "STAEformer", "FreqFlow", "PatchSTG")
HORIZON = 144


def load_timing_module():
    spec = importlib.util.spec_from_file_location("trafficfm_timing_models", TIMING_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Windows(Dataset):
    """Fixed-split windows with model-specific history input and raw h=144 label."""

    def __init__(self, data, index, method, global_mean, global_std, patch_scaler=None):
        self.data = data.astype(np.float32, copy=False)
        self.index = np.asarray(index, dtype=np.int64)
        self.method = method
        self.global_mean, self.global_std = global_mean, global_std
        self.patch_scaler = patch_scaler
        if not np.all(self.index[:, 1] - self.index[:, 0] == 144):
            raise ValueError("Expected the benchmark 144-step history")
        if not np.all(self.index[:, 2] - self.index[:, 1] == 144):
            raise ValueError("Expected the benchmark 144-step horizon")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        start, middle, end = self.index[item]
        raw_label = self.data[middle:end, :, 0] * self.global_std + self.global_mean
        label_144 = torch.from_numpy(raw_label[-1].copy())
        if self.method == "PatchSTG":
            patch_mean, patch_std = self.patch_scaler
            raw_history = self.data[start:middle, :, 0] * self.global_std + self.global_mean
            speed = ((raw_history - patch_mean) / patch_std)[..., None].astype(np.float32)
            # PatchSTG's released adapter intentionally reconstructs calendar
            # marks from absolute offsets (t=0 is day 0); it does *not* use
            # the day-of-week channel in processed_data, whose origin is day
            # 3 for METR-LA.  Mirror that convention to load this checkpoint.
            ticks = np.arange(start, middle, dtype=np.int64)
            marks = np.empty((144, speed.shape[1], 2), dtype=np.int64)
            marks[..., 0] = (ticks % 288)[:, None]
            marks[..., 1] = ((ticks // 288) % 7)[:, None]
            return (torch.from_numpy(speed), torch.from_numpy(marks)), label_144
        if self.method == "UniST":
            window = self.data[start:end].copy()
            window[144:, :, 0] = 0.0
            image = window[:, :, 0][None, :, None, :]
            marks = np.stack([window[:, 0, 2], window[:, 0, 1]], axis=-1)
            marks[:, 0] = np.clip(np.rint(marks[:, 0]), 0, 6)
            marks[:, 1] = np.clip(np.rint(marks[:, 1] * 288), 0, 287)
            return (torch.from_numpy(image), torch.from_numpy(marks.astype(np.int64))), label_144
        if self.method == "FreqFlow":
            return torch.from_numpy(self.data[start:middle, :, 0].copy()), label_144
        return torch.from_numpy(self.data[start:middle].copy()), label_144


def patch_scaler(data, train_index, mean, std):
    # Mirrors PatchSTG's adapter: fit on the temporal extent reached by train.
    raw = data[..., 0] * std + mean
    train_end = int(np.asarray(train_index)[:, 2].max())
    value = raw[:train_end]
    return float(value.mean()), float(value.std())


@torch.inference_mode()
def predict_144(method, model, batch, device, mean, std, patch_stats):
    if method == "PatchSTG":
        speed, marks = (value.to(device, non_blocking=True) for value in batch)
        patch_mean, patch_std = patch_stats
        return model(speed.float(), marks).squeeze(-1)[:, HORIZON - 1] * patch_std + patch_mean
    if method == "UniST":
        image, marks = (value.to(device, non_blocking=True) for value in batch)
        _loss, _mask, prediction, _target, _extra = model(
            [image, marks, image], mask_ratio=.5, mask_strategy="temporal", data=None, mode="forward")
        future = model.unpatchify(prediction)[:, HORIZON:, 0, :] * std + mean
        return future[:, HORIZON - 1]
    if method == "FreqFlow":
        history = batch.to(device, non_blocking=True)
        base, _ = model(history)
        t0 = torch.zeros((history.shape[0], 1), device=device)
        return (base[:, -HORIZON:] + model.flow(base, t0)[:, -HORIZON:])[:, HORIZON - 1] * std + mean
    prediction = model(batch.to(device, non_blocking=True))
    return prediction.squeeze(-1)[:, HORIZON - 1] * std + mean


def empirical_crps(samples, target):
    """Exact ensemble CRPS without allocating an S-by-S tensor."""
    count = samples.shape[1]
    first = (samples - target.unsqueeze(1)).abs().mean(dim=1)
    ordered, _ = samples.sort(dim=1)
    coef = (2 * torch.arange(count, device=samples.device, dtype=samples.dtype) - count + 1)
    pairwise = 2.0 * (ordered * coef.view(1, -1, 1)).sum(dim=1) / float(count * count)
    return first - .5 * pairwise


def collect_validation_sigma(method, model, loader, device, mean, std, patch_stats):
    sum_sq = count = None
    for number, (batch, label) in enumerate(loader, start=1):
        pred = predict_144(method, model, batch, device, mean, std, patch_stats)
        target = label.to(device, non_blocking=True)
        valid = torch.isfinite(target) & (target.abs() > 1e-5)
        squared = (pred - target).square() * valid
        if sum_sq is None:
            sum_sq = squared.sum(dim=0)
            count = valid.sum(dim=0)
        else:
            sum_sq += squared.sum(dim=0)
            count += valid.sum(dim=0)
        if number == 1 or number % 100 == 0:
            print("validation batch {}/{}".format(number, len(loader)), flush=True)
    sigma = torch.sqrt(sum_sq / count.clamp_min(1))
    fallback = sigma[count > 0].mean().clamp_min(.1)
    return torch.where(count > 0, sigma.clamp_min(.1), fallback)


def evaluate(method, model, loader, device, mean, std, patch_stats, sigma, samples, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = {"n": 0, "mae": 0., "crps": 0.}
    pooled_pred, pooled_target = [], []
    for number, (batch, label) in enumerate(loader, start=1):
        point = predict_144(method, model, batch, device, mean, std, patch_stats)
        target = label.to(device, non_blocking=True)
        valid = torch.isfinite(target) & (target.abs() > 1e-5)
        draws = point.unsqueeze(1) + sigma.view(1, 1, -1) * torch.randn(
            (point.shape[0], samples, point.shape[1]), device=device, generator=generator)
        crps = empirical_crps(draws, target)
        totals["n"] += int(valid.sum().item())
        totals["mae"] += float((point - target).abs()[valid].sum().item())
        totals["crps"] += float(crps[valid].sum().item())
        pooled_pred.append(point[valid].detach().cpu().numpy())
        pooled_target.append(target[valid].detach().cpu().numpy())
        if number == 1 or number % 100 == 0:
            print("test batch {}/{}".format(number, len(loader)), flush=True)
    return {
        "MAE": totals["mae"] / totals["n"],
        "CRPS": totals["crps"] / totals["n"],
        "WD": float(wasserstein_distance(np.concatenate(pooled_pred), np.concatenate(pooled_target))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", choices=("METR-LA", "PEMS-BAY"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/trafficfm_benchmark_runs/calibrated_point_probabilistic"))
    args = parser.parse_args()
    if args.samples < 2:
        raise ValueError("At least two samples are required for empirical CRPS")
    if not torch.cuda.is_available():
        raise RuntimeError("Use CUDA_VISIBLE_DEVICES to select one GPU")
    timing = load_timing_module()
    dataset_root = ROOT / "data" / "processed" / "STD-MAE" / args.dataset
    # The timing helper intentionally returns test indices only.  Calibration
    # needs the benchmark's existing validation split as well, so read the
    # complete index mapping here (without constructing a new split).
    with open(dataset_root / "data_in144_out144.pkl", "rb") as handle:
        data = pickle.load(handle)["processed_data"].astype(np.float32)
    with open(dataset_root / "index_in144_out144.pkl", "rb") as handle:
        index = pickle.load(handle)
    with open(dataset_root / "scaler_in144_out144.pkl", "rb") as handle:
        scaler = pickle.load(handle)["args"]
    mean, std = float(scaler["mean"]), float(scaler["std"])
    validation = index.get("val", index.get("valid"))
    if validation is None:
        raise KeyError("The shared index has no validation split")
    patch_stats = patch_scaler(data, index["train"], mean, std) if args.method == "PatchSTG" else None
    loaders = {
        "val": DataLoader(Windows(data, validation, args.method, mean, std, patch_stats), batch_size=args.batch_size, shuffle=False, pin_memory=True),
        "test": DataLoader(Windows(data, index["test"], args.method, mean, std, patch_stats), batch_size=args.batch_size, shuffle=False, pin_memory=True),
    }
    device = torch.device("cuda:0")
    model = timing.build_model(args.method, args.dataset, data, device)
    checkpoint = timing.checkpoint_default(args.method, args.dataset)
    timing.load_state(model, checkpoint, device)
    model.eval()
    sigma = collect_validation_sigma(args.method, model, loaders["val"], device, mean, std, patch_stats)
    metrics = evaluate(args.method, model, loaders["test"], device, mean, std, patch_stats, sigma, args.samples, args.seed)
    payload = {
        "method": args.method, "dataset": args.dataset, "horizon": HORIZON,
        "protocol": "fixed STD-MAE 144-to-144 split; validation-residual calibrated Gaussian predictive ensemble",
        "calibration_split": "fixed validation split only", "samples": args.samples, "seed": args.seed,
        "CRPS": "empirical ensemble CRPS E|X-y|-0.5E|X-X'|", "WD": "pooled scipy Wasserstein-1(point_forecast.flatten(), target.flatten())",
        "checkpoint": str(checkpoint), "mean_validation_sigma": float(sigma.mean().item()), "metrics_original_speed_scale": metrics,
    }
    output = args.output_dir / args.method / args.dataset.replace("-", "_") / "seed_{}".format(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(payload, indent=2))
    with open(output / "metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Horizon", "MAE", "CRPS", "WD"])
        writer.writeheader()
        writer.writerow({"Horizon": HORIZON, **metrics})
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
