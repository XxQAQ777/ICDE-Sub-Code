#!/usr/bin/env python3
"""STID on the benchmark's fixed STD-MAE 144->144 traffic windows."""

import argparse
import csv
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import STID


class IndexedWindows(Dataset):
    def __init__(self, data, index):
        self.data = data.astype(np.float32, copy=False)
        self.index = np.asarray(index, dtype=np.int64)
        if not np.all(self.index[:, 1] - self.index[:, 0] == 144):
            raise ValueError("Expected the shared 144-step history index")
        if not np.all(self.index[:, 2] - self.index[:, 1] == 144):
            raise ValueError("Expected the shared 144-step forecast index")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        start, split, end = self.index[item]
        return torch.from_numpy(self.data[start:split]), torch.from_numpy(self.data[split:end, :, 0])


def load_data(repo_root, dataset):
    root = repo_root / "data" / "processed" / "STD-MAE" / dataset
    with open(root / "data_in144_out144.pkl", "rb") as handle:
        data = pickle.load(handle)["processed_data"]
    with open(root / "index_in144_out144.pkl", "rb") as handle:
        index = pickle.load(handle)
    with open(root / "scaler_in144_out144.pkl", "rb") as handle:
        scaler = pickle.load(handle)["args"]
    return data, index, float(scaler["mean"]), float(scaler["std"])


def init_sums():
    return {name: {"n": 0, "sae": 0.0} for name in ("36", "72", "144", "Average")}


@torch.no_grad()
def evaluate(model, loader, device, mean, std):
    model.eval()
    sums = init_sums()
    for history, future in loader:
        prediction = model(history.to(device, non_blocking=True)).squeeze(-1) * std + mean
        label = future.to(device, non_blocking=True) * std + mean
        valid = torch.isfinite(label) & (label.abs() > 1e-5)
        error = (prediction - label).abs()
        for name, step in (("36", 35), ("72", 71), ("144", 143), ("Average", slice(None))):
            mask = valid[:, step]
            sums[name]["n"] += mask.sum().item()
            sums[name]["sae"] += error[:, step][mask].sum().item()
    # Deterministic delta forecast: CRPS and pointwise W1 are exactly MAE.
    return {
        name: {"MAE": item["sae"] / item["n"], "CRPS": item["sae"] / item["n"], "WD": item["sae"] / item["n"]}
        for name, item in sums.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("METR-LA", "PEMS-BAY"), required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/trafficfm_benchmark_runs/STID"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")

    repo_root = Path(__file__).resolve().parents[3]
    data, index, mean, std = load_data(repo_root, args.dataset)
    validation = index.get("val", index.get("valid"))
    if validation is None:
        raise KeyError("Shared index has no val/valid split")
    loaders = {
        "train": DataLoader(IndexedWindows(data, index["train"]), batch_size=args.batch_size, shuffle=True, pin_memory=True),
        "val": DataLoader(IndexedWindows(data, validation), batch_size=args.batch_size, shuffle=False, pin_memory=True),
        "test": DataLoader(IndexedWindows(data, index["test"]), batch_size=args.batch_size, shuffle=False, pin_memory=True),
    }
    model = STID(
        num_nodes=data.shape[1], input_len=144, output_len=144, input_dim=3,
        embed_dim=32, node_dim=32, temp_dim_tid=32, temp_dim_diw=32, num_layer=3, dropout=0.15,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_dir = args.output_dir / args.dataset.replace("-", "_") / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_model.pt"
    best, stale = float("inf"), 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for history_x, future_y in loaders["train"]:
            prediction = model(history_x.to(device, non_blocking=True)).squeeze(-1)
            loss = (prediction - future_y.to(device, non_blocking=True)).abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())
        val = evaluate(model, loaders["val"], device, mean, std)["Average"]["MAE"]
        record = {"epoch": epoch, "train_normalized_mae": float(np.mean(losses)), "val_original_mae": val}
        history.append(record)
        print(json.dumps(record), flush=True)
        if val < best:
            best, stale = val, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    metrics = evaluate(model, loaders["test"], device, mean, std)
    payload = {
        "method": "STID", "dataset": args.dataset, "seed": args.seed,
        "input_length": 144, "prediction_length": 144,
        "best_val_original_mae": best,
        "forecast_type": "deterministic_delta",
        "wd_definition": "mean pointwise W1(delta_prediction, delta_observation)",
        "metrics_original_scale": metrics,
    }
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    with (run_dir / "mae_crps_wd.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Horizon", "MAE", "CRPS", "WD"])
        writer.writeheader()
        for horizon, item in metrics.items():
            writer.writerow({"Horizon": horizon, **item})
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
