#!/usr/bin/env python3
"""Vanilla STUM on the benchmark's fixed METR-LA/PEMS-BAY 144->144 split."""

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

METHOD_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(METHOD_ROOT))
from src.stum.STUM import STUM  # noqa: E402

HORIZONS = (36, 72, 144)


class IndexedWindows(Dataset):
    def __init__(self, data, indices):
        self.data = data.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        assert np.all(self.indices[:, 1] - self.indices[:, 0] == 144)
        assert np.all(self.indices[:, 2] - self.indices[:, 1] == 144)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        start, split, end = self.indices[i]
        return torch.from_numpy(self.data[start:split]), torch.from_numpy(self.data[split:end, :, :1])


def load_data(repo_root, dataset):
    root = repo_root / "data" / "processed" / "STD-MAE" / dataset
    with open(root / "data_in144_out144.pkl", "rb") as handle:
        data = pickle.load(handle)["processed_data"]
    with open(root / "index_in144_out144.pkl", "rb") as handle:
        index = pickle.load(handle)
    with open(root / "scaler_in144_out144.pkl", "rb") as handle:
        scaler = pickle.load(handle)["args"]
    return data, index, float(scaler["mean"]), float(scaler["std"])


def make_sums():
    return {key: {"n": 0, "sae": 0.0, "sse": 0.0, "ape": 0.0} for key in ["36", "72", "144", "Average"]}


@torch.no_grad()
def evaluate(model, loader, device, mean, std):
    model.eval()
    sums, mae_sum, mae_n = make_sums(), 0.0, 0
    for x, y in loader:
        pred = model(x.to(device, non_blocking=True)).squeeze(-1) * std + mean
        label = y.to(device, non_blocking=True).squeeze(-1) * std + mean
        valid = torch.isfinite(label) & (label.abs() > 1e-5)
        diff = pred - label
        mae_sum += diff.abs()[valid].sum().item()
        mae_n += valid.sum().item()
        for key, t in [("36", 35), ("72", 71), ("144", 143), ("Average", slice(None))]:
            mask, current_diff, current_label = valid[:, t], diff[:, t], label[:, t]
            sums[key]["n"] += mask.sum().item()
            sums[key]["sae"] += current_diff.abs()[mask].sum().item()
            sums[key]["sse"] += current_diff.square()[mask].sum().item()
            sums[key]["ape"] += (current_diff.abs()[mask] / current_label.abs()[mask]).sum().item()
    result = {}
    for key, value in sums.items():
        mse = value["sse"] / value["n"]
        result[key] = {"MAE": value["sae"] / value["n"], "RMSE": float(np.sqrt(mse)), "MAPE": 100 * value["ape"] / value["n"]}
    return mae_sum / mae_n, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("METR-LA", "PEMS-BAY"), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=998244353)
    parser.add_argument("--num-mlrfs", type=int, default=2)
    parser.add_argument("--num-cells", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/trafficfm_benchmark_runs/STUM"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the formal STUM run")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    repo_root = METHOD_ROOT.parents[2]
    data, index, mean, std = load_data(repo_root, args.dataset)
    validation_index = index.get("val", index.get("valid"))
    if validation_index is None:
        raise KeyError("The fixed index file must contain a val or valid split")
    loaders = {
        "train": DataLoader(IndexedWindows(data, index["train"]), batch_size=args.batch_size,
                            shuffle=True, pin_memory=True, num_workers=0),
        "val": DataLoader(IndexedWindows(data, validation_index), batch_size=args.batch_size,
                          shuffle=False, pin_memory=True, num_workers=0),
        "test": DataLoader(IndexedWindows(data, index["test"]), batch_size=args.batch_size,
                           shuffle=False, pin_memory=True, num_workers=0),
    }
    # Match the official vanilla path: STUM has no external backbone.
    model_args = argparse.Namespace(
        device=str(device), node_num=data.shape[1], input_dim=data.shape[2], output_dim=1,
        seq_length=144, horizon=144, num_mlrfs=args.num_mlrfs, num_cells=args.num_cells,
        embed_dim=args.embed_dim, mlp=False, without_backbone=True, supports=[],
        lrate=args.lr, wdecay=args.weight_decay,
    )
    model = STUM(backbone=None, args=model_args).to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    run_dir = args.output_dir / args.dataset.replace("-", "_") / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best, wait, best_state, history = float("inf"), 0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in loaders["train"]:
            prediction = model(x.to(device, non_blocking=True))
            target = y.to(device, non_blocking=True)
            loss = (prediction - target).abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())
        val_mae, _ = evaluate(model, loaders["val"], device, mean, std)
        scheduler.step()
        record = {"epoch": epoch, "train_normalized_mae": float(np.mean(train_losses)), "val_original_mae": val_mae}
        history.append(record)
        print(json.dumps(record), flush=True)
        if val_mae < best:
            best, wait = val_mae, 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, run_dir / "best_model.pt")
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    _, metrics = evaluate(model, loaders["test"], device, mean, std)
    payload = {
        "method": "STUM", "dataset": args.dataset, "seed": args.seed,
        "input_length": 144, "prediction_length": 144,
        "scaler": {"mean": mean, "std": std}, "best_val_original_mae": best,
        "metrics_original_scale": metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    with open(run_dir / "metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Horizon", "MAE", "RMSE", "MAPE"])
        writer.writeheader()
        for horizon, values in metrics.items():
            writer.writerow({"Horizon": horizon, **values})
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
