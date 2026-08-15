#!/usr/bin/env python3
"""Documented in-house FreqFlow reconstruction on the fixed 144->144 split."""

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
from models.Flow_Spect import Model as FlowSpect  # noqa: E402


class Windows(Dataset):
    def __init__(self, data, index):
        self.data = data.astype(np.float32, copy=False)
        self.index = np.asarray(index, dtype=np.int64)
        assert np.all(self.index[:, 1] - self.index[:, 0] == 144)
        assert np.all(self.index[:, 2] - self.index[:, 1] == 144)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        start, split, end = self.index[i]
        return torch.from_numpy(self.data[start:split]), torch.from_numpy(self.data[split:end])


def load_data(repo_root, dataset):
    root = repo_root / "data" / "processed" / "STD-MAE" / dataset
    with open(root / "data_in144_out144.pkl", "rb") as handle:
        values = pickle.load(handle)["processed_data"][..., 0]
    with open(root / "index_in144_out144.pkl", "rb") as handle:
        index = pickle.load(handle)
    with open(root / "scaler_in144_out144.pkl", "rb") as handle:
        scaler = pickle.load(handle)["args"]
    return values, index, float(scaler["mean"]), float(scaler["std"])


def make_sums():
    return {name: {"n": 0, "sae": 0.0, "sse": 0.0, "ape": 0.0} for name in ("36", "72", "144", "Average")}


def metrics(prediction, label, sums):
    valid = torch.isfinite(label) & (label.abs() > 1e-5)
    diff = prediction - label
    for name, step in (("36", 35), ("72", 71), ("144", 143), ("Average", slice(None))):
        mask, d, y = valid[:, step], diff[:, step], label[:, step]
        sums[name]["n"] += mask.sum().item()
        sums[name]["sae"] += d.abs()[mask].sum().item()
        sums[name]["sse"] += d.square()[mask].sum().item()
        sums[name]["ape"] += (d.abs()[mask] / y.abs()[mask]).sum().item()


def summarise(sums):
    output = {}
    for name, value in sums.items():
        mse = value["sse"] / value["n"]
        output[name] = {
            "MAE": value["sae"] / value["n"],
            "RMSE": float(np.sqrt(mse)),
            "MAPE": 100.0 * value["ape"] / value["n"],
        }
    return output


def forecast(model, history):
    """Released spectral base trajectory plus the documented t=0 flow correction."""
    base, _ = model(history)
    t0 = torch.zeros((history.shape[0], 1), device=history.device)
    velocity = model.flow(base, t0)
    return base[:, -144:] + velocity[:, -144:], base


@torch.no_grad()
def evaluate(model, loader, device, mean, std):
    model.eval()
    sums = make_sums()
    for history, future in loader:
        prediction, _ = forecast(model, history.to(device, non_blocking=True))
        prediction = prediction * std + mean
        label = future.to(device, non_blocking=True) * std + mean
        metrics(prediction, label, sums)
    return summarise(sums)


def save_metrics(run_dir, payload):
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    with open(run_dir / "metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Horizon", "MAE", "RMSE", "MAPE"])
        writer.writeheader()
        for horizon, values in payload["metrics_original_scale"].items():
            writer.writerow({"Horizon": horizon, **values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("METR-LA", "PEMS-BAY"), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--flow-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/trafficfm_benchmark_runs/FreqFlow_reconstruction"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this run")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    repo_root = METHOD_ROOT.parents[2]
    values, index, mean, std = load_data(repo_root, args.dataset)
    validation = index.get("val", index.get("valid"))
    if validation is None:
        raise KeyError("The fixed index must include val/valid")
    loaders = {
        "train": DataLoader(Windows(values, index["train"]), batch_size=args.batch_size, shuffle=True, pin_memory=True),
        "val": DataLoader(Windows(values, validation), batch_size=args.batch_size, shuffle=False, pin_memory=True),
        "test": DataLoader(Windows(values, index["test"]), batch_size=args.batch_size, shuffle=False, pin_memory=True),
    }
    config = argparse.Namespace(
        seq_len=144, pred_len=144, individual=False, enc_in=values.shape[1],
        cut_freq=72, flow_time_dim=64, flow_hidden_multiplier=1.0,
        chan_attn_dim=64, chan_attn_heads=4, dropout=0.0,
    )
    model = FlowSpect(config).to(device)
    # The released flow head is lazily constructed; materialize it before the
    # optimizer so all of its parameters are trained.
    model.flow(torch.zeros((1, 288, values.shape[1]), device=device), torch.zeros((1, 1), device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70], gamma=0.5)
    run_dir = args.output_dir / args.dataset.replace("-", "_") / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best, wait, best_state, training_history = float("inf"), 0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for input_history, future in loaders["train"]:
            input_history = input_history.to(device, non_blocking=True)
            future = future.to(device, non_blocking=True)
            target = torch.cat([input_history, future], dim=1)
            base, _ = model(input_history)
            # Conditional linear flow from released base trajectory to target.
            t = torch.rand((input_history.shape[0], 1), device=device)
            xt = (1.0 - t[:, None]) * base + t[:, None] * target
            velocity_target = target - base
            velocity_prediction = model.flow(xt, t)
            refined_future = base[:, -144:] + model.flow(base, torch.zeros_like(t))[:, -144:]
            reconstruction = (refined_future - future).abs().mean()
            flow_loss = torch.nn.functional.mse_loss(velocity_prediction, velocity_target)
            loss = reconstruction + args.flow_weight * flow_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())
        val_metrics = evaluate(model, loaders["val"], device, mean, std)
        val_mae = val_metrics["Average"]["MAE"]
        scheduler.step()
        record = {"epoch": epoch, "train_reconstruction_plus_flow": float(np.mean(losses)), "val_original_mae": val_mae}
        training_history.append(record)
        print(json.dumps(record), flush=True)
        if val_mae < best:
            best, wait = val_mae, 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(best_state, run_dir / "best_model.pt")
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    payload = {
        "method": "FreqFlow in-house reconstruction", "official_reproduction": False,
        "dataset": args.dataset, "seed": args.seed, "input_length": 144, "prediction_length": 144,
        "flow_path": "linear path from released spectral base to observed trajectory",
        "loss": "forecast MAE + {} * flow velocity MSE".format(args.flow_weight),
        "best_val_original_mae": best,
        "metrics_original_scale": evaluate(model, loaders["test"], device, mean, std),
    }
    save_metrics(run_dir, payload)
    (run_dir / "history.json").write_text(json.dumps(training_history, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
