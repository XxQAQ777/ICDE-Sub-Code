#!/usr/bin/env python3
"""Causal 144->144 UniST runner on the benchmark's fixed graph-data split.

The official UniST implementation accepts a generic H x W spatial tensor.  For
graph traffic data, each sensor remains one token in the canonical 1 x N node
axis; no artificial 2-D geographical grid or new data split is created.
"""

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
sys.path.insert(0, str(METHOD_ROOT / "src"))
from model import UniST_model  # noqa: E402


class IndexedGraphWindows(Dataset):
    """Lazily expose fixed 144-history + 144-future windows as UniST tensors."""

    def __init__(self, data, indices):
        self.data = data.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        if not np.all(self.indices[:, 1] - self.indices[:, 0] == 144):
            raise ValueError("UniST history must be the benchmark's 144 steps")
        if not np.all(self.indices[:, 2] - self.indices[:, 1] == 144):
            raise ValueError("UniST horizon must be the benchmark's 144 steps")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start, _, end = self.indices[index]
        window = self.data[start:end]  # [288, nodes, (speed, tod, dow)]
        # [channel=1, time=288, height=1, width=nodes]
        values = torch.from_numpy(window[:, :, 0][None, :, None, :])
        # UniST TemporalEmbedding expects [weekday, time-of-day] in this order.
        marks = np.stack([window[:, 0, 2], window[:, 0, 1]], axis=-1)
        marks[:, 0] = np.clip(np.rint(marks[:, 0]), 0, 6)
        # STD-MAE stores five-minute time-of-day as [0, 1), while UniST's
        # embedding expects its integer 0..287 index.
        marks[:, 1] = np.clip(np.rint(marks[:, 1] * 288), 0, 287)
        marks = torch.from_numpy(marks.astype(np.int64, copy=False))
        # Prompt tuning is disabled for this scratch, fixed-split run, but the
        # official forward signature still requires a period tensor.
        return values, marks, values.clone()


def load_benchmark_data(repo_root, dataset):
    root = repo_root / "data" / "processed" / "STD-MAE" / dataset
    with open(root / "data_in144_out144.pkl", "rb") as handle:
        data = pickle.load(handle)["processed_data"]
    with open(root / "index_in144_out144.pkl", "rb") as handle:
        indices = pickle.load(handle)
    with open(root / "scaler_in144_out144.pkl", "rb") as handle:
        scaler = pickle.load(handle)["args"]
    return data, indices, float(scaler["mean"]), float(scaler["std"])


def init_sums():
    return {key: {"n": 0, "sae": 0.0, "sse": 0.0, "ape": 0.0} for key in ("36", "72", "144", "Average")}


@torch.no_grad()
def evaluate(model, loader, device, mean, std, dataset):
    model.eval()
    sums, mae_sum, mae_count = init_sums(), 0.0, 0
    for image, mark, period in loader:
        image = image.to(device, non_blocking=True)
        mark = mark.to(device, non_blocking=True)
        period = period.to(device, non_blocking=True)
        _, _, pred, target, _ = model(
            [image, mark, period], mask_ratio=0.5, mask_strategy="temporal",
            data=dataset, mode="forward",
        )
        # unpatchify gives [B, 288, H=1, W=num_nodes]; take future only.
        pred = model.unpatchify(pred)[:, 144:, 0, :] * std + mean
        label = model.unpatchify(target)[:, 144:, 0, :] * std + mean
        valid = torch.isfinite(label) & (label.abs() > 1e-5)
        diff = pred - label
        mae_sum += diff.abs()[valid].sum().item()
        mae_count += valid.sum().item()
        for key, step in (("36", 35), ("72", 71), ("144", 143), ("Average", slice(None))):
            mask = valid[:, step]
            current_diff, current_label = diff[:, step], label[:, step]
            sums[key]["n"] += mask.sum().item()
            sums[key]["sae"] += current_diff.abs()[mask].sum().item()
            sums[key]["sse"] += current_diff.square()[mask].sum().item()
            sums[key]["ape"] += (current_diff.abs()[mask] / current_label.abs()[mask]).sum().item()
    metrics = {}
    for key, item in sums.items():
        mse = item["sse"] / item["n"]
        metrics[key] = {"MAE": item["sae"] / item["n"], "RMSE": float(np.sqrt(mse)), "MAPE": 100.0 * item["ape"] / item["n"]}
    return mae_sum / mae_count, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("METR-LA", "PEMS-BAY"), required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--size", choices=("1", "2", "3", "4", "5"), default="1")
    parser.add_argument("--t-patch-size", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/trafficfm_benchmark_runs/UniST"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this formal UniST run")
    if 144 % args.t_patch_size:
        raise ValueError("t-patch-size must divide both 144-step history and horizon")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    repo_root = METHOD_ROOT.parents[2]
    data, indices, mean, std = load_benchmark_data(repo_root, args.dataset)
    validation = indices.get("val", indices.get("valid"))
    if validation is None:
        raise KeyError("The benchmark index lacks val/valid")
    loaders = {
        "train": DataLoader(IndexedGraphWindows(data, indices["train"]), batch_size=args.batch_size,
                              shuffle=True, pin_memory=True, num_workers=0),
        "val": DataLoader(IndexedGraphWindows(data, validation), batch_size=args.batch_size,
                            shuffle=False, pin_memory=True, num_workers=0),
        "test": DataLoader(IndexedGraphWindows(data, indices["test"]), batch_size=args.batch_size,
                             shuffle=False, pin_memory=True, num_workers=0),
    }
    # Only original UniST architecture arguments; prompt tuning needs an external
    # universal pretraining checkpoint and is not enabled for a fair scratch run.
    model_args = argparse.Namespace(
        dataset=args.dataset, size=args.size, patch_size=1, t_patch_size=args.t_patch_size,
        pos_emb="SinCos", no_qkv_bias=0, prompt_ST=0, his_len=144, pred_len=144,
        time_of_day_size=288, day_of_week_size=7,
    )
    model = UniST_model(model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    run_dir = args.output_dir / args.dataset.replace("-", "_") / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best, wait, best_state, history = float("inf"), 0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for image, mark, period in loaders["train"]:
            loss, _, _, _, _ = model(
                [image.to(device, non_blocking=True), mark.to(device, non_blocking=True),
                 period.to(device, non_blocking=True)],
                mask_ratio=0.5, mask_strategy="temporal", data=args.dataset, mode="backward",
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        val_mae, _ = evaluate(model, loaders["val"], device, mean, std, args.dataset)
        scheduler.step()
        record = {"epoch": epoch, "train_masked_mse": float(np.mean(losses)), "val_original_mae": val_mae}
        history.append(record)
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
    _, metrics = evaluate(model, loaders["test"], device, mean, std, args.dataset)
    payload = {
        "method": "UniST", "dataset": args.dataset, "seed": args.seed,
        "input_length": 144, "prediction_length": 144,
        "graph_tensor_layout": "[B, 1, 288, 1, num_nodes]",
        "patch_size": 1, "time_patch_size": args.t_patch_size,
        "prompt_tuning": False, "best_val_original_mae": best,
        "scaler": {"mean": mean, "std": std}, "metrics_original_scale": metrics,
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
