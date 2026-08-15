#!/usr/bin/env python3
"""Fixed-split 144->144 TSFlow runner for METR-LA and PEMS-BAY."""

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
from tsflow.model import TSFlowCond  # noqa: E402
from tsflow.utils.variables import Setting  # noqa: E402


class IndexedWindows(Dataset):
    def __init__(self, values, indices, univariate=False):
        self.values = values.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.univariate = univariate
        self.num_nodes = self.values.shape[1]
        assert np.all(self.indices[:, 1] - self.indices[:, 0] == 144)
        assert np.all(self.indices[:, 2] - self.indices[:, 1] == 144)

    def __len__(self):
        return len(self.indices) * self.num_nodes if self.univariate else len(self.indices)

    def __getitem__(self, index):
        if self.univariate:
            window_index, node_index = divmod(index, self.num_nodes)
            start, split, end = self.indices[window_index]
            return (
                torch.from_numpy(self.values[start:split, node_index]),
                torch.from_numpy(self.values[split:end, node_index]),
            )
        start, split, end = self.indices[index]
        return torch.from_numpy(self.values[start:split]), torch.from_numpy(self.values[split:end])


def load_data(repo_root, dataset):
    root = repo_root / "data" / "processed" / "STD-MAE" / dataset
    with open(root / "data_in144_out144.pkl", "rb") as handle:
        standardized = pickle.load(handle)["processed_data"][..., 0]
    with open(root / "index_in144_out144.pkl", "rb") as handle:
        index = pickle.load(handle)
    with open(root / "scaler_in144_out144.pkl", "rb") as handle:
        scaler = pickle.load(handle)["args"]
    # TSFlow's LongScaler is designed for positive raw-valued series.
    values = standardized * float(scaler["std"]) + float(scaler["mean"])
    return values, index


def fill_missing(values, observed, initial=None):
    """Forward-fill zero/missing traffic values without touching the targets used for metrics."""
    values = torch.nan_to_num(values, nan=0.0)
    denominator = observed.sum(dim=1, keepdim=True).clamp_min(1.0)
    fallback = (values * observed).sum(dim=1, keepdim=True) / denominator
    fallback = fallback.clamp_min(1e-3)
    filled = torch.where(observed.bool(), values, fallback.expand_as(values)).clone()
    if initial is not None:
        filled[:, 0] = torch.where(observed[:, 0].bool(), values[:, 0], initial)
    for time_index in range(1, values.shape[1]):
        filled[:, time_index] = torch.where(
            observed[:, time_index].bool(), values[:, time_index], filled[:, time_index - 1]
        )
    return filled


def make_batch(past, future, device, zero_as_missing=True):
    past, future = past.to(device, non_blocking=True), future.to(device, non_blocking=True)
    if zero_as_missing:
        past_observed = torch.isfinite(past) & (past.abs() > 1e-5)
        future_observed = torch.isfinite(future) & (future.abs() > 1e-5)
    else:
        past_observed = torch.isfinite(past)
        future_observed = torch.isfinite(future)
    past_observed = past_observed.to(dtype=past.dtype)
    future_observed = future_observed.to(dtype=future.dtype)
    filled_past = fill_missing(past, past_observed)
    filled_future = fill_missing(future, future_observed, initial=filled_past[:, -1])
    return {
        "past_target": filled_past,
        "future_target": filled_future,
        "past_observed_values": past_observed,
        "future_observed_values": future_observed,
        "mean": ((past * past_observed).sum(dim=1, keepdim=True) /
                 past_observed.sum(dim=1, keepdim=True).clamp_min(1.0)).clamp_min(1e-3),
        "metric_future_target": future,
    }


def make_sums():
    return {
        key: {"n": 0, "sae": 0.0, "sse": 0.0, "ape": 0.0, "crps": 0.0, "wd": 0.0}
        for key in ("36", "72", "144", "Average")
    }


@torch.no_grad()
def forecast_samples(model, batch, num_samples):
    """Return independent TSFlow trajectories with shape [B, samples, H]."""
    forecasts = []
    model.num_samples = 1
    for _ in range(num_samples):
        forecasts.append(model(batch["past_target"], batch["past_observed_values"], batch["mean"])[:, 0])
    return torch.stack(forecasts, dim=1)


@torch.no_grad()
def evaluate(model, loader, device, point_samples, progress=False, max_batches=0, zero_as_missing=True,
             probabilistic_metrics=False):
    model.eval()
    sums, mae_sum, mae_n = make_sums(), 0.0, 0
    for batch_number, (past, future) in enumerate(loader, start=1):
        if max_batches > 0 and batch_number > max_batches:
            break
        batch = make_batch(past, future, device, zero_as_missing=zero_as_missing)
        samples = forecast_samples(model, batch, point_samples)
        prediction, label = samples.mean(dim=1), batch["metric_future_target"]
        valid = torch.isfinite(label) & (label.abs() > 1e-5)
        diff = prediction - label
        mae_sum += diff.abs()[valid].sum().item()
        mae_n += valid.sum().item()
        for key, step in (("36", 35), ("72", 71), ("144", 143), ("Average", slice(None))):
            mask = valid[:, step]
            current_diff, current_label = diff[:, step], label[:, step]
            sums[key]["n"] += mask.sum().item()
            sums[key]["sae"] += current_diff.abs()[mask].sum().item()
            sums[key]["sse"] += current_diff.square()[mask].sum().item()
            sums[key]["ape"] += (current_diff.abs()[mask] / current_label.abs()[mask]).sum().item()
            if probabilistic_metrics:
                if point_samples < 2:
                    raise ValueError("CRPS requires at least two independent TSFlow samples")
                sample_slice = samples[:, :, step]
                wd = (sample_slice - current_label.unsqueeze(1)).abs().mean(dim=1)
                pairwise = (sample_slice.unsqueeze(2) - sample_slice.unsqueeze(1)).abs().mean(dim=(1, 2))
                crps = wd - 0.5 * pairwise
                sums[key]["wd"] += wd[mask].sum().item()
                sums[key]["crps"] += crps[mask].sum().item()
        if progress and (batch_number == 1 or batch_number % 100 == 0):
            print("evaluation batch {} / {}".format(batch_number, len(loader)), flush=True)
    metrics = {}
    for key, item in sums.items():
        mse = item["sse"] / item["n"]
        metrics[key] = {"MAE": item["sae"] / item["n"], "RMSE": float(np.sqrt(mse)), "MAPE": 100.0 * item["ape"] / item["n"]}
        if probabilistic_metrics:
            metrics[key]["CRPS"] = item["crps"] / item["n"]
            metrics[key]["WD"] = item["wd"] / item["n"]
    return mae_sum / mae_n, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("METR-LA", "PEMS-BAY"), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=6432)
    parser.add_argument("--train-batches-per-epoch", type=int, default=512)
    parser.add_argument(
        "--max-validation-batches", type=int, default=0,
        help="cap validation batches during training; 0 evaluates the entire fixed validation split",
    )
    parser.add_argument("--sampling-steps", type=int, default=16)
    parser.add_argument("--test-point-samples", type=int, default=5)
    parser.add_argument("--test-probability-samples", type=int, default=20,
                        help="independent TSFlow trajectories used for final CRPS/WD")
    parser.add_argument(
        "--setting", choices=("univariate", "multivariate"), default="univariate",
        help="univariate is the official TSFlow traffic setup: each sensor is one series",
    )
    parser.add_argument("--use-ema", action="store_true", help="use and update TSFlow EMA as in its official trainer")
    parser.add_argument("--zero-as-observed", action="store_true", help="legacy behavior; normally traffic zero is treated as missing")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/trafficfm_benchmark_runs/TSFlow"))
    parser.add_argument("--metrics-output-dir", type=Path, default=None, help="optional separate output directory for evaluation metrics")
    parser.add_argument("--evaluate-only", action="store_true", help="evaluate saved best_model.pt without training")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the formal TSFlow run")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    repo_root = METHOD_ROOT.parents[2]
    values, indices = load_data(repo_root, args.dataset)
    validation = indices.get("val", indices.get("valid"))
    if validation is None:
        raise KeyError("The benchmark index lacks val/valid")
    loaders = {
        "train": DataLoader(IndexedWindows(values, indices["train"], univariate=args.setting == "univariate"), batch_size=args.batch_size,
                              shuffle=True, pin_memory=True, num_workers=0),
        "val": DataLoader(IndexedWindows(values, validation, univariate=args.setting == "univariate"), batch_size=args.batch_size,
                            shuffle=False, pin_memory=True, num_workers=0),
        "test": DataLoader(IndexedWindows(values, indices["test"], univariate=args.setting == "univariate"), batch_size=args.batch_size,
                             shuffle=False, pin_memory=True, num_workers=0),
    }
    # context_freqs=1 and use_lags=False are required by the prescribed 144-step
    # input protocol: no observations prior to the fixed history window are read.
    model = TSFlowCond(
        setting=Setting.UNIVARIATE if args.setting == "univariate" else Setting.MULTIVARIATE,
        target_dim=1 if args.setting == "univariate" else values.shape[1],
        context_length=144,
        prediction_length=144,
        backbone_params={
            "input_dim": 1, "output_dim": 1, "step_emb": 64,
            "num_residual_blocks": 3, "residual_block": "s4", "hidden_dim": 64,
            "dropout": 0.0, "init_skip": False, "feature_skip": True,
        },
        prior_params={"kernel": "ou", "gamma": 1.0, "context_freqs": 1},
        optimizer_params={"lr": args.lr},
        ema_params={"beta": 0.9999, "update_after_step": 128, "update_every": 1},
        frequency="5min_144",
        normalization="longmean",
        use_lags=False,
        use_ema=args.use_ema,
        num_steps=args.sampling_steps,
        solver="euler",
        matching="random",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70], gamma=0.5)
    run_dir = args.output_dir / args.dataset.replace("-", "_") / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only:
        checkpoint = run_dir / "best_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError("No saved checkpoint for evaluation: {}".format(checkpoint))
        metrics_dir = run_dir if args.metrics_output_dir is None else args.metrics_output_dir / args.dataset.replace("-", "_") / f"seed_{args.seed}"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        _, metrics = evaluate(model, loaders["test"], device, point_samples=args.test_probability_samples, progress=True,
                              zero_as_missing=not args.zero_as_observed, probabilistic_metrics=True)
        payload = {
            "method": "TSFlow", "dataset": args.dataset, "seed": args.seed,
            "input_length": 144, "prediction_length": 144,
            "context_freqs": 1, "use_lags": False, "sampling_steps": args.sampling_steps,
            "zero_as_missing": not args.zero_as_observed,
            "test_point_forecast": "mean of {} flow samples".format(args.test_probability_samples),
            "CRPS": "empirical E|X-y|-0.5E|X-X'|", "WD": "empirical W1 to Dirac target, E|X-y|",
            "checkpoint": str(checkpoint), "metrics_original_scale": metrics,
        }
        (metrics_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
        with open(metrics_dir / "metrics.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Horizon", "MAE", "RMSE", "MAPE", "CRPS", "WD"])
            writer.writeheader()
            for horizon, values_ in metrics.items():
                writer.writerow({"Horizon": horizon, **values_})
        print(json.dumps(payload, indent=2), flush=True)
        return
    best, wait, best_state, history = float("inf"), 0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_index, (past, future) in enumerate(loaders["train"], start=1):
            batch = make_batch(past, future, device, zero_as_missing=not args.zero_as_observed)
            x1, x0, _, _, _, features = model._extract_features(batch)
            t = torch.rand((x1.shape[0], 1), device=device)
            loss_mask = torch.cat([batch["past_observed_values"], batch["future_observed_values"]], dim=1)
            loss = model.p_losses(x1, x0, t, features, loss_mask=loss_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            # The upstream PyTorch-Lightning trainer invokes this hook after
            # every optimizer step.  Our compact runner must do it explicitly.
            model.on_train_batch_end()
            losses.append(loss.item())
            if batch_index >= args.train_batches_per_epoch:
                break
        val_mae, _ = evaluate(
            model, loaders["val"], device, point_samples=1,
            max_batches=args.max_validation_batches,
            zero_as_missing=not args.zero_as_observed,
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_flow_mse": float(np.mean(losses)),
            "val_original_mae": val_mae,
            "validation_batches": args.max_validation_batches or "all",
        }
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
    _, metrics = evaluate(model, loaders["test"], device, point_samples=args.test_probability_samples,
                          zero_as_missing=not args.zero_as_observed, progress=True, probabilistic_metrics=True)
    payload = {
        "method": "TSFlow", "dataset": args.dataset, "seed": args.seed,
        "input_length": 144, "prediction_length": 144,
        "context_freqs": 1, "use_lags": False, "sampling_steps": args.sampling_steps,
        "zero_as_missing": not args.zero_as_observed,
        "test_point_forecast": "mean of {} flow samples".format(args.test_probability_samples),
        "CRPS": "empirical E|X-y|-0.5E|X-X'|", "WD": "empirical W1 to Dirac target, E|X-y|",
        "best_val_original_mae": best, "metrics_original_scale": metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    with open(run_dir / "metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Horizon", "MAE", "RMSE", "MAPE", "CRPS", "WD"])
        writer.writeheader()
        for horizon, values_ in metrics.items():
            writer.writerow({"Horizon": horizon, **values_})
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
