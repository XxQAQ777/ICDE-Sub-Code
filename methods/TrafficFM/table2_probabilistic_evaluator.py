#!/usr/bin/env python3
"""Reproduce the TrafficFM Table-II probabilistic evaluation protocol.

The evaluator draws exactly ten stochastic trajectories for every fixed test
window, inverse-transforms every trajectory to the original speed scale, and
then aggregates metrics.  It is deliberately separate from the older
``eval_probabilistic_144.py`` utility.
"""

import argparse
import csv
import importlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

import util
from engine import trainer


def empirical_crps(samples, target):
    """Pointwise empirical CRPS for samples shaped [B, S, N]."""
    count = samples.shape[1]
    first = (samples - target.unsqueeze(1)).abs().mean(dim=1)
    ordered, _ = samples.sort(dim=1)
    coefficient = 2 * torch.arange(count, device=samples.device, dtype=samples.dtype)
    coefficient = coefficient - count + 1
    pairwise = 2.0 * (ordered * coefficient.view(1, -1, 1)).sum(dim=1)
    pairwise = pairwise / float(count * count)
    return first - 0.5 * pairwise


def valid_mask(target, eps):
    """Table-II validity rule: finite, non-null original-scale observations."""
    return torch.isfinite(target) & (target.abs() > eps)


def make_model(args, device, scaler, supports):
    model_module = importlib.import_module("models.default")
    return trainer(
        scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid,
        args.dropout, args.learning_rate, args.weight_decay, device, supports,
        args.gcn_bool, args.addaptadj, supports[0], model_module=model_module,
        train_objective="model", blocks=args.blocks, layers=args.layers,
    ).model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("METR-LA", "PEMS-BAY"))
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--adjdata", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--horizons", nargs="+", type=int, default=[36, 72, 144])
    parser.add_argument("--seq-length", type=int, default=144)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--in-dim", type=int, default=3)
    parser.add_argument("--nhid", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--valid-eps", type=float, default=1e-5)
    parser.add_argument("--gcn-bool", action="store_true", default=True)
    parser.add_argument("--addaptadj", action="store_true", default=True)
    args = parser.parse_args()

    if args.num_samples != 10:
        raise ValueError("Table-II protocol requires exactly --num-samples 10")
    if any(h < 1 or h > args.seq_length for h in args.horizons):
        raise ValueError("All horizons must be within the prediction length")
    if len(set(args.horizons)) != len(args.horizons):
        raise ValueError("Duplicate horizons are not allowed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Table-II evaluation")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    data = util.load_unified_dataset(args.dataset_dir, args.batch_size,
                                     args.batch_size, args.batch_size)
    _, _, adjacency = util.load_adj(args.adjdata, "doubletransition")
    model_module = importlib.import_module("models.default")
    supports = model_module.make_supports(adjacency, device)
    model = make_model(args, device, data["scaler"], supports)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)

    # Each horizon owns its own pooled W1 buffers.  Pooling order is explicit:
    # trajectory, test window, then node; the target pool is window then node.
    horizon_state = {
        h: {"n": 0, "mae": 0.0, "crps": 0.0, "pred_pool": [], "target_pool": []}
        for h in args.horizons
    }
    loader = data["test_loader"]
    remaining = loader.size

    for batch_no, (x, y) in enumerate(loader.get_iterator(), start=1):
        take = min(len(x), remaining)
        remaining -= take
        history = torch.as_tensor(x[:take], dtype=torch.float32, device=device)
        history = history.permute(0, 3, 2, 1)
        history = F.pad(history, (1, 0, 0, 0))
        target = torch.as_tensor(y[:take, ..., 0], dtype=torch.float32, device=device)
        trajectories = []
        with torch.inference_mode():
            for _ in range(args.num_samples):
                normalized = model(history, fm_steps=args.sampling_steps).squeeze(-1)
                trajectories.append(data["scaler"].inverse_transform(normalized))
        samples = torch.stack(trajectories, dim=1)  # [B, S, T, N], raw scale
        point = samples.mean(dim=1)

        for horizon in args.horizons:
            step = horizon - 1
            sample_h = samples[:, :, step, :]
            target_h = target[:, step, :]
            mask = valid_mask(target_h, args.valid_eps)
            if not mask.any():
                continue
            crps = empirical_crps(sample_h, target_h)
            point_error = (point[:, step, :] - target_h).abs()
            state = horizon_state[horizon]
            state["n"] += int(mask.sum().item())
            state["mae"] += float(point_error[mask].sum().item())
            state["crps"] += float(crps[mask].sum().item())
            # W1 is pooled once per horizon after all test batches are read.
            # Make the documented flattening order explicit: trajectory,
            # test-window, node.  W1 itself is permutation invariant, but the
            # order is recorded so the table-generation artifact is auditable.
            pooled_predictions = sample_h.permute(1, 0, 2)
            pooled_mask = mask.unsqueeze(0).expand_as(pooled_predictions)
            state["pred_pool"].append(pooled_predictions[pooled_mask].detach().cpu().numpy())
            state["target_pool"].append(target_h[mask].detach().cpu().numpy())

        if batch_no == 1 or batch_no % 50 == 0:
            print(f"evaluation batch {batch_no}/{loader.num_batch}", flush=True)
        if remaining <= 0:
            break

    metrics = {}
    for horizon in args.horizons:
        state = horizon_state[horizon]
        if state["n"] == 0:
            raise RuntimeError(f"No valid observations at horizon {horizon}")
        pred_pool = np.concatenate(state["pred_pool"])
        target_pool = np.concatenate(state["target_pool"])
        wd = float(wasserstein_distance(pred_pool, target_pool))
        metrics[str(horizon)] = {
            "MAE": state["mae"] / state["n"],
            "CRPS": state["crps"] / state["n"],
            "WD": wd,
            "valid_values": state["n"],
            "pooled_prediction_values": int(pred_pool.size),
            "pooled_target_values": int(target_pool.size),
        }
    metrics["Average"] = {
        key: float(np.mean([metrics[str(h)][key] for h in args.horizons]))
        for key in ("MAE", "CRPS", "WD")
    }

    payload = {
        "method": "TrafficFM",
        "dataset": args.dataset,
        "protocol": "Table-II fixed STD-MAE 144-to-144 test split",
        "num_trajectories_per_test_condition": args.num_samples,
        "sampling_steps": args.sampling_steps,
        "prediction_point": "arithmetic mean of the ten raw-scale trajectories",
        "inverse_transform": "each trajectory is inverse-transformed before every metric",
        "validity_mask": "finite target and abs(target) > valid_eps on original speed scale",
        "crps": "mean_s(|x_s-y|) - 0.5 * mean_{s,s\'}(|x_s-x_s\'|), pointwise then pooled over valid window-node values",
        "wd": "scipy 1-Wasserstein per horizon after pooling trajectory-window-node predictions against window-node targets; no per-window W1 averaging",
        "aggregation": "horizon metrics average arithmetically for Average",
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "metrics_original_speed_scale": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Horizon", "MAE", "CRPS", "WD", "valid_values"])
        writer.writeheader()
        for horizon, values in metrics.items():
            writer.writerow({"Horizon": horizon, **{k: values.get(k, "") for k in writer.fieldnames[1:]}})
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
