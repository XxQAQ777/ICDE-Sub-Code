import os
import sys
import time
import argparse
import csv
import random
import numpy as np
import torch
import torch.optim as optim

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAFFICFM_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../TrafficFM"))
sys.path.append(TRAFFICFM_DIR)

import util
from model import STID


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_stats(pred, real, stats):
    mask = real != 0.0
    if mask.sum().item() == 0:
        return
    diff = pred - real
    abs_err = torch.abs(diff)[mask]
    sq_err = (diff ** 2)[mask]
    ape = (torch.abs(diff) / torch.clamp(torch.abs(real), min=1e-5))[mask]

    stats["ae"] += abs_err.sum().item()
    stats["se"] += sq_err.sum().item()
    stats["ape"] += ape.sum().item()
    stats["count"] += mask.sum().item()


def finish_stats(stats):
    count = max(stats["count"], 1)
    mae = stats["ae"] / count
    rmse = (stats["se"] / count) ** 0.5
    mape = stats["ape"] / count
    return mae, rmse, mape


def evaluate(model, loader, scaler, device, num_samples, horizons=(36, 72, 144)):
    model.eval()

    avg_stats = {"ae": 0.0, "se": 0.0, "ape": 0.0, "count": 0}
    horizon_stats = {
        h: {"ae": 0.0, "se": 0.0, "ape": 0.0, "count": 0}
        for h in horizons
    }

    seen = 0

    with torch.no_grad():
        for x, y in loader.get_iterator():
            remain = num_samples - seen
            if remain <= 0:
                break

            bsz = min(x.shape[0], remain)
            x = x[:bsz]
            y = y[:bsz]
            seen += bsz

            x = torch.tensor(x, dtype=torch.float32, device=device)
            real = torch.tensor(y[..., 0], dtype=torch.float32, device=device)

            pred_norm = model(x).squeeze(-1)
            pred = scaler.inverse_transform(pred_norm)

            masked_stats(pred, real, avg_stats)

            for h in horizons:
                masked_stats(pred[:, h - 1, :], real[:, h - 1, :], horizon_stats[h])

    avg_result = finish_stats(avg_stats)
    horizon_results = {h: finish_stats(horizon_stats[h]) for h in horizons}
    return avg_result, horizon_results


def measure_speed(model, loader, scaler, device, num_samples):
    model.eval()
    seen = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for x, y in loader.get_iterator():
            remain = num_samples - seen
            if remain <= 0:
                break

            bsz = min(x.shape[0], remain)
            x = x[:bsz]
            seen += bsz

            x = torch.tensor(x, dtype=torch.float32, device=device)
            pred_norm = model(x).squeeze(-1)
            _ = scaler.inverse_transform(pred_norm)

    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.time() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/TrafficFM/METR-LA-144-3feat")
    parser.add_argument("--num_nodes", type=int, default=207)
    parser.add_argument("--input_len", type=int, default=144)
    parser.add_argument("--output_len", type=int, default=144)
    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--node_dim", type=int, default=32)
    parser.add_argument("--temp_dim_tid", type=int, default=32)
    parser.add_argument("--temp_dim_diw", type=int, default=32)
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--no_day_in_week", action="store_true", help="disable day-of-week embedding")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save", type=str, default="methods/baselines/STID/checkpoints/stid_metrla.pt")
    parser.add_argument("--metrics_output", type=str, default=None,
                        help="CSV path for original-scale deterministic MAE/CRPS/WD")
    parser.add_argument("--horizons", type=str, default="36,72,144", help="comma-separated 1-indexed report horizons")
    args = parser.parse_args()
    horizons = tuple(int(item) for item in args.horizons.split(","))
    if not horizons or min(horizons) < 1 or max(horizons) > args.output_len:
        raise ValueError("--horizons must be within --output_len")

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Loading data:", args.data)

    dataloader = util.load_dataset(
        args.data,
        args.batch_size,
        args.batch_size,
        args.test_batch_size,
    )
    scaler = dataloader["scaler"]

    model = STID(
        num_nodes=args.num_nodes,
        input_len=args.input_len,
        output_len=args.output_len,
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        node_dim=args.node_dim,
        temp_dim_tid=args.temp_dim_tid,
        temp_dim_diw=args.temp_dim_diw,
        num_layer=args.num_layer,
        dropout=args.dropout,
        if_day_in_week=(not args.no_day_in_week and args.input_dim >= 3),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_mae = float("inf")
    bad_epochs = 0

    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        dataloader["train_loader"].shuffle()

        train_losses = []
        t1 = time.time()

        for it, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            x = torch.tensor(x, dtype=torch.float32, device=device)
            real = torch.tensor(y[..., 0], dtype=torch.float32, device=device)

            pred_norm = model(x).squeeze(-1)
            pred = scaler.inverse_transform(pred_norm)

            loss = util.masked_mae(pred, real, 0.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_losses.append(loss.item())

            if it % args.print_every == 0:
                print(f"Epoch {epoch:03d} Iter {it:04d} Train MAE {loss.item():.4f}", flush=True)

        train_time = time.time() - t1

        val_avg, _ = evaluate(
            model,
            dataloader["val_loader"],
            scaler,
            device,
            num_samples=dataloader["y_val"].shape[0],
            horizons=horizons,
        )
        val_mae, val_rmse, val_mape = val_avg

        print(
            f"Epoch {epoch:03d} | "
            f"Train MAE {np.mean(train_losses):.4f} | "
            f"Val MAE {val_mae:.4f} | "
            f"Val RMSE {val_rmse:.4f} | "
            f"Val MAPE {val_mape * 100:.2f}% | "
            f"Time {train_time:.2f}s",
            flush=True,
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_val_mae": best_val_mae,
                },
                args.save,
            )
            print("Saved best checkpoint:", args.save)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    print("\nLoading best checkpoint for test:", args.save)
    ckpt = torch.load(args.save, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_avg, test_horizons = evaluate(
        model,
        dataloader["test_loader"],
        scaler,
        device,
        num_samples=dataloader["y_test"].shape[0],
        horizons=horizons,
    )

    print("\n========== STID Test Results ==========")
    for h in horizons:
        mae, rmse, mape = test_horizons[h]
        print(f"Horizon {h:03d}: MAE {mae:.4f}, RMSE {rmse:.4f}, MAPE {mape * 100:.2f}%")

    mae, rmse, mape = test_avg
    print(f"Average    : MAE {mae:.4f}, RMSE {rmse:.4f}, MAPE {mape * 100:.2f}%")

    # STID is a deterministic point forecaster.  Under the standard
    # degenerate-distribution convention CRPS(delta_p, y) and pointwise
    # W1(delta_p, delta_y) both reduce exactly to absolute error / MAE.
    metrics_output = args.metrics_output
    if metrics_output is None:
        metrics_output = os.path.splitext(args.save)[0] + "_mae_crps_wd.csv"
    os.makedirs(os.path.dirname(metrics_output) or ".", exist_ok=True)
    with open(metrics_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Horizon", "MAE", "CRPS", "WD"])
        writer.writeheader()
        for h in horizons:
            h_mae, _, _ = test_horizons[h]
            writer.writerow({"Horizon": h, "MAE": h_mae, "CRPS": h_mae, "WD": h_mae})
        writer.writerow({"Horizon": "Average", "MAE": mae, "CRPS": mae, "WD": mae})
    print("Saved deterministic MAE/CRPS/WD metrics:", metrics_output, flush=True)

    speed = measure_speed(
        model,
        dataloader["test_loader"],
        scaler,
        device,
        num_samples=dataloader["y_test"].shape[0],
    )
    print("\n========== STID Speed ==========")
    print(f"Total inference time on complete test set, batch_size={args.test_batch_size}: {speed:.2f}s")


if __name__ == "__main__":
    main()
