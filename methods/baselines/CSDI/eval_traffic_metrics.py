import argparse
import csv
import json
import math
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from main_model import CSDI_Forecasting
from dataset_traffic import get_dataloader


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--data_path", required=True)
parser.add_argument("--index_path", required=True)
parser.add_argument("--modelfolder", required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--history_length", type=int, default=144)
parser.add_argument("--pred_length", type=int, default=144)
parser.add_argument("--nsample", type=int, default=1)
parser.add_argument("--out_prefix", required=True)
args = parser.parse_args()

with open(Path("config") / args.config, "r") as f:
    config = yaml.safe_load(f)

_, _, test_loader, scaler, mean_scaler, target_dim = get_dataloader(
    data_path=args.data_path,
    index_path=args.index_path,
    device=args.device,
    batch_size=config["train"]["batch_size"],
    history_length=args.history_length,
    pred_length=args.pred_length,
)

model = CSDI_Forecasting(config, args.device, target_dim).to(args.device)
model.load_state_dict(torch.load(Path("save") / args.modelfolder / "model.pth", map_location=args.device))
model.eval()

scaler = scaler.view(1, 1, -1)
mean_scaler = mean_scaler.view(1, 1, -1)

wanted = [36, 72, 144]
stats = {str(h): {"sse": 0.0, "sae": 0.0, "sape": 0.0, "count": 0.0, "mape_count": 0.0} for h in wanted}
stats["Average"] = {"sse": 0.0, "sae": 0.0, "sape": 0.0, "count": 0.0, "mape_count": 0.0}

future_idx = None

def update(name, pred, target, mask):
    pred = pred * scaler + mean_scaler
    target = target * scaler + mean_scaler
    diff = pred - target
    mask = mask > 0

    s = stats[str(name)]
    s["sse"] += ((diff ** 2) * mask).sum().item()
    s["sae"] += (diff.abs() * mask).sum().item()
    s["count"] += mask.sum().item()

    mape_mask = mask & (target.abs() > 1e-5)
    s["sape"] += ((diff.abs() / target.abs().clamp_min(1e-5)) * mape_mask).sum().item()
    s["mape_count"] += mape_mask.sum().item()

with torch.no_grad():
    pbar = tqdm(test_loader)
    for batch_no, batch in enumerate(pbar, start=1):
        samples, target, eval_points, _, _ = model.evaluate(batch, args.nsample)

        # samples: [B, nsample, N, T] -> [B, nsample, T, N]
        samples = samples.permute(0, 1, 3, 2)
        target = target.permute(0, 2, 1)
        eval_points = eval_points.permute(0, 2, 1)

        pred = samples.median(dim=1).values

        if future_idx is None:
            future_idx = torch.where(eval_points.sum(dim=(0, 2)) > 0)[0]
            print("future steps:", len(future_idx), "range:", int(future_idx[0]), int(future_idx[-1]))
            if len(future_idx) < max(wanted):
                raise RuntimeError(f"Only {len(future_idx)} future steps found")

        update("Average", pred, target, eval_points)

        for h in wanted:
            t = int(future_idx[h - 1])
            update(h, pred[:, t:t+1, :], target[:, t:t+1, :], eval_points[:, t:t+1, :])

        avg = stats["Average"]
        mae = avg["sae"] / max(avg["count"], 1.0)
        rmse = math.sqrt(avg["sse"] / max(avg["count"], 1.0))
        pbar.set_postfix(batch_no=batch_no, mae=mae, rmse=rmse)

rows = []
for key in ["36", "72", "144", "Average"]:
    s = stats[key]
    rows.append({
        "horizon": key,
        "MAE": s["sae"] / s["count"],
        "RMSE": math.sqrt(s["sse"] / s["count"]),
        "MAPE": 100.0 * s["sape"] / s["mape_count"],
    })

print("\nTraffic metrics")
for r in rows:
    print(f"{r['horizon']:>7}  MAE={r['MAE']:.6f}  RMSE={r['RMSE']:.6f}  MAPE={r['MAPE']:.6f}")

out_prefix = Path(args.out_prefix)
out_prefix.parent.mkdir(parents=True, exist_ok=True)

with open(str(out_prefix) + ".json", "w") as f:
    json.dump(rows, f, indent=2)

with open(str(out_prefix) + ".csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["horizon", "MAE", "RMSE", "MAPE"])
    writer.writeheader()
    writer.writerows(rows)

print("saved:", str(out_prefix) + ".json")
print("saved:", str(out_prefix) + ".csv")
