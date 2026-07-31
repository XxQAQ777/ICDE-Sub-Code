import argparse
import datetime
import json
import os
import yaml
import torch

from main_model import CSDI_Forecasting
from dataset_traffic import get_dataloader
from utils import train, evaluate


parser = argparse.ArgumentParser(description="CSDI traffic forecasting")
parser.add_argument("--config", type=str, default="traffic144_smoke.yaml")
parser.add_argument("--datatype", type=str, default="metrla144")
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--index_path", type=str, required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=20)
parser.add_argument("--history_length", type=int, default=144)
parser.add_argument("--pred_length", type=int, default=144)
args = parser.parse_args()

torch.manual_seed(args.seed)

with open(os.path.join("config", args.config), "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
print(args)
print(json.dumps(config, indent=4))

train_loader, valid_loader, test_loader, scaler, mean_scaler, target_dim = get_dataloader(
    data_path=args.data_path,
    index_path=args.index_path,
    device=args.device,
    batch_size=config["train"]["batch_size"],
    history_length=args.history_length,
    pred_length=args.pred_length,
)

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = f"./save/forecasting_{args.datatype}_{current_time}/"
print("target_dim:", target_dim)
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)
with open(os.path.join(foldername, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

model = CSDI_Forecasting(config, args.device, target_dim).to(args.device)

if args.modelfolder == "":
    train(model, config["train"], train_loader, valid_loader=valid_loader, foldername=foldername)
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth", map_location=args.device))

model.target_dim = target_dim
evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
)
