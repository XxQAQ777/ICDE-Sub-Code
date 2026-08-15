#!/usr/bin/env python3
"""Materialize the existing PEMS08 fixed 12->12 index for STID's npz loader."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--integer-tod", action="store_true", help="store time-of-day as indices 0..287")
    args = parser.parse_args()
    data = np.load(args.source / "data.npz")["data"].astype(np.float32)
    index = np.load(args.source / "index.npz")
    args.output.mkdir(parents=True, exist_ok=True)
    steps = np.arange(12, dtype=np.int64)[None, :]
    for split in ("train", "val", "test"):
        target = args.output / (split + ".npz")
        if target.exists():
            continue
        windows = index[split]
        if not np.all(windows[:, 1] - windows[:, 0] == 12) or not np.all(windows[:, 2] - windows[:, 1] == 12):
            raise ValueError("PEMS08 source must provide fixed 12->12 windows")
        x = data[windows[:, 0, None] + steps]
        y = data[windows[:, 1, None] + steps, :, :1]
        if args.integer_tod:
            x[..., 1] = np.rint(x[..., 1] * 288).clip(0, 287)
        np.savez_compressed(target, x=x, y=y)
        print(split, x.shape, y.shape, flush=True)


if __name__ == "__main__":
    main()
