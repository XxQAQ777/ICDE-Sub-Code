# TEDM — Detailed Usage Guide

This document covers the full feature set of the TEDM codebase. For a quick overview, see [README.md](README.md).

---

## Table of Contents

1. [Repository structure](#repository-structure)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Reproducing paper results](#reproducing-paper-results)
5. [Training on a new dataset](#training-on-a-new-dataset)
6. [Visualization and logging](#visualization-and-logging)
7. [Sampling from a trained model](#sampling-from-a-trained-model)
8. [Generating plots](#generating-plots)
9. [Configuration reference](#configuration-reference)

---

## Repository structure

```
tedm/
├── main.py                   # Entry point: train a model and evaluate on test set
├── sample.py                 # Sample forecasts from a trained checkpoint
├── requirements.txt
│
├── config/                   # One YAML config per dataset
│   ├── custom.yaml           # Template for custom datasets  <-- start here
│   ├── etth1.yaml
│   ├── etth2.yaml
│   ├── ettm1.yaml
│   ├── ettm2.yaml
│   ├── weather.yaml
│   ├── exchange.yaml
│   ├── solar_energy.yaml
│   ├── stock.yaml
│   └── ...
│
├── data/                     # CSV time-series files (one per dataset)
│   ├── etth1.csv, etth2.csv, ettm1.csv, ettm2.csv
│   ├── weather.csv, exchange.csv
│   ├── solar_energy.csv, stock.csv
│
├── models/
│   ├── networks.py           # Denoiser architectures: UNet, AttnNet, LinearNet, ConvLSTMNet
│   ├── precond.py            # Preconditioning: TEDM, EDM, iDDPM
│   ├── noise.py              # Noise schedules: TEDM, EDM, iDDPM
│   ├── loss.py               # Training loss / model wrapper
│   └── filters.py            # Signal processing utilities (Kaiser filter)
│
├── training/
│   ├── trainer.py            # Trainer class: train, evaluate, sample
│   ├── dataset.py            # TimeSeriesDataset: CSV -> sliding windows
│   ├── sampler.py            # Inference-time samplers: TEDM, EDM, DDIM
│   ├── scheduler.py          # LR scheduler with warmup
│   ├── metrics.py            # MSE, MAE, CRPS
│   └── logger.py             # Visdom + MLflow logger
│
└── utils/
    ├── plot.py               # Plot context/target/prediction from saved arrays
    ├── io.py                 # Config loading, dataset listing, result folder management
    └── seed.py
```

---

## Installation

Python 3.10 is required.

### Step 1 — Create a virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> All commands in this guide must be run from the **project root directory** (the folder containing `main.py`).

---

## Quick start

The fastest way to verify the setup using ETTh1:

```bash
# 1. Download ETTh1 from https://github.com/thuml/iTransformer and place it at data/etth1.csv

# 2. Train and evaluate
python main.py --dataset etth1

# 3. Sample and plot 3 random non-overlapping test windows in one step
python -m utils.plot --run-folder results/etth1_00001 --sample-indices random:3 --combine --output forecast.png
```

---

## Reproducing paper results

### Running experiments

Each dataset has a pre-tuned config in `config/`. To train and evaluate:

```bash
python main.py --dataset <dataset_name>
```

| Dataset | Command |
|---------|---------|
| ETTh1 | `python main.py --dataset etth1` |
| ETTh2 | `python main.py --dataset etth2` |
| ETTm1 | `python main.py --dataset ettm1` |
| ETTm2 | `python main.py --dataset ettm2` |
| Weather | `python main.py --dataset weather` |
| Exchange | `python main.py --dataset exchange` |
| Solar Energy | `python main.py --dataset solar_energy` |
| Stock | `python main.py --dataset stock` |

Results are saved to `results/<dataset>_00001/` by default:

| File | Description |
|------|-------------|
| `checkpoint-*.pt` | Model checkpoints saved every `log_freq` steps |
| `config.yaml` | Exact config used for this run (needed by the plotter) |
| `logs/log.txt` | Training log with loss and metric history |
| `resource_usage.csv` | GPU memory and timing statistics |

---

## Training on a new dataset

### Step 1 — Prepare your data

Place your dataset as a CSV file in the `data/` directory:

- One column per feature (multivariate) or a single column (univariate)
- An optional `date` column (automatically dropped if present)
- No missing values (preprocessing should be done beforehand)

Example: `data/my_data.csv`

```
date,feature_1,feature_2,feature_3
2020-01-01,1.23,4.56,7.89
2020-01-02,1.30,4.60,7.95
...
```

### Step 2 — Create a config file

Copy the template and rename it to match your dataset file stem:

```bash
cp config/custom.yaml config/my_data.yaml
```

Open `config/my_data.yaml` and update at minimum:

```yaml
dataloader:
  params:
    path: ./data/my_data.csv   # path to your CSV
    window: 96                 # input context length
    train: 0.7                 # fraction of data for training
    test: 0.2                  # fraction for test (remainder is validation)

model:
  denoiser:
    params:
      seq_len: 96              # must equal window above
      # feat_size is set automatically from the number of columns in your CSV

sampler:
  params:
    horizon: 96                # forecast horizon (how many steps ahead to predict)
```

The `feat_size` parameter is overridden automatically at runtime from the number of feature columns in your CSV.

### Step 3 — Train

```bash
python main.py --dataset my_data
```

The model trains for `trainer.max_epochs` iterations (default: 1000), saving a checkpoint every `logger.log_freq` steps. Test metrics (MSE, MAE) are printed at the end.

---

## Visualization and logging

Three logging backends are available, selected by the `trainer.logger.target` field in your config YAML.

### Option A — File logging only (default)

Metrics are written to a plain-text log in the results folder. No external server needed.

```yaml
trainer:
  logger:
    target: training.logger.BaseLogger
    params:
      enable: True
      print_to_console: False   # set True to also print each step to stdout
```

### Option B — Visdom (live web dashboard)

Streams loss curves and sample predictions to an interactive browser dashboard at `http://localhost:8097`.

```yaml
trainer:
  logger:
    target: training.logger.VisdomLogger
    params:
      enable: True
      server: 'localhost'
      port: 8097
      env_name: 'my_experiment'
```

Start the Visdom server before training:

```bash
python -m visdom.server -p 8097
```

### Option C — MLflow (experiment tracking)

Logs metrics to an MLflow tracking server, useful for comparing runs across hyperparameter sweeps.

```yaml
trainer:
  logger:
    target: training.logger.MLFlowLogger
    params:
      enable: True
      env_name: 'my_experiment'
```

Start the MLflow UI:

```bash
mlflow ui
```

Then open `http://localhost:5000`. Test MSE and MAE are automatically logged when the run completes.

---

## Sampling from a trained model

After training, generate and save predictions for specific windows from the test set:

```bash
python sample.py --dataset my_data --indices 0,1,2 --period test --save-as-numpy
```

This loads the latest checkpoint from `results/my_data_00001/`, runs the sampler for windows 0, 1, and 2, and saves a compressed numpy bundle:

```
results/my_data_00001/samples.npz
  ├── context   # (B, context_len, F)  — historical input
  ├── target    # (B, horizon, F)      — ground truth future
  ├── pred      # (B, horizon, F)      — model predictions (deterministic)
  │             # (B, horizon, F, K)   — K sample tracks (probabilistic)
  └── internal_indices                 — window indices in the dataset
```

| Flag | Description | Default |
|------|-------------|---------|
| `--dataset` | Dataset name (must match a config YAML stem) | `etth1` |
| `--indices` | Comma-separated window indices, e.g. `0,5,10` | `0` |
| `--period` | Split to sample from: `test` or `val` | `test` |
| `--save-as-numpy` | Save output arrays as `.npz` | off |
| `--save-prefix` | Filename prefix for the `.npz` bundle | `samples` |
| `--results-subdir` | Subdirectory inside the run folder to save arrays | `` |
| `--outdir` | Root output directory | `results` |
| `--gpu_id` | GPU device index | `0` |

---

## Generating plots

All plots use a consistent visual style: **black solid** for the input context, **black dotted** for the ground-truth target, and **orange solid** for the model prediction.

The plotter reads `config.yaml` from the run folder to detect whether the sampler was deterministic (single curve) or stochastic (mean + 10th/90th percentile band).

### From a saved `.npz` bundle

```bash
python -m utils.plot \
  --run-folder results/my_data_00001 \
  --prefix samples \
  --sample-index 0 \
  --feature-index 0 \
  --output forecast.png
```

Omit `--output` to show an interactive matplotlib window. A bare filename (no directory) is saved inside the run folder automatically.

**Multiple windows in one figure:**

```bash
python -m utils.plot \
  --run-folder results/my_data_00001 \
  --prefix samples \
  --sample-indices 0,1,2 \
  --combine \
  --output forecast_grid.png
```

### Random sampling mode

Sample and plot N randomly selected, non-overlapping test windows in a single command:

```bash
python -m utils.plot \
  --run-folder results/my_data_00001 \
  --sample-indices random:3 \
  --combine \
  --output forecast_random.png
```

`random:N` picks N start indices from a non-overlapping grid, runs `sample.py` internally, saves `samples_random.npz`, then plots. Use `--period val` to sample from the validation split.

### Multi-dataset mode

Generate a combined figure with one row per trained dataset:

```bash
# All run folders discovered automatically under results/
python -m utils.plot \
  --run-folders results \
  --sample-indices random:1 \
  --output multi_dataset.png

# A specific single dataset
python -m utils.plot \
  --run-folders results/my_data_00001 \
  --sample-indices random:1 \
  --output my_data.png
```

`--run-folders` accepts either a **parent directory** (every subfolder with a `config.yaml` becomes one row, sorted by name) or a **single run folder** (one-row figure).

### Plot options

| Flag | Description | Default |
|------|-------------|---------|
| `--run-folder` | Single experiment run folder | `` |
| `--run-folders` | Parent dir or single run folder for multi-dataset plot | `` |
| `--prefix` | Filename prefix of the `.npz` bundle to load | `samples` |
| `--sample-index` | Single window index to plot | `0` |
| `--sample-indices` | Comma-separated indices, `all`, or `random:N` | `` |
| `--period` | Split to sample from with `random:N`: `test` or `val` | `test` |
| `--combine` | Combine multiple windows into one multi-row figure | off |
| `--feature-index` | Feature dimension to plot | `0` |
| `--output` | Save path (PNG/PDF); omit for interactive display | `` |

---

## Configuration reference

All behaviour is controlled by the YAML config. Key fields:

### `dataloader`

| Field | Description |
|-------|-------------|
| `params.path` | Path to the CSV file |
| `params.train` | Fraction of rows used for training (e.g. `0.7`) |
| `params.test` | Fraction of rows used for test (e.g. `0.2`; remainder is validation) |
| `params.window` | Context window length (number of timesteps the model sees as input) |
| `params.past_k` | Additional conditioning steps prepended to the window (`0` = unconditional) |
| `params.padding` | Set to `1` to disable t-discretization (recommended); `0` otherwise |
| `batch_size` | Training batch size |

### `model.denoiser`

| Field | Description |
|-------|-------------|
| `target` | Architecture class: `models.networks.UNet`, `models.networks.AttnNet`, `models.networks.LinearNet`, `models.networks.ConvLSTMNet` |
| `params.seq_len` | Must equal `dataloader.params.window` |
| `params.feat_size` | Number of features — set automatically from data at runtime |
| `params.kaiser_size` | Kaiser filter size for UNet/AttnNet |
| `params.kaiser_beta` | Kaiser filter beta parameter |

### `model.preconditioning`

| Field | Options |
|-------|---------|
| `target` | `models.precond.TEDM` (default), `models.precond.EDM`, `models.precond.iDDPM` |

### `model.noise_schedule`

| Field | Description |
|-------|-------------|
| `target` | `models.noise.TEDM` (default), `models.noise.EDM`, `models.noise.iDDPM` |
| `params.sigma_data` | Data standard deviation normalisation (typically `1.0` after StandardScaler) |
| `params.scale_from_data` | Set `True` to estimate noise/scale schedules from data (TEDM's key feature) |
| `params.stats` | Statistics accumulation mode: `cumulative` (recommended) or `sliding` |

### `trainer`

| Field | Description |
|-------|-------------|
| `max_epochs` | Total training iterations |
| `ema.decay` | EMA decay for the model used at inference |
| `optimizer.params.lr` | Learning rate |
| `scheduler` | LR scheduler; `ReduceLROnPlateauWithWarmup` with warmup supported |
| `logger.params.enable` | Set `True` to enable logging |
| `logger.params.print_to_console` | Print per-step loss to console |

### `sampler`

| Field | Description |
|-------|-------------|
| `target` | `training.sampler.TEDM` (default), `training.sampler.EDM`, `training.sampler.DDIM` |
| `params.horizon` | Forecast horizon (number of future timesteps to predict) |
| `params.deterministic` | `True` for a single point forecast; `False` for probabilistic multi-track sampling |
| `params.n_tracks` | Number of sample paths when `deterministic: False` |
| `params.backward` | Use backward integration (recommended: `True`) |
| `params.clamp` | Clamp predictions to observed data range |
