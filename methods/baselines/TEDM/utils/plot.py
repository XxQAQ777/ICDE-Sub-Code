from __future__ import annotations
#!/usr/bin/env python
"""Plot sampled windows produced by sample.py.

This utility reads a ``config.yaml`` from the run folder to determine how to
interpret the prediction tensor:

1. Deterministic sampler (``sampler.params.deterministic=True``):
   - pred shape: ``(B, T, F)`` — single prediction curve.
2. Stochastic sampler (``deterministic=False``):
   - pred shape: ``(B, T, F, n_tracks)`` — mean + 10th/90th percentile band.

Plot style: black solid = context, black dotted = target, orange = prediction.

When ``--output`` is a bare filename (no directory), it is saved inside the
run folder automatically.

Usage examples::

    # Single window from a saved bundle
    python -m utils.plot \\
      --run-folder results/etth1_00001 \\
      --prefix samples --sample-index 0 --output plot.png

    # Multiple windows combined in one figure
    python -m utils.plot \\
      --run-folder results/etth1_00001 \\
      --sample-indices 0,1,2 --combine --output grid.png

If --output is omitted, an interactive window is shown (if backend allows).

Random sampling mode
--------------------
Pass ``--sample-indices random:N`` to automatically select N non-overlapping
test windows, invoke ``sample.py`` to generate predictions, and plot the
results in one step. The default count is 3 (``random`` without a suffix).
Control the split with ``--period test|val``.

Example::

    python -m utils.plot --run-folder results/etth1_00001 \\
      --sample-indices random:3 --combine --output forecast.png

Multi-dataset mode
------------------
Pass ``--run-folders <path>`` to generate a single figure with one row per
dataset. Two forms are accepted:

- **Parent directory** (e.g. ``results/``): every subfolder that contains a
  ``config.yaml`` becomes one row, sorted by folder name.
- **Single run folder** (e.g. ``results/etth1_00001``): one-row figure for
  that dataset only.

``--sample-indices`` defaults to ``random:1`` in this mode.

Examples::

    # All trained datasets in one figure
    python -m utils.plot --run-folders results \\
      --sample-indices random:1 --output multi_dataset.png

    # Single dataset via --run-folders
    python -m utils.plot --run-folders results/etth1_00001 \\
      --sample-indices random:1 --output etth1.png
"""
from __future__ import annotations
import argparse
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml
try:
    from omegaconf import OmegaConf
    _OMEGA_AVAILABLE = True
except ImportError:
    _OMEGA_AVAILABLE = False


def load_arrays(run_folder: Path, prefix: str, subdir: str | None = None):
    """Load saved arrays produced by Trainer.sample().

    Priority:
    Single bundled compressed file <prefix>.npz containing keys: context, target, pred.
    """
    base = run_folder if subdir is None else (run_folder / subdir)
    bundle_path = base / f"{prefix}.npz"
    if bundle_path.exists():
        data = np.load(bundle_path)
        required = ['context', 'target', 'pred']
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"Bundled file missing keys: {missing}")
        return data['context'], data['target'], data['pred']
    raise FileNotFoundError(
        f"Could not find bundled file {bundle_path}. "
        "Ensure you ran sample.py with --save-as-numpy after legacy format "
        "removal."
    )


def plot_sample(
        context,
        target,
        pred,
        sample_index: int,
        feature_index: int,
        cfg: dict,
        title: str | None = None):
    """Plot a single sample given config semantics.

    Parameters
    ----------
    context : (B, C, F)
    target  : (B, T, F)
    pred    : (B, T, F) or (B, T, F, K/S)
    cfg     : loaded config dict
    """
    sampler_cfg = cfg.get('sampler', {}).get('params', {})
    deterministic = sampler_cfg.get('deterministic', True)

    B = context.shape[0]
    assert 0 <= sample_index < B, f"sample_index out of range (0..{B-1})"
    ctx = context[sample_index, :, feature_index]
    tgt = target[sample_index, :, feature_index]

    fig, ax = plt.subplots(figsize=(10, 4))
    ctx_len = ctx.shape[0]
    tgt_len = tgt.shape[0]
    t_ctx = np.arange(ctx_len)
    t_tgt = np.arange(ctx_len, ctx_len + tgt_len)
    ax.plot(t_ctx, ctx, label='context', color='black')
    ax.plot(t_tgt, tgt, label='target', color='black', linestyle=':')

    # Case 1: deterministic
    if deterministic:
        if pred.ndim != 3:
            raise ValueError(
                "Deterministic expects pred.ndim == 3")
        prd = pred[sample_index, :, feature_index]
        ax.plot(t_tgt, prd, label='prediction', color='orange')

    # Case 2: stochastic sampler (multiple tracks)
    else:
        if pred.ndim != 4:
            raise ValueError("Stochastic sampler expects pred.ndim == 4")
        tracks = pred[sample_index, :, feature_index, :]  # (T, S)
        qs = np.quantile(tracks, [0.1, 0.9], axis=-1)  # shape (2, T)
        lower = qs[0]
        upper = qs[1]
        mean_curve = tracks.mean(axis=-1)
        ax.plot(t_tgt, mean_curve, label='mean', color='orange')
        ax.plot(
            t_tgt,
            lower,
            label='q0.1',
            color='orange',
            linestyle='--',
            alpha=0.7)
        ax.plot(
            t_tgt,
            upper,
            label='q0.9',
            color='orange',
            linestyle='--',
            alpha=0.7)
        ax.fill_between(
            t_tgt,
            lower,
            upper,
            color='tab:blue',
            alpha=0.15,
            label='interval')

    ax.axvline(ctx_len - 1, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_multiple(
        context,
        target,
        pred,
        sample_indices,
        feature_index: int,
        cfg: dict,
        combine: bool = True,
        base_title: str = "Samples"):
    """Plot multiple samples either as separate figures (combine=False) or a single multi-row figure (combine=True).

    Returns
    -------
    If combine: (fig, axes)
    Else: list[(fig, ax, idx)]
    """
    if not sample_indices:
        raise ValueError("sample_indices must be non-empty")
    if combine:
        n = len(sample_indices)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=False)
        if n == 1:
            axes = [axes]
        sampler_cfg = cfg.get('sampler', {}).get('params', {})
        deterministic = sampler_cfg.get('deterministic', True)
        for ax, idx in zip(axes, sample_indices):
            ctx = context[idx, :, feature_index]
            tgt = target[idx, :, feature_index]
            ctx_len = ctx.shape[0]
            tgt_len = tgt.shape[0]
            t_ctx = np.arange(ctx_len)
            t_tgt = np.arange(ctx_len, ctx_len + tgt_len)
            ax.plot(t_ctx, ctx, label='context', color='black')
            ax.plot(t_tgt, tgt, label='target', color='black', linestyle=':')
            if deterministic:
                prd = pred[idx, :, feature_index]
                ax.plot(t_tgt, prd, label='prediction', color='orange')
            else:
                tracks = pred[idx, :, feature_index, :]
                qs = np.quantile(tracks, [0.1, 0.9], axis=-1)
                lower = qs[0]
                upper = qs[1]
                mean_curve = tracks.mean(axis=-1)
                ax.plot(t_tgt, mean_curve, label='mean', color='orange')
                ax.plot(
                    t_tgt,
                    lower,
                    label='q0.1',
                    color='orange',
                    linestyle='--',
                    alpha=0.7)
                ax.plot(
                    t_tgt,
                    upper,
                    label='q0.9',
                    color='orange',
                    linestyle='--',
                    alpha=0.7)
                ax.fill_between(
                    t_tgt,
                    lower,
                    upper,
                    color='tab:blue',
                    alpha=0.15,
                    label='interval')
            ax.axvline(ctx_len - 1, color='gray', linestyle='--', linewidth=1)
            ax.set_ylabel('Value')
            ax.set_title(f"Sample {idx}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize='small')
        axes[-1].set_xlabel('Time')
        fig.suptitle(base_title)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        return fig, axes
    else:
        figs = []
        for idx in sample_indices:
            fig, ax = plot_sample(
                context, target, pred, idx, feature_index, cfg,
                title=f"Sample {idx} | Feature {feature_index}")
            figs.append((fig, ax, idx))
        return figs


def _load_cfg(run_folder: Path) -> dict:
    """Load config.yaml from a run folder, with fallback to config/<dataset>.yaml."""
    cfg_path = run_folder / 'config.yaml'
    if not cfg_path.exists():
        print(
            f"[WARN] config.yaml not found in {run_folder}. "
            "Falling back to dataset config if possible.")
        stem_parts = run_folder.name.split('_')
        dataset_name = '_'.join(
            stem_parts[:-1]) if len(stem_parts) > 1 else run_folder.name
        candidate = Path('config') / f"{dataset_name}.yaml"
        if candidate.exists():
            cfg_path = candidate
        else:
            raise FileNotFoundError(
                f"Could not find config.yaml in {run_folder} and fallback "
                f"{candidate} is missing.")
    if _OMEGA_AVAILABLE:
        try:
            return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        except Exception:
            pass
    with open(cfg_path, 'r') as f:
        raw = f.read()
    filtered = '\n'.join(
        [ln for ln in raw.splitlines() if '!!python/object' not in ln])
    return yaml.safe_load(filtered)


def _run_random_sampling(
        run_folder: Path,
        cfg: dict,
        n_random: int,
        period: str) -> list:
    """Pick N non-overlapping windows, run sample.py, return chosen indices.

    Saves results to ``<run_folder>/samples_random.npz``.
    """
    dl = cfg.get('dataloader', {}).get('params', {})
    spl = cfg.get('sampler', {}).get('params', {})
    total_w = dl['window'] + dl['past_k'] + dl.get('padding', 0) + spl['horizon']
    csv_path = Path(dl['path'])
    if not csv_path.is_absolute():
        csv_path = Path(__file__).parents[1] / csv_path
    with open(csv_path) as _f:
        n_rows = sum(1 for _ in _f) - 1
    test_rows = math.ceil(n_rows * dl.get('test', 0.2))
    n_avail = max(test_rows - total_w + 1, 0)
    grid = list(range(0, n_avail, total_w))
    if len(grid) == 0:
        print(f"[WARN] No non-overlapping windows available in the {period} "
              f"split for {Path(dl['path']).stem}; skipping.")
        return []
    if len(grid) < n_random:
        print(f"[WARN] Only {len(grid)} non-overlapping window(s) available "
              f"(requested {n_random}); using all available.")
        n_random = len(grid)
    rng = np.random.default_rng()
    chosen = sorted(rng.choice(grid, size=n_random, replace=False).tolist())
    print(f"[INFO] Random non-overlapping indices for "
          f"{Path(dl['path']).stem}: {chosen}")
    project_root = Path(__file__).parents[1]
    dataset_name = Path(dl['path']).stem
    subprocess.run([
        sys.executable, str(project_root / 'sample.py'),
        '--dataset',     dataset_name,
        '--indices',     ','.join(map(str, chosen)),
        '--period',      period,
        '--save-as-numpy',
        '--save-prefix', 'samples_random',
        '--outdir',      str(run_folder.parent.resolve()),
        '--keep-last',   'True',
    ], check=True, cwd=str(project_root))
    return chosen


def plot_multi_dataset(rows, feature_index: int = 0):
    """Plot a datasets × windows grid in a single combined figure.

    Parameters
    ----------
    rows : list of (label, ctx_list, tgt_list, prd_list)
        Each tuple is one dataset row. ``ctx_list``, ``tgt_list``, and
        ``prd_list`` are lists of 1-D arrays, one per window column.
        When all lists have length 1 the result is a single-column figure
        (original behaviour); length > 1 produces a column-per-window grid.
    feature_index : int
        Passed through; unused internally (arrays are already 1-D).

    Returns
    -------
    fig, axes  (axes is always 2-D: n_datasets × n_windows)
    """
    n_rows = len(rows)
    n_cols = len(rows[0][1])
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        squeeze=False)
    legend_handles = None
    for i, (label, ctx_list, tgt_list, prd_list) in enumerate(rows):
        for j, (ctx, tgt, prd) in enumerate(zip(ctx_list, tgt_list, prd_list)):
            ax = axes[i, j]
            ctx_len = ctx.shape[0]
            tgt_len = tgt.shape[0]
            t_ctx = np.arange(ctx_len)
            t_tgt = np.arange(ctx_len, ctx_len + tgt_len)
            l_ctx = ax.plot(
                t_ctx, ctx, color='black', linewidth=1.2, label='context')[0]
            l_tgt = ax.plot(
                t_tgt, tgt, color='black', linestyle=':', label='target')[0]
            l_prd = ax.plot(
                t_tgt, prd, color='orange', linewidth=1.2, label='prediction')[0]
            ax.axvline(ctx_len - 1, color='gray', linestyle='--',
                       linewidth=0.8, alpha=0.6)
            ax.grid(alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=8)
            # Dataset name on the leftmost column only
            if j == 0:
                ax.set_ylabel(label, fontsize=9)
            else:
                ax.set_yticklabels([])
            # Window header on the top row only (only when more than 1 column)
            if i == 0 and n_cols > 1:
                ax.set_title(f"Window {j + 1}", fontsize=9)
            if i == 0 and j == 0:
                legend_handles = [l_ctx, l_tgt, l_prd]
        axes[i, -1]  # ensure last column ticks are visible
    axes[-1, n_cols // 2].set_xlabel('Time')
    if legend_handles is not None:
        fig.legend(
            legend_handles, ['context', 'target', 'prediction'],
            loc='lower center', bbox_to_anchor=(0.5, 0.01),
            ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig, axes


def main():
    parser = argparse.ArgumentParser(
        description='Plot context/target/prediction from saved numpy arrays.')
    parser.add_argument(
        '--run-folder',
        type=str,
        default='',
        help='Path to a single experiment run folder (e.g., results/etth1_00001)')
    parser.add_argument(
        '--run-folders',
        type=str,
        default='',
        help='Path to a parent directory (e.g. results/) or a single run folder '
             'for multi-dataset plot. Each subfolder with config.yaml becomes '
             'one row. Overrides --run-folder.')
    parser.add_argument(
        '--prefix',
        type=str,
        default='samples',
        help='Filename prefix used when saving arrays')
    parser.add_argument(
        '--subdir',
        type=str,
        default='',
        help='Optional subdirectory inside run folder where arrays are stored')
    parser.add_argument(
        '--sample-index', type=int, default=0,
        help='Index of the sample to plot (used if --sample-indices not provided)')
    parser.add_argument(
        '--sample-indices', type=str, default='',
        help='Comma-separated list of sample indices, "all", or "random:N" to '
             'automatically pick N non-overlapping windows and run sample.py '
             '(e.g. --sample-indices random:3; default N=3 if omitted)')
    parser.add_argument(
        '--period', type=str, default='test', choices=['test', 'val'],
        help='Dataset period to sample from (used with --sample-indices random:N)')
    parser.add_argument(
        '--combine',
        action='store_true',
        help='If multiple indices, combine into a single multi-row figure')
    parser.add_argument(
        '--overlay-dir',
        type=str,
        default='',
        help='(Deprecated, ignored)')
    parser.add_argument(
        '--feature-index',
        type=int,
        default=0,
        help='Feature index to plot')
    parser.add_argument(
        '--title',
        type=str,
        default='',
        help='Optional plot title')
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='Path to save figure instead of showing')
    parser.add_argument(
        '--with-externals', action='store_true',
        help='Add external model rows from nsdiff exports for side-by-side comparison')
    parser.add_argument(
        '--externals-dir',
        type=str,
        default='nsdiff/best_mean_first_window_exports',
        help='Directory containing external model npz exports')
    parser.add_argument(
        '--external-win-index',
        type=int,
        default=0,
        help='External window index to load from externals (e.g., win0)')
    parser.add_argument(
        '--panel-width',
        type=float,
        default=2.2,
        help='Width (inches) per panel in externals row grid')
    parser.add_argument(
        '--panel-height',
        type=float,
        default=1.9,
        help='Height (inches) per panel in externals row grid')
    parser.add_argument(
        '--wspace',
        type=float,
        default=0.15,
        help='Horizontal space between panels (matplotlib wspace)')
    args = parser.parse_args()

    # --run-folders: multi-dataset mode
    if args.run_folders:
        p = Path(args.run_folders)
        if (p / 'config.yaml').exists():
            run_folder_list = [p]
        else:
            run_folder_list = sorted(
                [d for d in p.iterdir()
                 if d.is_dir() and (d / 'config.yaml').exists()])
            if not run_folder_list:
                raise FileNotFoundError(
                    f"No run subfolders with config.yaml found in {p}")
        rows = []
        _si = (args.sample_indices.strip().lower()
               if args.sample_indices else 'random:1')
        for rf in run_folder_list:
            cfg = _load_cfg(rf)
            dl = cfg.get('dataloader', {}).get('params', {})
            dataset_label = Path(dl['path']).stem
            if _si.startswith('random'):
                n_rand = int(_si.split(':', 1)[1]) if ':' in _si else 1
                chosen = _run_random_sampling(rf, cfg, n_rand, args.period)
                if not chosen:
                    continue
                prefix = 'samples_random'
            else:
                chosen = [int(i.strip()) for i in _si.split(',') if i.strip()]
                prefix = args.prefix
            ctx, tgt, prd = load_arrays(rf, prefix, args.subdir or None)
            feat = args.feature_index
            sampler_cfg = cfg.get('sampler', {}).get('params', {})
            deterministic = sampler_cfg.get('deterministic', True)
            ctx_list, tgt_list, prd_list = [], [], []
            for k in range(ctx.shape[0]):
                ctx_list.append(ctx[k, :, feat])
                tgt_list.append(tgt[k, :, feat])
                prd_list.append(prd[k, :, feat] if deterministic
                                else prd[k, :, feat].mean(axis=-1))
            rows.append((dataset_label, ctx_list, tgt_list, prd_list))
        fig, _ = plot_multi_dataset(rows, args.feature_index)
        out = args.output
        if out and Path(out).parent == Path('.'):
            out = str(p / out)
        if out:
            fig.savefig(out, dpi=150)
            print(f"Saved figure to {out}")
        else:
            plt.show()
        return

    run_folder = Path(args.run_folder)
    if not args.run_folder:
        raise SystemExit("Either --run-folder or --run-folders is required.")
    # Resolve bare filenames (no directory component) relative to run_folder
    if args.output and Path(args.output).parent == Path('.'):
        args.output = str(run_folder / args.output)

    # Load config early — needed by random mode and by plotting logic below
    cfg = _load_cfg(run_folder)

    # Random mode: pick N non-overlapping windows, run sample.py, then plot
    _si = args.sample_indices.strip().lower()
    if _si.startswith('random'):
        n_random = int(_si.split(':', 1)[1]) if ':' in _si else 3
        chosen = _run_random_sampling(run_folder, cfg, n_random, args.period)
        if not chosen:
            print("[INFO] No windows available to plot; exiting.")
            return
        args.prefix = 'samples_random'
        args.sample_indices = ','.join(map(str, chosen))

    context, target, pred = load_arrays(
        run_folder, args.prefix, args.subdir or None)

    # Determine indices
    if args.sample_indices:
        if args.sample_indices.strip().lower() == 'all':
            indices = list(range(context.shape[0]))
        else:
            indices = [int(i.strip())
                       for i in args.sample_indices.split(',') if i.strip()]
    else:
        indices = [args.sample_index]

    # If bundle includes metadata internal_indices, try to remap to row
    # positions inside arrays
    bundle_internal_indices = None
    try:
        # Reopen bundle to inspect metadata if available
        bundle_path = (
            run_folder if (
                args.subdir == '' or args.subdir is None) else (
                run_folder / args.subdir)) / f"{args.prefix}.npz"
        if bundle_path.exists():
            with np.load(bundle_path) as _d:
                if 'internal_indices' in _d:
                    bundle_internal_indices = _d['internal_indices']
    except Exception:
        pass

    # Decide whether dynamic reconstruction is needed.
    # We only need it if user requested an INTERNAL index not included in
    # saved internal_indices. If user supplied a simple row position (no
    # --external-indices translation) AND there is only one row in the
    # bundle, we treat that as row 0 regardless of its internal index value.
    need_dynamic = False
    if bundle_internal_indices is not None:
        saved_set = set(bundle_internal_indices.tolist())
        missing = [i for i in indices if i not in saved_set]
        # Special case: single-row bundle, user asked for 0 while internal
        # index != 0 -> map index 0 -> that row instead of reconstructing.
        if missing and context.shape[0] == 1 and indices == [0]:
            print(
                f"[INFO] Remapping requested index 0 to sole saved internal index {bundle_internal_indices[0]}.")
            indices = [int(bundle_internal_indices[0])]
            missing = []
        if missing:
            need_dynamic = True

    if need_dynamic:
        print("[INFO] Dynamic reconstruction for indices outside saved bundle.")
        # Rebuild dataset and pull raw windows for those internal indices
        from training.dataset import TimeSeriesDataset
        dl_cfg = cfg.get('dataloader', {}).get('params', {})
        sampler_cfg = cfg.get('sampler', {}).get('params', {})
        ds = TimeSeriesDataset(
            path=dl_cfg['path'],
            window=dl_cfg['window'],
            horizon=sampler_cfg['horizon'],
            past_k=dl_cfg['past_k'],
            padding=dl_cfg['padding'],
            period='test',
            train=dl_cfg['train'],
            test=dl_cfg['test'],
        )
        seq_len = sampler_cfg['horizon']  # predicted length
        context_extra = dl_cfg['past_k'] + dl_cfg['padding'] + dl_cfg['window']
        # For test: each ds sample length = window + past_k + padding + horizon
        ctx_len = dl_cfg['window'] + dl_cfg['past_k'] + dl_cfg['padding']
        new_contexts = []
        new_targets = []
        # Note: predictions cannot be reconstructed here; we will raise if
        # needed
        pred_available = False
        for idx in indices:
            x = ds[idx].numpy()
            new_contexts.append(x[:ctx_len][None, ...])
            new_targets.append(x[ctx_len:ctx_len + seq_len][None, ...])
        context = np.concatenate(new_contexts, axis=0)
        target = np.concatenate(new_targets, axis=0)
        pred = np.zeros_like(target)  # placeholder
        if args.output:
            print("[WARN] Predictions not available for dynamically "
                  "reconstructed indices (outside sampling bundle). "
                  "Only context/target plotted.")

    if not need_dynamic:
        if bundle_internal_indices is not None:
            idx_pos = {
                val: pos for pos, val in enumerate(
                    bundle_internal_indices.tolist())}
            try:
                row_positions = [idx_pos[i] for i in indices]
            except KeyError as e:
                raise KeyError(
                    f"Requested internal index {e} not found in saved "
                    f"bundle indices {bundle_internal_indices.tolist()}."
                )
            context = context[row_positions]
            target = target[row_positions]
            pred = pred[row_positions]
            # After remap, local plotting indices are sequential 0..n-1
            indices = list(range(len(row_positions)))
        else:
            # treat indices directly as row positions
            for i in indices:
                if i < 0 or i >= context.shape[0]:
                    raise IndexError(
                        f"Row index {i} out of range (0..{context.shape[0]-1})")
            context = context[indices]
            target = target[indices]
            pred = pred[indices]
            indices = list(range(len(indices)))

    # Multi-sample plotting
    if len(indices) > 1:
        title = args.title or "Forecast samples"
        if args.combine:
            fig, _ = plot_multiple(
                context, target, pred, indices, args.feature_index, cfg,
                combine=True, base_title=title)
            if args.output:
                fig.savefig(args.output, dpi=150)
                print(f"Saved figure to {args.output}")
            else:
                plt.show()
        else:
            figs = plot_multiple(
                context, target, pred, indices, args.feature_index, cfg,
                combine=False)
            for fig, _, idx in figs:
                out = args.output.replace('.png', f'_{idx}.png') if args.output else ''
                if out:
                    fig.savefig(out, dpi=150)
                    print(f"Saved figure to {out}")
                else:
                    plt.show()
        return

    # Single-sample plotting (after all substitutions). Optionally include
    # external model rows.
    if len(indices) == 1:
        idx = indices[0]
        if not args.with_externals:
            title = args.title or f"Sample {idx} | Feature {args.feature_index}"
            fig, _ = plot_sample(
                context, target, pred, idx, args.feature_index, cfg,
                title=title)
            if args.output:
                fig.savefig(args.output, dpi=150)
                print(f"Saved figure to {args.output}")
            else:
                plt.show()
            return
        else:
            # Build a multi-row grid: first row TEDM (current arrays), subsequent rows from externals
            # Prepare TEDM 1D series
            sampler_cfg = cfg.get('sampler', {}).get('params', {})
            deterministic = sampler_cfg.get('deterministic', True)
            feat = args.feature_index
            ctx = context[idx, :, feat]
            tgt = target[idx, :, feat]
            if deterministic:
                prd = pred[idx, :, feat]
            else:
                # use mean across tracks
                prd = pred[idx, :, feat].mean(axis=-1)

            rows = [("TEDM", ctx, tgt, prd)]
            # Determine dataset stem to pick external files
            dl_cfg = cfg.get('dataloader', {}).get('params', {})
            dataset_stem = Path(dl_cfg['path']).stem  # ettm2, etth1, etc.
            # Map to external prefix like ETTm2 or ETTh1 (preserve lowercase
            # m/h)
            external_prefix = 'ETT' + dataset_stem[3:]
            ext_dir = Path(args.externals_dir)
            if not ext_dir.exists():
                print(f"[WARN] Externals dir not found: {ext_dir}")
            else:
                pattern = f"{external_prefix}_*_win{args.external_win_index}.npz"
                matches = sorted([p for p in ext_dir.glob(pattern)])
                ext_rows = {}
                for p in matches:
                    name = p.stem  # e.g., ETTm2_NsDiff_win0
                    try:
                        model_name = name.split('_', 1)[1].rsplit('_win', 1)[0]
                    except Exception:
                        model_name = name
                    try:
                        with np.load(p) as ed:
                            h = ed['history']
                            t = ed['target']
                            m = ed['mean'] if 'mean' in ed else None
                            # Ensure shapes are (T,F)
                            if h.ndim == 2 and h.shape[0] == 7 and h.shape[1] != 7:
                                h = h.T
                            if t.ndim == 2 and t.shape[0] == 7 and t.shape[1] != 7:
                                t = t.T
                            if m is not None and m.ndim == 2 and m.shape[0] == 7 and m.shape[1] != 7:
                                m = m.T
                            if m is None:
                                print(
                                    f"[WARN] External file {p.name} has no 'mean'; skipping.")
                                continue
                            c_row = h[:, feat]
                            t_row = t[:, feat]
                            p_row = m[:, feat]
                            shift = t_row.mean() - p_row.mean()
                            p_row = p_row + shift
                            ext_rows[model_name] = (
                                model_name, c_row, t_row, p_row)
                    except Exception as e:
                        print(f"[WARN] Failed to load external {p.name}: {e}")
                # Desired left-to-right order
                desired = ["NsDiff", "TMDM", "DiffusionTS", "TimeDiff"]
                for key in desired:
                    if key in ext_rows:
                        rows.append(ext_rows[key])

            n = len(rows)
            # Small panels for paper-friendly figure; configurable via CLI
            per_w, per_h = float(args.panel_width), float(args.panel_height)
            fig, axes = plt.subplots(
                1, n, figsize=(
                    per_w * n, per_h), sharey=True)
            fig.subplots_adjust(wspace=float(args.wspace))
            if n == 1:
                axes = [axes]
            legend_handles = None
            for i, (ax, (title_name, c, t, pred)) in enumerate(zip(axes, rows)):
                ctx_len = c.shape[0]
                tgt_len = t.shape[0]
                t_ctx = np.arange(ctx_len)
                t_tgt = np.arange(ctx_len, ctx_len + tgt_len)
                # Styles: context black, target black dotted, prediction
                # orange; no vertical separator
                l_ctx = ax.plot(
                    t_ctx,
                    c,
                    label='context',
                    color='black',
                    linewidth=1.0)[0]
                l_tgt = ax.plot(
                    t_tgt,
                    t,
                    label='target',
                    color='black',
                    linestyle=':')[0]
                l_prd = ax.plot(
                    t_tgt,
                    pred,
                    label='prediction',
                    color='orange',
                    linewidth=0.9)[0]
                if i == 0:
                    legend_handles = [l_ctx, l_tgt, l_prd]
                ax.set_title(title_name, fontsize=10)
                ax.grid(alpha=0.25)
                # Compact ticks and labels for small panels
                ax.tick_params(axis='both', which='major', labelsize=8)
                # Hide y tick labels for inner panels to reduce clutter
                if i > 0:
                    ax.set_yticklabels([])
            # Common labels, shared legend, and tight layout
            # Centered legend like v2, but nudged above the x-axis to avoid
            # overlap
            if legend_handles is not None:
                fig.legend(
                    legend_handles, ['context', 'target', 'prediction'],
                    loc='lower center', bbox_to_anchor=(0.5, 0.02),
                    ncol=3, frameon=False, fontsize=9)
            if args.title:
                fig.suptitle(args.title, fontsize=11)
                fig.tight_layout(rect=[0, 0.12, 1, 0.92])
            else:
                fig.tight_layout(rect=[0, 0.12, 1, 1])
            # Place the y-label centered with respect to panel heights (after
            # layout)
            try:
                y0 = axes[0].get_position().y0
                y1 = axes[0].get_position().y1
                y_mid = 0.5 * (y0 + y1)
                fig.text(
                    0.005,
                    y_mid,
                    dataset_stem.upper(),
                    rotation=90,
                    va='center',
                    ha='center',
                    fontsize=9)
            except Exception:
                # Fallback to figure center if positions unavailable
                fig.text(
                    0.005,
                    0.5,
                    dataset_stem.upper(),
                    rotation=90,
                    va='center',
                    ha='center',
                    fontsize=9)
            if args.output:
                fig.savefig(args.output, dpi=300)
                print(f"Saved figure to {args.output}")
            else:
                plt.show()
            return


if __name__ == '__main__':
    main()
