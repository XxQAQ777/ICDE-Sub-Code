"""Animate TEDM rollout transformations for presentation-quality visuals.

This script generates rollout animations from a trained run, emphasizing how
the input window is transformed across Euler steps into the prediction horizon.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

if __package__ in (None, ''):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import click
import numpy as np
from omegaconf import OmegaConf
import yaml

from training.trainer import Trainer
from training.dataset import TimeSeriesDataset
from training.rollout_viz import EpisodeSelector, run_episode_sequence
import utils

utils.set_seed(2025)


def _resolve_run_folder(dataset: str, run_folder: str, outdir: str) -> Path:
    if run_folder:
        path = Path(run_folder)
        if not path.exists():
            raise FileNotFoundError(f'run folder does not exist: {path}')
        return path

    root = Path(outdir)
    if not root.exists():
        raise FileNotFoundError(f'output root does not exist: {root}')
    matches = sorted(
        [p for p in root.glob(f'{dataset}_*') if p.is_dir()],
        key=lambda p: p.name
    )
    if not matches:
        raise FileNotFoundError(
            f'No run folders found under {root} for dataset prefix "{dataset}_"'
        )
    return matches[-1]


def _parse_start_token(start: str) -> str | int:
    if start in ('first', 'last'):
        return start
    try:
        return int(start)
    except ValueError as exc:
        raise ValueError(
            f"Invalid --start '{start}'. Use first, last, or an integer index."
        ) from exc


def _latest_checkpoint(run_folder: Path) -> Path:
    ckpts = sorted(
        run_folder.glob('checkpoint-*.pt'),
        key=lambda p: int(p.stem.split('-')[-1]),
    )
    if not ckpts:
        raise FileNotFoundError(
            f'No checkpoints found in {run_folder}. Train first or pass a different run folder.'
        )
    return ckpts[-1]


def _to_plain_types(obj):
    if isinstance(obj, dict):
        return {k: _to_plain_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_types(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_plain_types(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _load_run_config(dataset: str, run_folder: Path):
    cfg_path = run_folder / 'config.yaml'
    if not cfg_path.exists():
        _, configs = utils.load_config(exp=1, dataset=dataset, ablations=False)
        return configs[0]

    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            raw_cfg = yaml.unsafe_load(f)
        raw_cfg = _to_plain_types(raw_cfg)
        return OmegaConf.create(raw_cfg)
    except Exception as exc:
        print(f'[WARN] Could not load {cfg_path} ({exc}). Falling back to config/{dataset}.yaml')
        _, configs = utils.load_config(exp=1, dataset=dataset, ablations=False)
        return configs[0]


@click.command()
@click.option('--dataset',
              help='Dataset name (same as CSV stem in data/).',
              metavar='STR',
              type=click.Choice(utils.list_datasets()),
              required=True)
@click.option('--run-folder',
              help='Path to an existing run folder (preferred).',
              metavar='DIR',
              type=str,
              default='')
@click.option('--outdir',
              help='Root folder used to locate latest run if --run-folder is not provided.',
              metavar='DIR',
              type=str,
              default='results')
@click.option('--gpu-id',
              help='GPU id to use.',
              metavar='INT',
              type=str,
              required=False,
              default='0')
@click.option('--period',
              help='Dataset split to animate.',
              type=click.Choice(['val', 'test']),
              default='test')
@click.option('--start',
              help='Episode start: first, last, or explicit internal index.',
              metavar='STR',
              type=str,
              default='last')
@click.option('--episodes',
              help='Number of episodes to export.',
              metavar='INT',
              type=int,
              default=1)
@click.option('--stride',
              help='Stride multiplier for non-overlapping episode hop (effective hop = stride * input_window_len).',
              metavar='INT',
              type=int,
              default=1)
@click.option('--prepend-mode',
              help='Boundary prepend strategy in TEDM rollout.',
              type=click.Choice(['last', 'first']),
              default='last')
@click.option('--stats-mode',
              help='Override noise stats mode for visualization.',
              type=click.Choice(['as-config', 'cumulative', 'sliding']),
              default='as-config')
@click.option('--feature-index',
              help='Feature index to render.',
              metavar='INT',
              type=int,
              default=0)
@click.option('--max-steps',
              help='Optional cap on rollout Euler steps.',
              metavar='INT',
              type=int,
              default=None)
@click.option('--save-gif',
              help='Save animation as .gif.',
              is_flag=True,
              default=False)
@click.option('--output-dir',
              help='Directory for exported animations.',
              metavar='DIR',
              type=str,
              default='')
@click.option('--fps',
              help='Animation frames per second.',
              metavar='INT',
              type=int,
              default=12)
@click.option('--dpi',
              help='Output DPI.',
              metavar='INT',
              type=int,
              default=120)
@click.option('--interp-frames',
              help='Interpolated frames between Euler steps.',
              metavar='INT',
              type=int,
              default=2)
@click.option('--end-hold-seconds',
              help='Pause duration at the final unfolded prediction before GIF loops.',
              metavar='FLOAT',
              type=float,
              default=1.5)
@click.option('--init-hold-seconds',
              help='Pause duration for the initial black connected input line before dots move.',
              metavar='FLOAT',
              type=float,
              default=1.5)
@click.option('--title',
              help='Optional custom title for the animation.',
              metavar='STR',
              type=str,
              default='')
def main(**kwargs):
    opts = SimpleNamespace(**kwargs)

    run_folder = _resolve_run_folder(opts.dataset, opts.run_folder, opts.outdir)
    print(f'[INFO] Using run folder: {run_folder}')

    config = _load_run_config(opts.dataset, run_folder)
    config.update({'cur_dir': run_folder, 'gpu_id': opts.gpu_id})
    if config.trainer.logger.params.get('enable', False):
        config.trainer.logger.params.update({'enable': False})

    trainer = Trainer(config)
    ckpt = _latest_checkpoint(run_folder)
    trainer.load(int(ckpt.stem.split('-')[-1]))

    seq_len = int(config.model.denoiser.params.seq_len)
    past_k = int(config.dataloader.params.past_k)
    padding = int(config.dataloader.params.padding)
    context_len = seq_len + past_k + padding
    non_overlap_stride = context_len * max(1, int(opts.stride))

    selector = EpisodeSelector(
        period=opts.period,
        start=_parse_start_token(opts.start),
        episodes=opts.episodes,
        stride=non_overlap_stride,
    )

    config.dataloader.params.update({
        'period': opts.period,
        'horizon': config.sampler.params.horizon
    })
    dataset = TimeSeriesDataset(**config.dataloader.params)
    indices = selector.resolve_indices(len(dataset))

    output_dir = Path(opts.output_dir) if opts.output_dir else (run_folder / 'animations')
    output_dir.mkdir(parents=True, exist_ok=True)
    default_stats_mode = str(config.model.noise_schedule.params.get('stats', 'as-config'))
    stats_label = default_stats_mode if opts.stats_mode == 'as-config' else opts.stats_mode

    print(
        f"[INFO] Episodes: {indices} | period={opts.period} | "
        f"prepend={opts.prepend_mode} | stats={stats_label} | "
        f"non_overlap_stride={non_overlap_stride}"
    )

    default_name = (
        f'rollout_seq_start-{indices[0]}_n-{len(indices)}_feat-{opts.feature_index}'
        f'_prepend-{opts.prepend_mode}_stats-{stats_label}.gif'
    )
    output_path = output_dir / default_name
    episode_title = (
        opts.title
        if opts.title
        else (
            f'TEDM Rollout Sequence | {opts.dataset.upper()} dataset | feat={opts.feature_index}'
        )
    )

    result = run_episode_sequence(
        trainer=trainer,
        dataset=dataset,
        dataset_indices=indices,
        feature_index=opts.feature_index,
        prepend_mode=opts.prepend_mode,
        stats_mode=opts.stats_mode,
        max_steps=opts.max_steps,
        title=episode_title,
        fps=opts.fps,
        dpi=opts.dpi,
        interp_frames=opts.interp_frames,
        end_hold_seconds=opts.end_hold_seconds,
        init_hold_seconds=opts.init_hold_seconds,
        save_gif=opts.save_gif,
        output_path=output_path,
    )

    for ep in result['episodes']:
        final_curve = ep['states']['mean'][-1]
        pred = ep['prediction']
        if pred.ndim == 3:
            final_pred = pred[0, :, opts.feature_index]
        else:
            final_pred = pred[0, :, opts.feature_index, :].mean(axis=-1)
        curve_diff = float(np.mean(np.abs(final_curve - final_pred)))
        print(
            f'[INFO] Episode idx={ep["dataset_idx"]}: frames={ep["states"]["mean"].shape[0]} '
            f'final_curve_len={len(final_curve)} | mae(final_curve, rollout)={curve_diff:.6e}'
        )

    if opts.save_gif:
        print(f'[INFO] Saved GIF: {output_path}')
    else:
        import matplotlib.pyplot as plt
        plt.show()

    trainer.release_resources()
    print('[INFO] Done.')


if __name__ == '__main__':
    main()
