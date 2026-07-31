from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter

from utils import stats as stats_utils


@dataclass
class EpisodeSelector:
    period: str = 'val'
    start: str | int = 'last'
    episodes: int = 1
    stride: int = 1

    def resolve_indices(self, num_samples: int) -> list[int]:
        if num_samples <= 0:
            raise ValueError('Empty dataset split. No episodes available.')
        if self.episodes < 1:
            raise ValueError('episodes must be >= 1')
        if self.stride < 1:
            raise ValueError('stride must be >= 1')

        if self.start == 'first':
            first_idx = 0
        elif self.start == 'last':
            first_idx = num_samples - 1 - (self.episodes - 1) * self.stride
        elif isinstance(self.start, int):
            first_idx = self.start
        else:
            raise ValueError(
                f"Invalid start='{self.start}'. Use 'first', 'last', or an integer index."
            )

        if first_idx < 0 or first_idx >= num_samples:
            raise IndexError(
                f"start index {first_idx} out of bounds for split of size {num_samples}"
            )

        indices = [first_idx + i * self.stride for i in range(self.episodes)]
        if indices[-1] >= num_samples:
            raise IndexError(
                f'Requested episodes exceed split size ({num_samples}). '
                f'Last requested index: {indices[-1]}'
            )
        return indices


class TraceCollector:
    """Collects rollout windows for every Euler step."""

    def __init__(self, deterministic: bool, quantiles: tuple[float, float] = (0.1, 0.9)):
        self.deterministic = deterministic
        self.quantiles = quantiles
        self._steps: list[int] = []
        self._windows: list[np.ndarray] = []

    def __call__(self, *, phase: str, step: int, x_window: torch.Tensor | None, **_: Any):
        if x_window is None:
            return
        arr = x_window.detach().float().cpu().numpy()
        self._steps.append(int(step))
        self._windows.append(arr)

    @property
    def num_states(self) -> int:
        return len(self._windows)

    @property
    def steps(self) -> list[int]:
        return list(self._steps)

    def to_feature_series(self, feature_index: int) -> dict[str, np.ndarray]:
        if not self._windows:
            raise RuntimeError('No rollout states were collected.')

        first = self._windows[0]
        if first.ndim != 3:
            raise ValueError(f'Expected window tensor with ndim=3. Got {first.ndim}.')
        if feature_index < 0 or feature_index >= first.shape[1]:
            raise IndexError(
                f'feature_index={feature_index} out of bounds for F={first.shape[1]}'
            )

        if self.deterministic:
            mean = np.stack([w[0, feature_index, :] for w in self._windows], axis=0)
            return {'mean': mean}

        q_low, q_high = self.quantiles
        per_state = [w[:, feature_index, :] for w in self._windows]  # [(tracks, seq_len), ...]
        mean = np.stack([x.mean(axis=0) for x in per_state], axis=0)
        lower = np.stack([np.quantile(x, q_low, axis=0) for x in per_state], axis=0)
        upper = np.stack([np.quantile(x, q_high, axis=0) for x in per_state], axis=0)
        return {'mean': mean, 'lower': lower, 'upper': upper}


@dataclass
class AnimationStyle:
    input_color: str = '#111111'
    pred_color: str = '#F28E2B'
    target_color: str = '#2A9D8F'
    context_span_color: str = '#EAEAEA'
    forecast_span_color: str = '#FFF1E0'
    grid_color: str = '#C8C8C8'
    bg_color: str = '#FBFBFB'
    input_lw: float = 2.2
    pred_lw: float = 2.8
    target_lw: float = 2.2
    band_alpha: float = 0.22
    frame_lw: float = 0.9
    title_size: int = 18
    subtitle_size: int = 12
    tick_size: int = 11


class RolloutAnimator:
    def __init__(self, style: AnimationStyle | None = None, fps: int = 12, dpi: int = 120,
                 interp_frames: int = 2, end_hold_seconds: float = 1.5,
                 init_hold_seconds: float = 1.5):
        self.style = style or AnimationStyle()
        self.fps = int(fps)
        self.dpi = int(dpi)
        self.interp_frames = max(0, int(interp_frames))
        self.end_hold_seconds = max(0.0, float(end_hold_seconds))
        self.init_hold_seconds = max(0.0, float(init_hold_seconds))

    @staticmethod
    def _apply_minimal_axes(ax, x_limits: tuple[float, float], y_limits: tuple[float, float]):
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        for spine in ax.spines.values():
            spine.set_visible(False)

    @staticmethod
    def _trim_figure(fig: plt.Figure):
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.93)

    def _build_render_frames(
        self,
        mean_states: np.ndarray,
        seq_len: int,
        start0: int,
        lower_states: np.ndarray | None = None,
        upper_states: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        def _prepend_initial_hold(local_frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if not local_frames:
                return local_frames
            hold_frames = int(round(self.init_hold_seconds * max(1, self.fps)))
            if hold_frames <= 0:
                return local_frames
            first = local_frames[0]
            held = []
            for _ in range(hold_frames):
                hold_frame = dict(first)
                hold_frame['hold'] = True
                held.append(hold_frame)
            return held + local_frames

        n_states = mean_states.shape[0]
        starts = start0 + np.arange(n_states, dtype=np.float32)
        frames: list[dict[str, Any]] = []

        if n_states == 1:
            x = np.arange(seq_len, dtype=np.float32) + starts[0]
            frame = {'step': 0, 'x': x, 'mean': mean_states[0]}
            if lower_states is not None and upper_states is not None:
                frame['lower'] = lower_states[0]
                frame['upper'] = upper_states[0]
            frames.append(frame)
            hold_frames = int(round(self.end_hold_seconds * max(1, self.fps)))
            for _ in range(hold_frames):
                hold_frame = {'step': 0, 'x': x, 'mean': mean_states[0], 'hold': True}
                if lower_states is not None and upper_states is not None:
                    hold_frame['lower'] = lower_states[0]
                    hold_frame['upper'] = upper_states[0]
                frames.append(hold_frame)
            return _prepend_initial_hold(frames)

        for i in range(n_states - 1):
            for j in range(self.interp_frames + 1):
                alpha = j / float(self.interp_frames + 1)
                x_start = (1.0 - alpha) * starts[i] + alpha * starts[i + 1]
                x = np.arange(seq_len, dtype=np.float32) + x_start
                mean = (1.0 - alpha) * mean_states[i] + alpha * mean_states[i + 1]
                frame = {'step': i + alpha, 'x': x, 'mean': mean}
                if lower_states is not None and upper_states is not None:
                    frame['lower'] = (
                        (1.0 - alpha) * lower_states[i] + alpha * lower_states[i + 1]
                    )
                    frame['upper'] = (
                        (1.0 - alpha) * upper_states[i] + alpha * upper_states[i + 1]
                    )
                frames.append(frame)

        x_last = np.arange(seq_len, dtype=np.float32) + starts[-1]
        last = {'step': float(n_states - 1), 'x': x_last, 'mean': mean_states[-1]}
        if lower_states is not None and upper_states is not None:
            last['lower'] = lower_states[-1]
            last['upper'] = upper_states[-1]
        frames.append(last)
        hold_frames = int(round(self.end_hold_seconds * max(1, self.fps)))
        if hold_frames > 0:
            for _ in range(hold_frames):
                hold_frame = {
                    'step': float(n_states - 1),
                    'x': x_last,
                    'mean': mean_states[-1],
                    'hold': True,
                }
                if lower_states is not None and upper_states is not None:
                    hold_frame['lower'] = lower_states[-1]
                    hold_frame['upper'] = upper_states[-1]
                frames.append(hold_frame)
        return _prepend_initial_hold(frames)

    def create_animation(
        self,
        context: np.ndarray,
        target: np.ndarray,
        mean_states: np.ndarray,
        lower_states: np.ndarray | None = None,
        upper_states: np.ndarray | None = None,
        prepend_mode: str = 'last',
        stats_mode: str = 'cumulative',
        title: str | None = None,
    ) -> tuple[plt.Figure, FuncAnimation]:
        style = self.style
        ctx_len = context.shape[0]
        seq_len = target.shape[0]
        total_len = ctx_len + seq_len
        pred_start = ctx_len - seq_len

        start0 = pred_start
        if start0 < 0:
            raise ValueError('context length must be >= prediction length')

        render_frames = self._build_render_frames(
            mean_states=mean_states,
            lower_states=lower_states,
            upper_states=upper_states,
            seq_len=seq_len,
            start0=start0,
        )
        steps_total = max(0, mean_states.shape[0] - 1)

        y_pool = [context, target, mean_states.reshape(-1)]
        if lower_states is not None:
            y_pool.append(lower_states.reshape(-1))
        if upper_states is not None:
            y_pool.append(upper_states.reshape(-1))
        y_min = float(np.min(np.concatenate(y_pool)))
        y_max = float(np.max(np.concatenate(y_pool)))
        y_pad = max(1e-6, 0.08 * (y_max - y_min))
        y_limits = (y_min - y_pad, y_max + y_pad)

        fig, ax = plt.subplots(figsize=(16, 9), dpi=self.dpi)
        fig.patch.set_facecolor('white')
        ax.set_facecolor(style.bg_color)

        t_ctx = np.arange(ctx_len)
        t_tgt = np.arange(ctx_len, total_len)

        ax.axvspan(0, ctx_len - 1, color=style.context_span_color, alpha=0.65, zorder=0)
        ax.axvspan(ctx_len - 0.5, total_len - 1, color=style.forecast_span_color, alpha=0.65, zorder=0)
        ax.axvline(ctx_len - 0.5, color='#808080', linestyle='--', linewidth=1.2, zorder=1)

        context_line, = ax.plot(
            t_ctx,
            context,
            color=style.input_color,
            linewidth=style.input_lw,
            label='input',
            zorder=2,
        )
        target_line, = ax.plot(
            [],
            [],
            color=style.target_color,
            linewidth=style.target_lw,
            label='target',
            zorder=3,
        )
        pred_line, = ax.plot(
            [],
            [],
            color=style.pred_color,
            linewidth=style.pred_lw,
            label='prediction',
            zorder=4,
        )
        pred_scatter = ax.scatter(
            [],
            [],
            s=44,
            c=[style.input_color],
            edgecolors='none',
            zorder=5,
        )
        band_ref = {'poly': None}
        orange_rgb = np.asarray(mcolors.to_rgb(style.pred_color), dtype=np.float32)
        initial_values = mean_states[0]
        max_delta = float(np.max(np.abs(mean_states - initial_values[None, :])))
        max_delta = max(max_delta, 1e-8)

        base_title = title or 'TEDM Rollout'
        ax.set_title(base_title, fontsize=style.title_size, pad=12)
        self._apply_minimal_axes(ax, x_limits=(-1, total_len), y_limits=y_limits)
        self._trim_figure(fig)

        def update(frame_idx: int):
            frame = render_frames[frame_idx]
            step_pos = float(frame['step'])
            progress = 1.0 if steps_total == 0 else min(1.0, step_pos / float(steps_total))
            step_idx = int(np.floor(step_pos + 1e-8))
            ax.set_title(
                f'{base_title} | Episode 1/1 | Euler step {step_idx}/{steps_total}',
                fontsize=style.title_size,
                pad=12,
            )

            x_vals = frame['x']
            y_vals = frame['mean']
            pred_scatter.set_offsets(np.column_stack([x_vals, y_vals]))

            delta = np.abs(y_vals - initial_values) / max_delta
            brightness = np.clip(progress + 0.35 * delta * (1.0 - progress), 0.0, 1.0)
            colors = orange_rgb[None, :] * brightness[:, None]
            pred_scatter.set_facecolors(colors)
            pred_scatter.set_edgecolors(colors)

            show_initial = step_pos <= 1e-9
            show_final = progress >= (1.0 - 1e-9)
            show_context = show_initial or show_final
            if show_context:
                context_line.set_data(t_ctx, context)
                context_line.set_alpha(1.0)
            else:
                context_line.set_data([], [])

            if show_final:
                pred_line.set_data(x_vals, y_vals)
                pred_line.set_alpha(1.0)
                target_line.set_data(t_tgt, target)
                target_line.set_alpha(1.0)
            else:
                pred_line.set_data([], [])
                target_line.set_data([], [])

            if band_ref['poly'] is not None:
                band_ref['poly'].remove()
                band_ref['poly'] = None
            artists = [context_line, pred_scatter, pred_line, target_line]

            if 'lower' in frame and 'upper' in frame and show_final:
                band_ref['poly'] = ax.fill_between(
                    x_vals,
                    frame['lower'],
                    frame['upper'],
                    color=style.pred_color,
                    alpha=style.band_alpha,
                    linewidth=0,
                    zorder=2,
                )
                artists.append(band_ref['poly'])
            return artists

        anim = FuncAnimation(
            fig,
            update,
            frames=len(render_frames),
            interval=1000 / max(1, self.fps),
            blit=False,
            repeat=True,
        )
        return fig, anim

    def create_episode_sequence_animation(
        self,
        episodes: list[dict[str, Any]],
        prepend_mode: str = 'last',
        stats_mode: str = 'cumulative',
        title: str | None = None,
    ) -> tuple[plt.Figure, FuncAnimation]:
        if not episodes:
            raise ValueError('episodes must be non-empty')

        style = self.style
        ctx_len = int(episodes[0]['context'].shape[0])
        seq_len = int(episodes[0]['target'].shape[0])
        total_len = ctx_len + seq_len

        for ep in episodes:
            if ep['context'].shape[0] != ctx_len or ep['target'].shape[0] != seq_len:
                raise ValueError('All episodes must share the same context and target lengths.')

        pred_start = ctx_len - seq_len
        if pred_start < 0:
            raise ValueError('context length must be >= prediction length')

        y_parts = []
        for ep in episodes:
            y_parts.extend([ep['context'], ep['target'], ep['mean_states'].reshape(-1)])
            if ep.get('lower_states', None) is not None:
                y_parts.append(ep['lower_states'].reshape(-1))
            if ep.get('upper_states', None) is not None:
                y_parts.append(ep['upper_states'].reshape(-1))

        y_min = float(np.min(np.concatenate(y_parts)))
        y_max = float(np.max(np.concatenate(y_parts)))
        y_pad = max(1e-6, 0.08 * (y_max - y_min))
        y_limits = (y_min - y_pad, y_max + y_pad)

        start0 = pred_start
        t_ctx = np.arange(ctx_len)
        t_tgt = np.arange(ctx_len, total_len)
        orange_rgb = np.asarray(mcolors.to_rgb(style.pred_color), dtype=np.float32)

        episode_meta: list[dict[str, Any]] = []
        timeline: list[tuple[int, dict[str, Any]]] = []
        for ep_idx, ep in enumerate(episodes):
            mean_states = ep['mean_states']
            lower_states = ep.get('lower_states', None)
            upper_states = ep.get('upper_states', None)
            local_frames = self._build_render_frames(
                mean_states=mean_states,
                lower_states=lower_states,
                upper_states=upper_states,
                seq_len=seq_len,
                start0=start0,
            )
            initial_values = mean_states[0]
            max_delta = float(np.max(np.abs(mean_states - initial_values[None, :])))
            episode_meta.append({
                'steps_total': max(0, mean_states.shape[0] - 1),
                'initial_values': initial_values,
                'max_delta': max(max_delta, 1e-8),
                'context': ep['context'],
                'target': ep['target'],
                'dataset_idx': ep['dataset_idx'],
            })
            for frame in local_frames:
                timeline.append((ep_idx, frame))

        fig, ax = plt.subplots(figsize=(16, 9), dpi=self.dpi)
        fig.patch.set_facecolor('white')
        ax.set_facecolor(style.bg_color)

        ax.axvspan(0, ctx_len - 1, color=style.context_span_color, alpha=0.65, zorder=0)
        ax.axvspan(ctx_len - 0.5, total_len - 1, color=style.forecast_span_color, alpha=0.65, zorder=0)
        ax.axvline(ctx_len - 0.5, color='#808080', linestyle='--', linewidth=1.2, zorder=1)

        context_line, = ax.plot(
            [],
            [],
            color=style.input_color,
            linewidth=style.input_lw,
            label='input',
            zorder=2,
        )
        target_line, = ax.plot(
            [],
            [],
            color=style.target_color,
            linewidth=style.target_lw,
            label='target',
            zorder=3,
        )
        pred_line, = ax.plot(
            [],
            [],
            color=style.pred_color,
            linewidth=style.pred_lw,
            label='prediction',
            zorder=4,
        )
        pred_scatter = ax.scatter(
            [],
            [],
            s=44,
            c=[style.input_color],
            edgecolors='none',
            zorder=5,
        )
        band_ref = {'poly': None}

        base_title = title or 'TEDM Rollout Sequence'
        ax.set_title(base_title, fontsize=style.title_size, pad=12)
        self._apply_minimal_axes(ax, x_limits=(-1, total_len), y_limits=y_limits)
        self._trim_figure(fig)
        total_episodes = len(episodes)

        def update(frame_idx: int):
            ep_idx, frame = timeline[frame_idx]
            meta = episode_meta[ep_idx]

            step_pos = float(frame['step'])
            steps_total = meta['steps_total']
            progress = 1.0 if steps_total == 0 else min(1.0, step_pos / float(steps_total))
            step_idx = int(np.floor(step_pos + 1e-8))
            ax.set_title(
                (
                    f'{base_title} | Episode {ep_idx + 1}/{total_episodes} '
                    f'| Euler step {step_idx}/{steps_total}'
                ),
                fontsize=style.title_size,
                pad=12,
            )

            x_vals = frame['x']
            y_vals = frame['mean']
            pred_scatter.set_offsets(np.column_stack([x_vals, y_vals]))

            delta = np.abs(y_vals - meta['initial_values']) / meta['max_delta']
            brightness = np.clip(progress + 0.35 * delta * (1.0 - progress), 0.0, 1.0)
            colors = orange_rgb[None, :] * brightness[:, None]
            pred_scatter.set_facecolors(colors)
            pred_scatter.set_edgecolors(colors)

            show_initial = step_pos <= 1e-9
            show_final = progress >= (1.0 - 1e-9)
            show_context = show_initial or show_final
            if show_context:
                context_line.set_data(t_ctx, meta['context'])
                context_line.set_alpha(1.0)
            else:
                context_line.set_data([], [])

            if show_final:
                pred_line.set_data(x_vals, y_vals)
                pred_line.set_alpha(1.0)
                target_line.set_data(t_tgt, meta['target'])
                target_line.set_alpha(1.0)
            else:
                pred_line.set_data([], [])
                target_line.set_data([], [])

            if band_ref['poly'] is not None:
                band_ref['poly'].remove()
                band_ref['poly'] = None
            artists = [context_line, pred_scatter, pred_line, target_line]

            if 'lower' in frame and 'upper' in frame and show_final:
                band_ref['poly'] = ax.fill_between(
                    x_vals,
                    frame['lower'],
                    frame['upper'],
                    color=style.pred_color,
                    alpha=style.band_alpha,
                    linewidth=0,
                    zorder=2,
                )
                artists.append(band_ref['poly'])
            return artists

        anim = FuncAnimation(
            fig,
            update,
            frames=len(timeline),
            interval=1000 / max(1, self.fps),
            blit=False,
            repeat=True,
        )
        return fig, anim

    def save_gif(self, animation: FuncAnimation, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = PillowWriter(fps=max(1, self.fps))
        animation.save(
            str(output_path),
            writer=writer,
            dpi=self.dpi,
            savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0},
        )


def _resolve_stats_mode(noise_sch, stats_mode: str) -> str:
    if stats_mode == 'as-config':
        return getattr(noise_sch, 'stats', 'as-config')
    return stats_mode


@contextmanager
def override_tedm_stats(noise_sch, stats_mode: str):
    if stats_mode == 'as-config':
        yield
        return
    if stats_mode not in ('cumulative', 'sliding'):
        raise ValueError(
            f"Invalid stats_mode='{stats_mode}'. Use 'as-config', 'cumulative', or 'sliding'."
        )
    if not all(hasattr(noise_sch, attr) for attr in ('stats', '_mean', '_var')):
        raise TypeError(
            'stats_mode override is only supported for TEDM noise schedules '
            'with stats/_mean/_var attributes.'
        )

    old_stats = noise_sch.stats
    old_mean = noise_sch._mean
    old_var = noise_sch._var

    if stats_mode == 'cumulative':
        mean_fn = stats_utils.cumulative_mean
        var_fn = stats_utils.cumulative_var
    else:
        mean_fn = stats_utils.sliding_mean
        var_fn = stats_utils.sliding_var

    noise_sch.stats = stats_mode
    noise_sch._mean = partial(
        mean_fn,
        window_size=getattr(noise_sch, 'sliding_window', None),
        ema_beta=getattr(noise_sch, 'ema_beta', None),
    )
    noise_sch._var = partial(
        var_fn,
        window_size=getattr(noise_sch, 'sliding_window', None),
        ema_beta=getattr(noise_sch, 'ema_beta', None),
    )

    try:
        yield
    finally:
        noise_sch.stats = old_stats
        noise_sch._mean = old_mean
        noise_sch._var = old_var


def run_episode(
    trainer,
    dataset,
    dataset_idx: int,
    feature_index: int,
    prepend_mode: str = 'last',
    stats_mode: str = 'as-config',
    max_steps: int | None = None,
    title: str | None = None,
    fps: int = 12,
    dpi: int = 120,
    interp_frames: int = 2,
    end_hold_seconds: float = 1.5,
    init_hold_seconds: float = 1.5,
    save_gif: bool = False,
    output_path: Path | None = None,
) -> dict[str, Any]:
    payload = _collect_episode_rollout(
        trainer=trainer,
        dataset=dataset,
        dataset_idx=dataset_idx,
        feature_index=feature_index,
        prepend_mode=prepend_mode,
        stats_mode=stats_mode,
        max_steps=max_steps,
    )

    animator = RolloutAnimator(
        fps=fps,
        dpi=dpi,
        interp_frames=interp_frames,
        end_hold_seconds=end_hold_seconds,
        init_hold_seconds=init_hold_seconds,
    )
    fig, anim = animator.create_animation(
        context=payload['context'],
        target=payload['target'],
        mean_states=payload['mean_states'],
        lower_states=payload.get('lower_states', None),
        upper_states=payload.get('upper_states', None),
        prepend_mode=prepend_mode,
        stats_mode=payload['resolved_stats_mode'],
        title=title,
    )

    if save_gif:
        if output_path is None:
            raise ValueError('output_path must be provided when save_gif=True')
        animator.save_gif(anim, output_path)
        plt.close(fig)

    return {
        'figure': fig,
        'animation': anim,
        'prediction': payload['prediction'],
        'states': payload['states'],
        'context': payload['context'],
        'target': payload['target'],
        'resolved_stats_mode': payload['resolved_stats_mode'],
    }


def _collect_episode_rollout(
    trainer,
    dataset,
    dataset_idx: int,
    feature_index: int,
    prepend_mode: str = 'last',
    stats_mode: str = 'as-config',
    max_steps: int | None = None,
) -> dict[str, Any]:
    seq_len = int(trainer.config.model.denoiser.params.seq_len)
    past_k = int(trainer.config.dataloader.params.past_k)
    padding = int(trainer.config.dataloader.params.padding)
    context_len = seq_len + past_k + padding

    if dataset_idx < 0 or dataset_idx >= len(dataset):
        raise IndexError(
            f'dataset_idx={dataset_idx} out of bounds for split with size {len(dataset)}'
        )

    sample = dataset[dataset_idx]
    context_window = sample[:context_len, :]         # (context_len, F)
    target_window = sample[context_len:context_len + seq_len, :]  # (seq_len, F)

    cond = context_window.unsqueeze(0).to(trainer.model.device)

    collector = TraceCollector(deterministic=trainer.sampler.deterministic)
    with torch.no_grad():
        with trainer.use_eval_noise_sch():
            with override_tedm_stats(trainer.sampler.noise_sch, stats_mode):
                pred = trainer.sampler.rollout(
                    cond,
                    prepend_mode=prepend_mode,
                    trace_hook=collector,
                    max_steps=max_steps,
                )

    traces = collector.to_feature_series(feature_index)
    context_series = context_window[:, feature_index].detach().cpu().numpy()
    target_series = target_window[:, feature_index].detach().cpu().numpy()
    states = {'mean': traces['mean']}
    if 'lower' in traces and 'upper' in traces:
        states['lower'] = traces['lower']
        states['upper'] = traces['upper']

    resolved_stats_mode = _resolve_stats_mode(trainer.sampler.noise_sch, stats_mode)
    return {
        'dataset_idx': dataset_idx,
        'prediction': pred.detach().cpu().numpy(),
        'states': states,
        'context': context_series,
        'target': target_series,
        'mean_states': states['mean'],
        'lower_states': states.get('lower', None),
        'upper_states': states.get('upper', None),
        'resolved_stats_mode': resolved_stats_mode,
    }


def run_episode_sequence(
    trainer,
    dataset,
    dataset_indices: list[int],
    feature_index: int,
    prepend_mode: str = 'last',
    stats_mode: str = 'as-config',
    max_steps: int | None = None,
    title: str | None = None,
    fps: int = 12,
    dpi: int = 120,
    interp_frames: int = 2,
    end_hold_seconds: float = 1.5,
    init_hold_seconds: float = 1.5,
    save_gif: bool = False,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if not dataset_indices:
        raise ValueError('dataset_indices must be non-empty')

    episodes_payload = []
    predictions = []
    for idx in dataset_indices:
        payload = _collect_episode_rollout(
            trainer=trainer,
            dataset=dataset,
            dataset_idx=idx,
            feature_index=feature_index,
            prepend_mode=prepend_mode,
            stats_mode=stats_mode,
            max_steps=max_steps,
        )
        episodes_payload.append(payload)
        predictions.append(payload['prediction'])

    animator = RolloutAnimator(
        fps=fps,
        dpi=dpi,
        interp_frames=interp_frames,
        end_hold_seconds=end_hold_seconds,
        init_hold_seconds=init_hold_seconds,
    )
    fig, anim = animator.create_episode_sequence_animation(
        episodes=episodes_payload,
        prepend_mode=prepend_mode,
        stats_mode=episodes_payload[0]['resolved_stats_mode'],
        title=title,
    )

    if save_gif:
        if output_path is None:
            raise ValueError('output_path must be provided when save_gif=True')
        animator.save_gif(anim, output_path)
        plt.close(fig)

    return {
        'figure': fig,
        'animation': anim,
        'episode_indices': dataset_indices,
        'predictions': predictions,
        'episodes': episodes_payload,
        'resolved_stats_mode': episodes_payload[0]['resolved_stats_mode'],
    }
