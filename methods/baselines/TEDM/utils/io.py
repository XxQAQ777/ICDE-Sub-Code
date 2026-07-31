from __future__ import annotations
import sys
import itertools
from pathlib import Path
import importlib
from omegaconf import OmegaConf
import yaml
import torch
import warnings
from time import perf_counter


class ResourceUsageTracker:
    """Tracks GPU memory and batch timing for training and evaluation with minimal intrusion.

    Usage:
        tracker = ResourceUsageTracker(device, results_folder)
        tracker.start_batch('train')
        ... run forward/backward ...
        tracker.end_batch('train')
        # After test evaluation
        tracker.write_csv()
    """

    def __init__(self, device, results_folder):
        self.device = device
        self.results_folder = Path(results_folder)
        self._use_cuda = torch.cuda.is_available() and 'cuda' in str(device)
        self.train_times = []
        self.train_mems = []
        self.test_times = []
        self.test_mems = []
        self._tic = None

    def _sync(self):
        if self._use_cuda:
            torch.cuda.synchronize(self.device)

    def start_batch(self, phase: str):
        if phase not in ('train', 'test'):
            return
        if self._use_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
        self._sync()
        self._tic = perf_counter()

    def end_batch(self, phase: str):
        if self._tic is None or phase not in ('train', 'test'):
            return
        self._sync()
        elapsed = perf_counter() - self._tic
        if self._use_cuda:
            peak_mem = torch.cuda.max_memory_allocated(
                self.device) / (1024 ** 2)
        else:
            peak_mem = 0.0
        if phase == 'train':
            self.train_times.append(elapsed)
            self.train_mems.append(peak_mem)
        else:
            self.test_times.append(elapsed)
            self.test_mems.append(peak_mem)
        self._tic = None

    def summary(self):
        import numpy as _np

        def avg(lst):
            return float(_np.mean(lst)) if lst else 0.0
        return {
            'avg_train_mem_mb': avg(self.train_mems),
            'avg_train_time_s': avg(self.train_times),
            'avg_test_mem_mb': avg(self.test_mems),
            'avg_test_time_s': avg(self.test_times)
        }

    def write_csv(self):
        stats = self.summary()
        csv_path = self.results_folder / 'resource_usage.csv'
        header = 'avg_train_mem_mb,avg_train_time_s,avg_test_mem_mb,avg_test_time_s\n'
        line = (
            f"{stats['avg_train_mem_mb']:.4f},"
            f"{stats['avg_train_time_s']:.6f},"
            f"{stats['avg_test_mem_mb']:.4f},"
            f"{stats['avg_test_time_s']:.6f}\n"
        )
        if not csv_path.exists():
            with open(csv_path, 'w') as f:
                f.write(header)
                f.write(line)
        else:
            with open(csv_path, 'a') as f:
                f.write(line)
        return stats


def load_config(exp: int, dataset: str, ablations: bool = False,
                config_path: str = 'ablations/ablations.yaml'):
    base_config = OmegaConf.load(f'config/{dataset}.yaml')

    if not ablations:
        return [''], [base_config]

    abl_yaml = OmegaConf.load(config_path)

    if exp not in abl_yaml.experiments:
        available = sorted(abl_yaml.experiments.keys())
        raise ValueError(
            f"Experiment {exp} not found in {config_path}. "
            f"Available IDs: {available}"
        )

    experiment = abl_yaml.experiments[exp]
    all_paths = [o.path for o in experiment.overrides]
    all_values = [list(o["values"]) for o in experiment.overrides]

    descs = []
    confs = []

    mode = experiment.get('mode', 'product')
    if mode == 'paired':
        combos = zip(*all_values)
    elif mode == 'product':
        combos = itertools.product(*all_values)
    else:
        raise ValueError(f"Unknown experiment mode '{mode}'. Use 'product' or 'paired'.")

    requires_denoiser = experiment.get('requires_denoiser')
    requires_denoisers = list(experiment.get('requires_denoisers', []) or [])
    if requires_denoiser:
        denoiser_list = [requires_denoiser]
    elif requires_denoisers:
        denoiser_list = requires_denoisers
    else:
        denoiser_list = [None]

    combos = list(combos)
    for denoiser in denoiser_list:
        for combo in combos:
            cfg = OmegaConf.merge(base_config, {})
            if denoiser:
                OmegaConf.update(cfg, 'model.denoiser.target', denoiser)
            for path, value in zip(all_paths, combo):
                OmegaConf.update(cfg, path, value)

            parts = []
            if denoiser:
                parts.append(denoiser.rsplit('.', 1)[-1])
            for path, value in zip(all_paths, combo):
                leaf = path.rsplit('.', 1)[-1]
                pv = value.replace('.', '_') if isinstance(value, str) else value
                parts.append(f"{leaf}_{pv}")
            descs.append('_abl_' + '_'.join(parts))
            confs.append(cfg)

    return descs, confs


_DIFFUSION_PRESETS = {
    'edm': {
        'model.noise_schedule.target': 'models.noise.EDM',
        'model.preconditioning.target': 'models.precond.EDM',
        'sampler.target': 'training.sampler.EDM',
        'dataloader.params.padding': 0,
    },
    'iddpm': {
        'model.noise_schedule.target': 'models.noise.iDDPM',
        'model.preconditioning.target': 'models.precond.iDDPM',
        'sampler.target': 'training.sampler.DDIM',
        'dataloader.params.padding': 0,
    },
}


def apply_diffusion_preset(config, diffusion: str) -> None:
    for path, value in _DIFFUSION_PRESETS.get(diffusion, {}).items():
        OmegaConf.update(config, path, value)


def count_total_experiments(config_path: str = 'ablations/ablations.yaml') -> int:
    conf = OmegaConf.load(config_path)
    total = 0
    for experiment in conf.experiments.values():
        if experiment.get('mode', 'product') == 'paired':
            run_count = len(experiment.overrides[0]['values'])
        else:
            run_count = 1
            for override in experiment.overrides:
                run_count *= len(override['values'])
        total += run_count
    return total


def list_datasets() -> list[str]:
    return [
        d.stem for d in (
            Path(__file__).parents[1] /
            'data').iterdir() if 'csv' in d.name]


def set_results_folder(
    config: OmegaConf,
    dataset: str,
    root: str,
    keep_last: bool,
    desc: str = '',
    ablations: bool = False
) -> Path:
    # Assert valid dataset
    dataset_from_config = Path(config.dataloader.params.path).stem
    if dataset != dataset_from_config:
        warnings.warn(
            "Dataset names from config and CLI don't match. Choosing from config.")
        dataset = dataset_from_config

    results_folder = Path(root)
    all_runs = list(sorted(results_folder.glob(f'{dataset + desc}*')))
    if not ablations:
        all_runs = [
            all_run for all_run in all_runs
            if 'abl' not in str(all_run).split('_')]
    else:
        all_runs = [all_run for all_run in all_runs
                    if 'abl' in str(all_run).split('_')]

    if len(all_runs) == 0:
        suffix = f'_{1:05d}'
    else:
        run_id = all_runs[-1].stem.split('_')
        shift = 0 if keep_last else 1
        next_run_id = int(run_id[-1]) + shift
        suffix = f'_{next_run_id:05d}'

    results_folder /= Path(dataset + desc + suffix)
    results_folder.mkdir(parents=True, exist_ok=True)

    return results_folder


def instantiate_from_config(config, *args, from_module=None):
    if config is None:
        return None
    if not hasattr(config, 'target'):
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config.target.rsplit(".", 1)
    if from_module is not None:
        cls = getattr(from_module, cls)
    else:
        cls = getattr(importlib.import_module(module, package=None), cls)
    config_params = config.params if hasattr(config, 'params') else {}
    return cls(*args, **config_params)


def class_from_string(class_name):
    module, cls = class_name.rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls


def get_model_parameters_info(model):
    parameters = {'overall': {'trainable': 0, 'non_trainable': 0, 'total': 0}}
    for child_name, child_module in model.named_children():
        parameters[child_name] = {'trainable': 0, 'non_trainable': 0}
        for pn, p in child_module.named_parameters():
            if p.requires_grad:
                parameters[child_name]['trainable'] += p.numel()
            else:
                parameters[child_name]['non_trainable'] += p.numel()
        parameters[child_name]['total'] = parameters[child_name]['trainable'] + \
            parameters[child_name]['non_trainable']

        parameters['overall']['trainable'] += parameters[child_name]['trainable']
        parameters['overall']['non_trainable'] += parameters[child_name]['non_trainable']
        parameters['overall']['total'] += parameters[child_name]['total']

    # format the numbers
    def format_number(num):
        K = 2**10
        M = 2**20
        G = 2**30
        if num > G:  # G
            uint = 'G'
            num = round(float(num) / G, 2)
        elif num > M:
            uint = 'M'
            num = round(float(num) / M, 2)
        elif num > K:
            uint = 'K'
            num = round(float(num) / K, 2)
        else:
            uint = ''

        return '{}{}'.format(num, uint)

    def format_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                format_dict(v)
            else:
                d[k] = format_number(v)

    format_dict(parameters)
    return parameters


def write_args(args, path):
    args_dict = dict((name, getattr(args, name))
                     for name in dir(args) if not name.startswith('_'))
    with open(path, 'a') as args_file:
        args_file.write('==> torch version: {}\n'.format(torch.__version__))
        args_file.write(
            '==> cudnn version: {}\n'.format(
                torch.backends.cudnn.version()))
        args_file.write('==> Cmd:\n')
        args_file.write(str(sys.argv))
        args_file.write('\n==> args:\n')
        for k, v in sorted(args_dict.items()):
            args_file.write('  %s: %s\n' % (str(k), str(v)))


def save_config_to_yaml(config, path):
    if not str(path).endswith('.yaml'):
        raise ValueError(f"Expected .yaml path, got: {path}")
    config = OmegaConf.to_container(config, resolve=True)
    with open(path, 'w') as f:
        f.write(yaml.dump(config))
