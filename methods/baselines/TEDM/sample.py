"""Sample from a trained TEDM model.

This script loads a trained model checkpoint and samples from it for given
window indices.

Usage Examples
--------------
Basic sampling (internal indices):
    python sample.py --dataset ettm2 --indices 0 --period test --save-as-numpy

Random plot mode
----------------
``utils/plot.py`` invokes this script automatically when
``--sample-indices random:N`` is passed.  Indices are chosen to be
non-overlapping across the requested split and the results are saved as
``samples_random.npz`` inside the run folder before plotting.

See Also
--------
utils.plot : Module for plotting sampled windows.
"""
from pathlib import Path
from types import SimpleNamespace
import click
from training.trainer import Trainer
import utils

utils.set_seed(2025)


@click.command()
@click.option('--dataset',
              help='name of dataset. Same as file stem',
              metavar='STR',
              type=click.Choice(utils.list_datasets()),
              required=True,
              default='etth1')
@click.option('--outdir', help='path for saving results',
              metavar='DIR', type=str, default='results')
@click.option('--keep-last',
              help='do not create new folder for current run',
              metavar='BOOL',
              type=click.BOOL,
              default=True)
@click.option('--gpu_id', help='gpu device', metavar='INT',
              type=str, required=False, default=0)
@click.option('--indices',
              help='Comma-separated window indices (e.g., 0,5,10)',
              metavar='STR',
              type=str,
              default='0')
@click.option('--period', help='Dataset period to sample from',
              type=click.Choice(['test', 'val']), default='test')
@click.option('--save-as-numpy', is_flag=True, default=False,
              help='Save outputs as numpy arrays (off by default)')
@click.option('--save-prefix',
              help='Filename prefix for saved numpy arrays',
              type=str,
              default='samples')
@click.option('--results-subdir',
              help='Optional subdirectory inside run folder for saved arrays',
              type=str,
              default='')
@click.option('--diffusion',
              help='Diffusion backend: tedm (default), edm, or iddpm',
              type=click.Choice(['tedm', 'edm', 'iddpm']),
              default='tedm')
def main(**kwargs):
    opts = SimpleNamespace(**kwargs)
    indices = [int(i.strip())
               for i in opts.indices.split(',') if i.strip() != '']

    # Load single config (no ablations)
    _, configs = utils.load_config(
        exp=1, dataset=opts.dataset, ablations=False)
    config = configs[0]
    cur_dir = utils.set_results_folder(
        config,
        opts.dataset,
        opts.outdir,
        opts.keep_last,
        desc='',
        ablations=False)
    config.update({'cur_dir': cur_dir, 'gpu_id': opts.gpu_id})
    utils.apply_diffusion_preset(config, opts.diffusion) # apply edm or iddpm config

    print(f"[INFO] Run folder: {cur_dir}")
    print(
        f"[INFO] Indices: {indices} Period: {opts.period} "
        f"Save numpy: {opts.save_as_numpy} "
        f"Subdir: '{opts.results_subdir}' Prefix: {opts.save_prefix}")
    trainer = Trainer(config)
    ckpts = list(Path(cur_dir).glob('checkpoint-*.pt'))
    if len(ckpts) == 0:
        print('No checkpoints found. Training model first...')
        trainer.train()
    context_arr, target_arr, pred_arr = trainer.sample(
        indices=indices, period=opts.period, save_as_numpy=opts.save_as_numpy,
        save_prefix=opts.save_prefix, results_subdir=opts.results_subdir or
        None)
    # Ensure config.yaml is written for plotting even if logger wasn't closed
    # during sampling
    from utils.io import save_config_to_yaml
    cfg_path = Path(cur_dir) / 'config.yaml'
    if not cfg_path.exists():
        save_config_to_yaml(config, cfg_path)
        print(f"[INFO] Wrote config.yaml to {cfg_path}")
    print(
        f"Sampling complete. Shapes -> context: {context_arr.shape}, "
        f"target: {target_arr.shape}, pred: {pred_arr.shape}")


if __name__ == '__main__':
    main()
