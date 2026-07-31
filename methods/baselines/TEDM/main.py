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
# ablations
@click.option('--ablations', help='Ablations or not',
              metavar='BOOL', type=click.BOOL, default=False)
@click.option('--exp', help='Experiment id for ablation',
              metavar='INT', type=int, default=1)
@click.option('--diffusion',
              help='Diffusion backend: tedm (default), edm, or iddpm',
              type=click.Choice(['tedm', 'edm', 'iddpm']),
              default='tedm')
@click.option('--epochs', help='Override trainer.max_epochs (e.g. 10 for a quick smoke test)',
              metavar='INT', type=int, default=None)
def main(**kwargs):
    opts = SimpleNamespace(**kwargs)
    descs, configs = utils.load_config(
        exp=opts.exp, dataset=opts.dataset, ablations=opts.ablations)

    for desc, config in zip(descs, configs):
        cur_dir = utils.set_results_folder(
            config,
            opts.dataset,
            opts.outdir,
            opts.keep_last,
            desc=desc,
            ablations=opts.ablations)
        config.update({'cur_dir': cur_dir, 'gpu_id': opts.gpu_id})
        utils.apply_diffusion_preset(config, opts.diffusion) # apply edm or iddpm config
        if opts.epochs is not None:
            from omegaconf import OmegaConf
            OmegaConf.update(config, 'trainer.max_epochs', opts.epochs)

        trainer = Trainer(config)
        trainer.train()
        trainer.evaluate_metrics('test')
        trainer.release_resources()


if __name__ == '__main__':
    main()
