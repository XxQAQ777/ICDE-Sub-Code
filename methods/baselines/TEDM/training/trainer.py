from __future__ import annotations
from contextlib import contextmanager
import time
import gc
import torch
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from torch.optim import Optimizer
from ema_pytorch import EMA
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from models.loss import Model
from . import sampler
from .metrics import mse_and_mae, crps, qice
import utils
from utils.io import ResourceUsageTracker


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config):
        super().__init__()
        # Initialize variables
        self.trainer = config.trainer
        self.loaded = False
        self.config = config
        self.milestone = 0
        self.step = 0
        self.results_folder = config.cur_dir
        self.trainer.logger.params.update({'save_dir': self.results_folder})
        self.logger = utils.instantiate_from_config(self.trainer.logger)
        self.logger.set_config(self.config)
        # Check dataloader is compatible with model resolution
        assert config.dataloader.params.window == config.model.denoiser.params.seq_len, \
            'Model resolution `seq_len` must equal dataloader `window`'

        # Create dataloader
        config.dataloader.params.update({
            'period': 'train',
            'horizon': config.sampler.params.horizon
        })
        dataloader = torch.utils.data.DataLoader(
            utils.instantiate_from_config(config.dataloader),
            batch_size=config.dataloader.batch_size,
            shuffle=config.dataloader.shuffle,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            sampler=None,
            drop_last=config.dataloader.shuffle
        )
        self.dl = cycle(dataloader)

        # Build model
        config.model.denoiser.params.update(
            {'feat_size': dataloader.dataset.feat_size})
        self.model: nn.Module = Model(config)
        self.logger.log_info(
            str(utils.get_model_parameters_info(self.model.base)))

        # Build EMA model
        self.ema = EMA(
            self.model.base,
            beta=config.trainer.ema.decay,
            update_every=config.trainer.ema.update_interval
        ).to(self.model.device)

        # Build sampler from EMA model
        config.sampler.params.update({
            'seq_len': config.dataloader.params.window,
            'past_k': config.dataloader.params.past_k,
            'padding': config.dataloader.params.padding
        })
        self.eval_noise_sch = self.model.noise_sch.eval()

        self.sampler = utils.instantiate_from_config(
            config.sampler,
            self.ema.ema_model, self.eval_noise_sch,
            from_module=sampler
        )
        # Build optimizer
        self.opt: Optimizer = utils.instantiate_from_config(
            self.trainer.optimizer,
            filter(lambda p: p.requires_grad, self.model.base.parameters())
        )
        # Fix learning rate schedule
        self.sch = utils.instantiate_from_config(
            self.trainer.scheduler,
            self.opt
        )

        # Resource usage tracker
        self.resource_tracker = ResourceUsageTracker(
            self.model.device, self.results_folder)

    def save(self, milestone):
        self.logger.log_info(
            f"Save current model to {str(self.results_folder / f'checkpoint-{milestone}.pt')}"
        )
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(
            data, str(
                self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone):
        self.logger.log_info(
            f"Resume from {str(self.results_folder / f'checkpoint-{milestone}.pt')}")
        device = self.model.device
        data = torch.load(
            str(self.results_folder / f'checkpoint-{milestone}.pt'),
            map_location=device)
        try:
            missing, unexpected = self.model.load_state_dict(
                data['model'], strict=False)
            if missing or unexpected:
                self.logger.log_info(
                    f"[LOAD] Non-strict load: missing={missing} unexpected={unexpected}"
                )
        except RuntimeError as e:
            self.logger.log_info(
                f"[LOAD] Strict load failed ({e}); continuing without weights for missing layers.")
            self.model.load_state_dict(data['model'], strict=False)
        self.step = data.get('step', 0)
        if 'opt' in data:
            try:
                self.opt.load_state_dict(data['opt'])
            except Exception as e:
                self.logger.log_info(f"[LOAD] Skipping optimizer load: {e}")
        if 'ema' in data:
            try:
                self.ema.load_state_dict(data['ema'])
            except Exception as e:
                self.logger.log_info(f"[LOAD] Skipping EMA load: {e}")
        self.milestone = milestone

    def train(self, n_steps=None):
        """Train the model.

        Args:
            n_steps: If provided, train for this many steps and return.
                     Can be called repeatedly to train in chunks (self.step
                     persists across calls). If None, train until max_epochs
                     (original behaviour).
        """
        device = self.model.device
        target = n_steps if n_steps is not None else (self.trainer.max_epochs - self.step)

        if self.step == 0:
            self._train_tic = time.time()
            self.logger.log_info('Start training ...')
        self.model.train()

        steps_done = 0
        with tqdm(initial=0, total=target) as pbar:
            while steps_done < target and self.step < self.trainer.max_epochs:
                total_loss = torch.tensor(0.0, device=device)
                self.resource_tracker.start_batch('train')
                for _ in range(self.trainer.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    loss = self.model(data)
                    loss = loss / self.trainer.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.detach()
                total_loss = total_loss.item()

                pbar.set_description(f'Train loss: {total_loss:.6f}')

                clip_grad_norm_(self.model.base.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                steps_done += 1
                self.ema.update()

                self.resource_tracker.end_batch('train')

                with torch.no_grad():
                    if self.step % self.logger.log_freq == 0:

                        # save model
                        self.milestone += 1
                        self.save(self.milestone)

                        # log training loss
                        self.logger.log_info(
                            f'Step {self.step}, Train loss: {total_loss:.6f}')
                        self.logger.plot(
                            'loss', 'train', 'Train Loss', self.step, total_loss)

                        # Track model training progress with random training
                        # sample predictions
                        try:
                            self.logger.add_images(
                                *self.model.get_visuals(),
                                period='train', global_step=self.step)
                        except ConnectionError:
                            self.logger._start_server()

                        # evaluate metrics on validation set
                        if self.trainer.validation_check:
                            self.evaluate_metrics(period='val', step=self.step)

                pbar.update(1)

        if self.step >= self.trainer.max_epochs:
            self.logger.log_info(f'Training done, time: {time.time() - self._train_tic:.2f}')
            self.loaded = True

    def evaluate_metrics(self, period, step=None):
        with self.use_eval_noise_sch():
            samples, reals = self.evaluate(period=period)
        mse, mae = mse_and_mae(reals, samples)
        raw_samples, raw_reals = self._inverse_eval_arrays(samples, reals)
        if self.sampler.deterministic:
            CRPS, QICE = None, None
        else:
            CRPS = crps(reals, samples)
            QICE = qice(reals, samples)
        prob_metrics = f", CRPS: {CRPS}, QICE: {QICE}" if CRPS is not None else ""
        self.logger.log_info(
            f"{period.capitalize()} data: MSE: {mse:.4f}, MAE: {mae:.4f}{prob_metrics}")
        if period == 'val':
            self.logger.plot('MSE', 'Val', 'Val mse', step, mse)
        if period == 'test':
            print(f"Test dataset MSE: {mse}, MAE: {mae}{prob_metrics}")
            self._print_raw_horizon_metrics(raw_samples, raw_reals)
            stats = self.resource_tracker.write_csv()
            self.logger.log_info(
                "Resource usage saved: "
                f"train_mem={stats['avg_train_mem_mb']:.2f}MB "
                f"train_time={stats['avg_train_time_s']:.4f}s "
                f"test_mem={stats['avg_test_mem_mb']:.2f}MB "
                f"test_time={stats['avg_test_time_s']:.4f}s"
            )
            self.logger.close()
        return mse

    def evaluate(self, period='test'):
        # Create dataloader
        self.config.dataloader.params.update({
            'period': period,
            'horizon': self.config.sampler.params.horizon
        })
        prob_forecast = not self.sampler.deterministic
        batch_size = self.sampler.batch_size if prob_forecast else self.config.dataloader.batch_size
        eval_dataset = utils.instantiate_from_config(self.config.dataloader)
        self._last_eval_scaler = getattr(eval_dataset, 'scaler', None)
        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=None,
            drop_last=False
        )
        seq_len = self.config.model.denoiser.params.seq_len
        num_feat = self.config.model.denoiser.params.feat_size

        self.logger.log_info('Begin to evaluate ...')
        if isinstance(self.sampler, sampler.TEDM) and prob_forecast:
            samples = np.empty([0, seq_len, num_feat, self.sampler.n_tracks])
        else:
            samples = np.empty([0, seq_len, num_feat])
        reals = np.empty([0, seq_len, num_feat])

        if not self.loaded:
            milestones = sorted([
                int(ck.stem.split('-')[-1])
                for ck in Path(self.config.cur_dir).iterdir()
                if ck.name.endswith('pt')
            ])
            self.load(milestones[-1])
            self.loaded = True

        # Set model eval mode
        tic = time.time()
        self.ema.ema_model.eval()
        with tqdm(total=len(dataloader)) as evalbar:
            for idx, batch in enumerate(dataloader):
                if period == 'test':
                    self.resource_tracker.start_batch('test')
                x = batch.to(self.model.device)
                sample = self.sampler.rollout(x[:, :-seq_len, :])
                samples = np.row_stack(
                    [samples, sample.detach().cpu().numpy()])
                reals = np.row_stack(
                    [reals, x[:, -seq_len:, :].detach().cpu().numpy()])

                if period == 'val':
                    evalbar.set_description("Validation")
                else:
                    evalbar.set_description("Testing")

                real = x[:, -seq_len:, :].cpu().numpy()
                pred = sample.cpu().numpy()
                if period == 'val' and self.trainer.validation_viz:
                    self.logger.add_image(real, pred, period='val')
                elif period == 'test':
                    self.logger.add_image(real, pred, period='test')
                if period == 'test':
                    self.resource_tracker.end_batch('test')

                evalbar.update(1)

        self.logger.log_info(f'Sampling done, time: {time.time() - tic:.2f}')
        if period == 'val':
            self.ema.ema_model.train()

        return samples, reals


    def _inverse_eval_arrays(self, samples, reals):
        pred = samples
        true = reals

        if pred.ndim == true.ndim + 1:
            pred = pred.mean(axis=-1)

        pred = np.asarray(pred, dtype=np.float32)
        true = np.asarray(true, dtype=np.float32)

        scaler = getattr(self, '_last_eval_scaler', None)
        if scaler is None:
            return pred, true

        feat = pred.shape[-1]
        pred_raw = scaler.inverse_transform(pred.reshape(-1, feat)).reshape(pred.shape)
        true_raw = scaler.inverse_transform(true.reshape(-1, feat)).reshape(true.shape)

        return pred_raw.astype(np.float32), true_raw.astype(np.float32)

    @staticmethod
    def _masked_metrics(pred, true):
        pred = np.asarray(pred, dtype=np.float64)
        true = np.asarray(true, dtype=np.float64)
        mask = np.abs(true) > 1e-3

        if not np.any(mask):
            return float('nan'), float('nan'), float('nan')

        err = pred - true
        mae = np.mean(np.abs(err)[mask])
        rmse = np.sqrt(np.mean((err ** 2)[mask]))
        mape = np.mean(np.abs(err[mask] / true[mask]))

        return mae, mape, rmse

    def _print_raw_horizon_metrics(self, pred_raw, true_raw):
        total_horizon = pred_raw.shape[1]
        print("Raw-scale TEDM metrics after inverse StandardScaler:")

        for h in (36, 72, 144):
            if h <= total_horizon:
                mae, mape, rmse = self._masked_metrics(
                    pred_raw[:, h - 1, :],
                    true_raw[:, h - 1, :]
                )
                print(
                    f"Evaluate best model on test data for horizon {h}, "
                    f"Test MAE: {mae:.4f}, Test MAPE: {mape:.4f}, Test RMSE: {rmse:.4f}"
                )

        mae, mape, rmse = self._masked_metrics(pred_raw, true_raw)
        print(
            f"On average over {total_horizon} horizons, "
            f"Test MAE: {mae:.4f}, Test MAPE: {mape:.4f}, Test RMSE: {rmse:.4f}"
        )

    def sample(
            self,
            indices,
            period='test',
            save_as_numpy=False,
            save_prefix='samples',
            results_subdir=None):
        """Sample predictions for specific window indices from the dataset.

        Parameters
        ----------
        indices : list[int] | int
            Index or list of indices of windows in the specified period dataset.
        period : str
            One of 'test' or 'val'.
        save_as_numpy : bool
            If True, saves input, target, prediction arrays to disk.
        save_prefix : str
            Base filename prefix when saving numpy arrays.
        results_subdir : str | None
            Optional subdirectory under results folder to store numpy outputs.

        Returns
        -------
        context_windows : np.ndarray shape (B, context_len, F)
        target_windows  : np.ndarray shape (B, pred_len, F)
        predictions     : np.ndarray
            Shape (B, pred_len, F) for deterministic; (B, pred_len, F, S) or (B, pred_len, F, K) for probabilistic.
        """
        assert period in [
            'test', 'val'], "Sampling only supported for test or val periods."
        if isinstance(indices, int):
            indices = [indices]

        # Rebuild dataloader for the given period to access the underlying
        # dataset deterministically
        self.config.dataloader.params.update({
            'period': period,
            'horizon': self.config.sampler.params.horizon
        })
        dataset = utils.instantiate_from_config(self.config.dataloader)
        seq_len = self.config.model.denoiser.params.seq_len
        past_k = self.config.dataloader.params.past_k
        padding = self.config.dataloader.params.padding
        enlarge_for_cond = past_k + padding
        # total conditioning portion inside each sample window
        context_len = seq_len + enlarge_for_cond

        device = self.model.device
        if not self.loaded:
            milestones = sorted([
                int(ck.stem.split('-')[-1])
                for ck in Path(self.config.cur_dir).iterdir()
                if ck.name.endswith('pt')
            ])
            self.load(milestones[-1])

        self.ema.ema_model.eval()
        contexts = []
        targets = []
        preds = []

        for internal_idx in indices:
            internal_x = dataset[internal_idx]
            x_np = internal_x.numpy()
            context_window = x_np[:context_len, :]
            target_window = x_np[context_len:context_len + seq_len, :]

            # Build batch of size 1 to feed the sampler rollout
            tensor_x = internal_x.unsqueeze(0).to(device)
            with torch.no_grad():
                # Pass only the conditioning portion (excluding target horizon)
                # to rollout
                cond = tensor_x[:, :-seq_len, :]
                # shape depends on deterministic / probabilistic
                sample = self.sampler.rollout(cond)
            sample_np = sample.detach().cpu().numpy()

            contexts.append(context_window[None, ...])
            targets.append(target_window[None, ...])
            preds.append(sample_np)

        context_arr = np.concatenate(contexts, axis=0)
        target_arr = np.concatenate(targets, axis=0)
        pred_arr = np.concatenate(preds, axis=0)

        if save_as_numpy:
            out_dir = self.results_folder if results_subdir is None else (
                self.results_folder / results_subdir)
            out_dir.mkdir(parents=True, exist_ok=True)
            bundle_path = out_dir / f'{save_prefix}.npz'
            save_kwargs = dict(
                context=context_arr,
                target=target_arr,
                pred=pred_arr,
                internal_indices=np.array(
                    indices,
                    dtype=np.int64))
            np.savez_compressed(bundle_path, **save_kwargs)
            msg = f"Saved sample bundle: {bundle_path.name} (keys: context,target,pred) in {out_dir}"
            if getattr(self.logger, 'enabled', False):
                self.logger.log_info(msg)
            else:
                print(msg)

        # Print resource usage summary (reuse resource tracker summary if
        # available)
        if hasattr(self, 'resource_tracker'):
            stats = self.resource_tracker.summary()
            print(
                'Resource usage (aggregated so far) -> '
                f"avg_train_mem_mb={stats['avg_train_mem_mb']:.2f}, "
                f"avg_train_time_s={stats['avg_train_time_s']:.4f}, "
                f"avg_test_mem_mb={stats['avg_test_mem_mb']:.2f}, "
                f"avg_test_time_s={stats['avg_test_time_s']:.4f}"
            )

        return context_arr, target_arr, pred_arr

    def release_resources(self):
        torch.cuda.empty_cache()
        gc.collect()

    @contextmanager
    def use_eval_noise_sch(self):
        """Temporarily swap in the eval noise scheduler, then restore the training noise scheduler."""
        train_sch = self.ema.ema_model.noise_sch
        try:
            self.ema.ema_model.noise_sch = self.eval_noise_sch
            yield
        finally:
            self.ema.ema_model.noise_sch = train_sch
