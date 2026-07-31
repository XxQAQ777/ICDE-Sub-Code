import os
import time
import torch
import numpy as np
import utils
import re
import subprocess
import socket


class BaseLogger(object):
    """A base logger that handles writing logs to files and TensorBoard."""

    def __init__(
            self,
            log_freq=100,
            enable=False,
            save_dir=None,
            args=None,
            print_to_console=False,
            **kwargs):
        # Control logging via flag
        self.enabled = enable
        self.log_freq = log_freq
        self.print_to_console = print_to_console
        self.config = None

        if self.enabled:
            self.save_dir = save_dir or os.getcwd()
            os.makedirs(self.save_dir, exist_ok=True)

            # Save the args and config
            self.config_dir = os.path.join(self.save_dir, 'configs')
            os.makedirs(self.config_dir, exist_ok=True)
            file_name = os.path.join(self.config_dir, 'args.txt')
            utils.write_args(args, file_name)

            # Log directory
            log_dir = os.path.join(self.save_dir, 'logs')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # Text writer
            self.text_writer = open(
                os.path.join(
                    log_dir,
                    'log.txt'),
                'a')  # 'w')
            if args and getattr(args, 'tensorboard', False):
                self.log_info('using tensorboard')
                self.tb_writer = torch.utils.tensorboard.SummaryWriter(
                    log_dir=log_dir)
            else:
                self.tb_writer = None
        else:
            self.text_writer = None
            self.tb_writer = None

    def plot(self, var_name, split_name, title_name, x, y):
        pass

    def set_config(self, config):
        self.config = config

    def log_info(self, message):
        """Log an informational message to console and file."""
        if not self.enabled:
            return
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        entry = f"{timestamp}: {message}" + os.linesep
        if self.print_to_console:
            print(entry, end='')
        self.text_writer.write(entry)
        self.text_writer.flush()

    def add_image(self, *_, **kwargs):
        """Log an image via TensorBoard if enabled."""
        if self.enabled and self.tb_writer:
            self.tb_writer.add_image(**kwargs)

    def add_images(self, x, xn, D_xn, period="train", global_step=None):
        pass

    def close(self):
        if self.enabled:
            utils.save_config_to_yaml(
                config=self.config, path=os.path.join(
                    self.save_dir, 'config.yaml'))
            if self.text_writer:
                self.text_writer.close()
            if self.tb_writer:
                self.tb_writer.close()


class VisdomLogger(BaseLogger):
    """A logger that extends BaseLogger with Visdom plotting."""

    def __init__(
            self,
            vis_freq=50,
            log_freq=100,
            server='localhost',
            port=8097,
            verbose=False,
            save_dir=None,
            env_name='TEDM2',
            enable=False,
            print_to_console=False,
            args=None):
        global visdom
        import visdom

        # Initialize BaseLogger
        super().__init__(save_dir=save_dir, log_freq=log_freq, enable=enable,
                         args=args, print_to_console=print_to_console)

        # self.env = f"{env_name}_{str(save_dir).split('/')[-1]}"
        self.env = 'main'
        self.vis_freq = vis_freq
        self.server = server
        self.port = port
        self.plots = {}

        try:
            if self.enabled:
                self.vis = visdom.Visdom(server=server, port=port)
                if not self.vis.check_connection():
                    self._start_server()

                self.vis.close(env=self.env)
        except Exception as e:
            self.log_info(f"Visdom connection failed: {e}")

    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def _start_server(self):
        """Start Visdom server if it's not already running."""
        if not self.is_port_in_use(self.port):
            print(f"Starting Visdom server on port {self.port}...")
            subprocess.Popen(
                ['python', '-m', 'visdom.server', '-p', str(self.port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # Give the server a moment to start
            time.sleep(3)
            print("Visdom server started.")
        else:
            print(f"Visdom server is already running on port {self.port}.")

    def log_info(self, message):
        super().log_info(message)

    def plot(self, var_name, split_name, title_name, x, y):
        if not self.enabled:
            return

        X = np.array([x])
        Y = np.array([y])
        if var_name not in self.plots:
            opts = dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            )
            self.plots[var_name] = self.vis.line(
                X=X, Y=Y, opts=opts, env=self.env)
        else:
            self.vis.line(
                X=X,
                Y=Y,
                win=self.plots[var_name],
                name=split_name,
                update='append',
                env=self.env)

    def add_image(self, real, pred, period='test'):
        """Plot a single real vs. predicted time-series sample in Visdom."""
        if not self.enabled:
            return
        b, s, f = pred.shape[:-1] if pred.ndim > 3 else pred.shape
        # Use a fixed example so validation/test plots are comparable over time.
        rb = 0
        rf = 0
        t = np.arange(s)
        if pred.ndim > 3:
            pred_many_tracks = np.quantile(
                pred[rb, :, rf, :],
                q=[0.025, 0.5, 0.975],
                axis=-1,
            ).T
        Y = np.column_stack([
            real[rb, :, rf],
            pred[rb, :, rf] if pred.ndim == 3 else pred_many_tracks
        ])

        if not hasattr(self, 'add_image_win') and period == 'val':
            pred_legend = ['pred'] if pred.ndim == 3 else [
                'q02.5', 'q50', 'q97.5']
            self.add_image_win = self.vis.line(
                Y=Y,
                X=t,
                opts=dict(
                    legend=['real'] + pred_legend,
                    title=f"{period.capitalize()} Samples",
                    xlabel='Time',
                    ylabel='Value'
                ),
                env=self.env
            )
        elif period == 'val':
            self.vis.line(
                Y=Y,
                X=t,
                win=self.add_image_win if period == 'val' else None,
                update='replace',
                env=self.env
            )
        if period == 'test':
            pred_legend = ['pred'] if pred.ndim == 3 else [
                'q02.5', 'q50', 'q97.5']
            self.add_image_win_test = self.vis.line(
                Y=Y,
                X=t,
                opts=dict(
                    legend=['real'] + pred_legend,
                    title=f"{period.capitalize()} Samples",
                    xlabel='Time',
                    ylabel='Value'
                ),
                env=self.env
            )

    def add_images(self, x, xn, D_xn, global_step=None, period="train"):
        """Plot original, noisy, and denoised time-series samples together."""
        if not self.enabled:
            return
        x = x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)
        xn = xn.cpu().numpy() if hasattr(xn, 'cpu') else np.array(xn)
        D_xn = D_xn.cpu().numpy() if hasattr(D_xn, 'cpu') else np.array(D_xn)

        b, f, s = x.shape
        rb = np.random.randint(0, b)
        rf = np.random.randint(0, f)
        t = np.arange(s)
        Y = np.column_stack([x[rb, rf], xn[rb, rf], D_xn[rb, rf]])

        if not hasattr(self, 'win'):
            self.win = self.vis.line(
                X=t,
                Y=Y,
                opts=dict(
                    legend=['x', 'noisy x', 'denoised x'],
                    title=f'{period.capitalize()} data samples',
                    xlabel='Time',
                    ylabel='Value',
                    linecolor=np.array([
                        [0, 0, 0],
                        [255, 0, 255],
                        [0, 255, 0]
                    ])
                ),
                env=self.env
            )
        else:
            self.vis.line(
                Y=Y,
                X=t,
                win=self.win,
                update='replace',
                env=self.env
            )

    def close(self):
        super().close()


class MLFlowLogger(BaseLogger):
    """A logger that extends BaseLogger with MLFlow."""

    def __init__(
            self,
            vis_freq=50,
            log_freq=100,
            server='localhost',
            port=8097,
            verbose=False,
            save_dir=None,
            env_name='TEDM2',
            enable=False,
            print_to_console=False,
            args=None):
        global mlflow
        import mlflow

        super().__init__(log_freq=log_freq, enable=enable, save_dir=save_dir, args=args, print_to_console=print_to_console)

        if not enable:
            return

        self.experiment_name = env_name
        run_name = str(save_dir).split('/')[-1]

        try:
            if self.enabled:
                self._start_server(run_name)
        except Exception as e:
            self.log_info(f"MLflow connection failed: {e}")

    def _start_server(self, run_name):
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=str(run_name))

    def log_info(self, message):
        if "test" in message.lower():
            mse = re.search(r'MSE:\s([\d\.]+)', message)
            mae = re.search(r'MAE:\s([\d\.]+)', message)

            if mse and mae:
                mse_value = float(mse.group(1))
                mae_value = float(mae.group(1))
                mlflow.log_metric("test_mse", mse_value)
                mlflow.log_metric("test_mae", mae_value)

        super().log_info(message)

    def plot(self, var_name, split_name, title_name, x, y):
        if not self.enabled:
            return
        mlflow.log_metric(title_name, y, step=x)

    def add_image(self, real, pred, period=None):
        pass

    def add_images(self, x, xn, D_xn, period=None, global_step=None):
        pass

    def close(self):
        if hasattr(self, 'run'):
            mlflow.end_run()
        super().close()
