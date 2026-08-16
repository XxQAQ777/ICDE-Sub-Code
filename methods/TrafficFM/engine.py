# engine.py

import torch

import torch.nn as nn

import torch.optim as optim

from model import *

import util



class trainer():

    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit, model_module=None, train_objective='point', blocks=8, layers=3):

        # 如果提供了 model_module, 使用它; 否则使用默认的 model_cfg

        if model_module is None:

            from model_cfg import gwnet

        else:

            gwnet = model_module.gwnet



        self.model = gwnet(

            device, num_nodes, dropout,

            supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit,

            in_dim=in_dim, out_dim=seq_length,

            residual_channels=nhid, dilation_channels=nhid,

            skip_channels=nhid * 8, end_channels=nhid * 16,
            blocks=blocks, layers=layers,

        )

        self.model.to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)

        self.loss = util.masked_mae

        self.train_objective = train_objective

        self.scaler = scaler

        self.clip = 5



    def train(self, input, real_val):

        self.model.train()

        self.optimizer.zero_grad()

        input = nn.functional.pad(input, (1, 0, 0, 0))

        real = torch.unsqueeze(real_val, dim=1)  # [B, 1, N, T], raw scale

        if self.train_objective == 'model':
            # model-internal loss expects normalized target with shape [B, T, N, 1]
            y_norm = self.scaler.transform(real).transpose(1, 3).contiguous()
            loss = self.model(input, y=y_norm)
        elif self.train_objective == 'point':
            # Point-loss ablation: keep the default TrafficFM architecture,
            # but train against MAE using a short FM sampling path for feasibility.
            output = self.model(input, fm_steps=2)
            output = output.transpose(1, 3)
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real, 0.0)
        else:
            raise ValueError(f"Unknown train_objective: {self.train_objective}")

        loss.backward()

        if self.clip is not None:

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        # Lightweight train metrics. For FM objective this samples with fewer ODE steps.
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            output = self.model(input, fm_steps=5)
            output = output.transpose(1, 3)
            predict = self.scaler.inverse_transform(output)

            mape = util.masked_mape(predict, real, 0.0).item()
            rmse = util.masked_rmse(predict, real, 0.0).item()
            mae = util.masked_mae(predict, real, 0.0).item()
        if was_training:
            self.model.train()

        return loss.item(), mape, rmse, mae



    def eval(self, input, real_val):

        self.model.eval()

        with torch.no_grad():

            input = nn.functional.pad(input, (1, 0, 0, 0))

            output = self.model(input)

            output = output.transpose(1, 3)

            real = torch.unsqueeze(real_val, dim=1)

            predict = self.scaler.inverse_transform(output)

            loss = self.loss(predict, real, 0.0)

            mape = util.masked_mape(predict, real, 0.0).item()

            rmse = util.masked_rmse(predict, real, 0.0).item()

            mae = util.masked_mae(predict, real, 0.0).item()

        return loss.item(), mape, rmse, mae
