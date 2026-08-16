# train.py

import os
import csv

import time

import argparse

import numpy as np

import random

import torch

import torch.nn as nn

import torch.distributed as dist

import torch.multiprocessing as mp



import util

from engine import trainer

import importlib



parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='default', help='model name (e.g., default, ablation1, ablation2)')
parser.add_argument('--train_objective', type=str, default='point', choices=['point', 'model'], help='point: external masked MAE; model: use model(input, y=...) internal objective')

parser.add_argument('--devices', type=str, default='0', help='comma-separated GPU ids, e.g. "0,1"')

parser.add_argument('--data', type=str, default='data/PEMS-BAY-144-3feat-row', help='data path')
parser.add_argument('--unified_data', type=str, default=None,
                    help='STD-MAE dataset directory; uses its fixed 144-to-144 index dynamically')

parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx_bay.pkl', help='adj data path')

parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')

parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')

parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')

parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')

parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')

parser.add_argument('--seq_length', type=int, default=144, help='input sequence length')

parser.add_argument('--nhid', type=int, default=16, help='')
parser.add_argument('--blocks', type=int, default=8, help='WaveNet temporal blocks')
parser.add_argument('--layers', type=int, default=3, help='dilated layers per temporal block')

parser.add_argument('--in_dim', type=int, default=3, help='inputs dimension')

parser.add_argument('--num_nodes', type=int, default=325, help='number of nodes')

parser.add_argument('--batch_size', type=int, default=4, help='global batch size (must be divisible by number of GPUs)')

parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')

parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay rate')

parser.add_argument('--epochs', type=int, default=10, help='')
parser.add_argument('--patience', type=int, default=0,
                    help='early-stop after this many non-improving validation epochs; 0 disables it')
parser.add_argument('--max_train_batches', type=int, default=0,
                    help='optional per-epoch cap for a fast diagnostic run; 0 uses every fixed-split batch')
parser.add_argument('--max_validation_batches', type=int, default=0,
                    help='optional validation cap for a fast diagnostic run; 0 uses every fixed-split batch')

parser.add_argument('--print_every', type=int, default=5000, help='')

parser.add_argument('--seed', type=int, default=99, help='random seed')

parser.add_argument('--save', type=str, default='./gara_0106/pems_adj', help='save path')

parser.add_argument('--expid', type=int, default=1, help='experiment id')

parser.add_argument('--dist_port', type=str, default='12356', help='tcp port for DDP init, e.g., 12355')
parser.add_argument('--skip_final_test', action='store_true',
                    help='avoid the legacy full-tensor test pass; use the streaming probabilistic evaluator instead')

args = parser.parse_args()





def parse_device_ids(devices_str):

    try:

        return [int(d.strip()) for d in devices_str.split(',') if d.strip() != '']

    except Exception as e:

        print(f'Failed to parse --devices "{devices_str}", defaulting to [0]. Error: {e}')

        return [0]





def set_device_for_rank(rank, device_ids):

    use_cuda = torch.cuda.is_available()

    if use_cuda:

        local_gpu_id = device_ids[rank]

        torch.cuda.set_device(local_gpu_id)

        device = torch.device(f'cuda:{local_gpu_id}')

        torch.backends.cudnn.benchmark = True

        return device, local_gpu_id

    else:

        print('CUDA not available, using CPU.')

        return torch.device('cpu'), None





def save_model(engine, path):

    model_to_save = engine.model.module if hasattr(engine.model, 'module') else engine.model

    torch.save(model_to_save.state_dict(), path)





def load_model(engine, path, device):

    state = torch.load(path, map_location=device)

    model_to_load = engine.model.module if hasattr(engine.model, 'module') else engine.model

    model_to_load.load_state_dict(state)





def setup_ddp(rank, world_size, port):

    backend = 'nccl' if torch.cuda.is_available() else 'gloo'

    dist.init_process_group(

        backend=backend,

        init_method=f'tcp://127.0.0.1:{port}',

        world_size=world_size,

        rank=rank

    )





def ddp_avg(value, device, world_size):

    if dist.is_initialized() and world_size > 1:

        t = torch.tensor(value, device=device, dtype=torch.float32)

        dist.all_reduce(t, op=dist.ReduceOp.SUM)

        t = t / world_size

        return t.item()

    else:

        return float(value)





def run(rank, device_ids):

    use_cuda = torch.cuda.is_available()

    world_size = len(device_ids) if use_cuda else 1

    is_distributed = world_size > 1



    # 动态导入模型模块

    try:

        model_module = importlib.import_module(f'models.{args.model}')

        if rank == 0:

            print(f'Successfully loaded model: models.{args.model}')

    except ImportError:

        if rank == 0:

            print(f'Model models.{args.model} not found, falling back to default model_cfg')

        import model_cfg as model_module



    # 固定随机种子

    torch.manual_seed(args.seed)

    np.random.seed(args.seed)

    random.seed(args.seed)



    # 设备设置

    device, local_gpu_id = set_device_for_rank(rank, device_ids)



    # DDP 初始化

    if is_distributed:

        setup_ddp(rank, world_size, args.dist_port)

        # 强制要求 batch_size 可整除

        if args.batch_size % world_size != 0:

            if rank == 0:

                print(f'Error: --batch_size ({args.batch_size}) must be divisible by number of GPUs ({world_size}).')

            dist.barrier()

            raise ValueError('batch_size must be divisible by world_size')



    # 加载数据和图

    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)

    dataloader = (util.load_unified_dataset(args.unified_data, args.batch_size, args.batch_size, args.batch_size)
                  if args.unified_data else util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size))

    scaler = dataloader['scaler']



    # 支撑矩阵

    supports = model_module.make_supports(adj_mx, device)

    if rank == 0:

        print("supports:", [s.shape for s in supports], supports[0].dtype, supports[0].device)

        print(args)



    adjinit = None if args.randomadj else supports[0]

    if args.aptonly:

        supports = None



    # 构建引擎

    engine_inst = trainer(

        scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,

        args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,

        adjinit, model_module=model_module, train_objective=args.train_objective,
        blocks=args.blocks, layers=args.layers,

    )



    # DDP 包装

    if is_distributed:

        engine_inst.model = torch.nn.parallel.DistributedDataParallel(

            engine_inst.model,

            device_ids=[local_gpu_id] if local_gpu_id is not None else None,

            output_device=local_gpu_id if local_gpu_id is not None else None,

            find_unused_parameters=True  # 关键：避免“未使用参数”导致的归约不同步

        )

        if rank == 0:

            print(f'Using DistributedDataParallel on GPUs: {device_ids}')



    if rank == 0:
        # 计算并打印总参数量
        model_for_count = engine_inst.model.module if hasattr(engine_inst.model, 'module') else engine_inst.model
        total_params = sum(p.numel() for p in model_for_count.parameters())
        trainable_params = sum(p.numel() for p in model_for_count.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        print("start training...", flush=True)

        # 创建日志文件路径
        log_dir = os.path.dirname(args.save) if os.path.dirname(args.save) else '.'
        os.makedirs(log_dir, exist_ok=True)
        log_file = args.save + "_training_log.txt"

        # 初始化日志文件，写入表头和模型参数信息
        with open(log_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write(f"Training Log - Experiment ID: {args.expid}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Data: {args.data}\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")
            f.write(f"Arguments: {args}\n")
            f.write("="*100 + "\n\n")



    his_loss = []

    val_time = []

    train_time = []
    best_valid_loss = float('inf')
    non_improving_epochs = 0



    per_rank_batch = args.batch_size // world_size if is_distributed else args.batch_size



    for epoch in range(1, args.epochs + 1):

        train_loss = []

        train_mape = []

        train_rmse = []

        train_mae = []



        t1 = time.time()



        # 保证各 rank shuffle 一致

        np.random.seed(args.seed + epoch)

        dataloader['train_loader'].shuffle()



        for it, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):

            # x,y 是全局 batch，大小固定为 args.batch_size

            trainx = torch.tensor(x, dtype=torch.float32, device=device).permute(0, 3, 2, 1)

            trainy = torch.tensor(y, dtype=torch.float32, device=device).permute(0, 3, 2, 1)



            # 按 rank 等分切片（不允许空/不等长）

            if is_distributed:

                bsz = trainx.shape[0]

                # 由于上面断言，bsz == args.batch_size，且能整除 world_size

                start = rank * per_rank_batch

                end = (rank + 1) * per_rank_batch

                trainx_local = trainx[start:end]

                trainy_local = trainy[start:end]

            else:

                trainx_local = trainx

                trainy_local = trainy



            metrics = engine_inst.train(trainx_local, trainy_local[:, 0, :, :])

            avg_loss = ddp_avg(metrics[0], device, world_size)

            avg_mape = ddp_avg(metrics[1], device, world_size)

            avg_rmse = ddp_avg(metrics[2], device, world_size)

            avg_mae = ddp_avg(metrics[3], device, world_size)



            if rank == 0:

                train_loss.append(avg_loss)

                train_mape.append(avg_mape)

                train_rmse.append(avg_rmse)

                train_mae.append(avg_mae)



                if it % args.print_every == 0:

                    print(

                        f'Iter: {it:03d}, '

                        f'Train Loss: {train_loss[-1]:.4f}, '

                        f'Train MAPE: {train_mape[-1]:.4f}, '

                        f'Train RMSE: {train_rmse[-1]:.4f}, '

                        f'Train MAE: {train_mae[-1]:.4f}',

                        flush=True

                    )

            if args.max_train_batches > 0 and it + 1 >= args.max_train_batches:
                break



        t2 = time.time()

        if rank == 0:

            train_time.append(t2 - t1)



        # 验证

        valid_loss = []

        valid_mape = []

        valid_rmse = []

        valid_mae = []



        s1 = time.time()

        for it, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):

            testx = torch.tensor(x, dtype=torch.float32, device=device).permute(0, 3, 2, 1)

            testy = torch.tensor(y, dtype=torch.float32, device=device).permute(0, 3, 2, 1)



            if is_distributed:

                bsz = testx.shape[0]

                # val_loader 也会 pad 到 batch_size，因此同样整除

                start = rank * per_rank_batch

                end = (rank + 1) * per_rank_batch

                testx_local = testx[start:end]

                testy_local = testy[start:end]

            else:

                testx_local = testx

                testy_local = testy



            metrics = engine_inst.eval(testx_local, testy_local[:, 0, :, :])

            avg_loss = ddp_avg(metrics[0], device, world_size)

            avg_mape = ddp_avg(metrics[1], device, world_size)

            avg_rmse = ddp_avg(metrics[2], device, world_size)

            avg_mae = ddp_avg(metrics[3], device, world_size)



            if rank == 0:

                valid_loss.append(avg_loss)

                valid_mape.append(avg_mape)

                valid_rmse.append(avg_rmse)

                valid_mae.append(avg_mae)

            if args.max_validation_batches > 0 and it + 1 >= args.max_validation_batches:
                break



        s2 = time.time()

        if rank == 0:

            print('Epoch: {:03d}, Inference Time: {:.4f} secs'.format(epoch, (s2 - s1)))

            val_time.append(s2 - s1)



            mtrain_loss = float(np.mean(train_loss))

            mtrain_mape = float(np.mean(train_mape))

            mtrain_rmse = float(np.mean(train_rmse))

            mtrain_mae = float(np.mean(train_mae))



            mvalid_loss = float(np.mean(valid_loss))

            mvalid_mape = float(np.mean(valid_mape))

            mvalid_rmse = float(np.mean(valid_rmse))

            mvalid_mae = float(np.mean(valid_mae))

            his_loss.append(mvalid_loss)



            log = ('Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Train MAE: {:.4f}, '

                   'Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Valid MAE: {:.4f}, Training Time: {:.4f}/epoch')

            print(log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mtrain_mae,

                             mvalid_loss, mvalid_mape, mvalid_rmse, mvalid_mae, (t2 - t1)), flush=True)

            # 将日志写入文件
            with open(log_file, 'a') as f:
                f.write(log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mtrain_mae,
                                   mvalid_loss, mvalid_mape, mvalid_rmse, mvalid_mae, (t2 - t1)) + "\n")


            save_path = args.save + "_epoch_" + str(epoch) + "_" + str(round(mvalid_loss, 2)) + ".pth"

            save_model(engine_inst, save_path)

            if mvalid_loss < best_valid_loss:
                best_valid_loss = mvalid_loss
                non_improving_epochs = 0
                # Stable name for evaluation scripts; epoch-specific files are retained.
                save_model(engine_inst, args.save + "_best.pth")
            else:
                non_improving_epochs += 1

            if args.patience > 0 and non_improving_epochs >= args.patience:
                print(
                    f'Early stopping at epoch {epoch}: no validation improvement for '
                    f'{args.patience} epochs.', flush=True
                )
                with open(log_file, 'a') as f:
                    f.write(f'Early stopping at epoch {epoch}: no validation improvement for {args.patience} epochs.\\n')
                break



        if is_distributed:

            dist.barrier()



    if rank == 0:

        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))

        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))



        if args.skip_final_test:
            print('Skipping legacy full-tensor test; checkpoints are ready for streaming evaluation.', flush=True)
            return

        # 测试：仅在 rank 0 上执行

        bestid = int(np.argmin(his_loss))

        best_path = args.save + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"

        load_model(engine_inst, best_path, device)



        outputs = []

        realy = torch.tensor(dataloader['y_test'], dtype=torch.float32, device=device)

        realy = realy.transpose(1, 3)[:, 0, :, :]



        # 在测试时直接使用未包装的模型以避免 DDP 通信

        model_for_infer = engine_inst.model.module if hasattr(engine_inst.model, 'module') else engine_inst.model

        model_for_infer.eval()



        for it, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):

            testx = torch.tensor(x, dtype=torch.float32, device=device)

            testx = testx.permute(0, 3, 2, 1)

            with torch.no_grad():

                # Match trainer.train/eval: PEMS08 has 12 history steps while
                # this temporal stack has a 13-step receptive field.
                testx = nn.functional.pad(testx, (1, 0, 0, 0))

                preds = model_for_infer(testx)

                preds = preds.permute(0, 3, 2, 1)

            outputs.append(preds.squeeze())



        yhat = torch.cat(outputs, dim=0)

        yhat = yhat[:realy.size(0), ...]
        # Save full raw-scale predictions for transition-window evaluation.
        pred_raw_all = scaler.inverse_transform(yhat).detach().cpu().numpy().transpose(0, 2, 1).astype(np.float32)
        true_raw_all = realy.detach().cpu().numpy().transpose(0, 2, 1).astype(np.float32)

        pred_save_dir = args.save + "_predictions"
        os.makedirs(pred_save_dir, exist_ok=True)
        pred_save_path = os.path.join(pred_save_dir, f"exp{args.expid}_pred_true.npz")
        np.savez_compressed(pred_save_path, pred=pred_raw_all, true=true_raw_all)
        print(f"Saved full test predictions to {pred_save_path}")




        print("Training finished")

        print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

        # 将训练完成信息写入日志文件
        with open(log_file, 'a') as f:
            f.write("\n" + "="*100 + "\n")
            f.write("Training finished\n")
            f.write(f"The valid loss on best model is {str(round(his_loss[bestid], 4))}\n")
            f.write(f"Best model path: {best_path}\n")
            f.write("="*100 + "\n\n")
            f.write("Test Results:\n")
            f.write("-"*100 + "\n")


        amae = []

        amape = []

        armse = []
        selected_metrics = {}

        for h in range(args.seq_length):

            pred = scaler.inverse_transform(yhat[:, :, h])

            real = realy[:, :, h]

            metrics = util.metric(pred, real)

            # util.metric returns fractional MAPE (e.g. 0.0887).  The shared
            # PEMS08 report convention is percentage, matching the other
            # baseline runners, so convert only at reporting time.
            mape_percent = metrics[1] * 100.0

            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'

            print(log.format(h + 1, metrics[0], mape_percent, metrics[2]))

            # 将每个horizon的测试结果写入日志文件
            with open(log_file, 'a') as f:
                f.write(log.format(h + 1, metrics[0], mape_percent, metrics[2]) + "\n")

            amae.append(metrics[0])

            amape.append(mape_percent)

            armse.append(metrics[2])

            if (h + 1) in (3, 6, 12):
                selected_metrics[str(h + 1)] = {
                    'MAE': float(metrics[0]),
                    'RMSE': float(metrics[2]),
                    'MAPE': float(mape_percent),
                }



        log = f'On average over {args.seq_length} horizons, Test MAE: {np.mean(amae):.4f}, Test MAPE: {np.mean(amape):.4f}, Test RMSE: {np.mean(armse):.4f}'

        print(log)

        final_save_path = args.save + "_exp" + str(args.expid) + "_best_" + str(round(his_loss[bestid], 2)) + ".pth"

        # 将平均测试结果写入日志文件
        with open(log_file, 'a') as f:
            f.write("-"*100 + "\n")
            f.write(log + "\n")
            f.write("="*100 + "\n\n")
            f.write(f"Final best model saved at: {final_save_path}\n")

        # Compact report for the native PEMS08 12->12 setup.  MAPE is the
        # percentage returned by util.metric and must not be multiplied again.
        selected_metrics['Average'] = {
            'MAE': float(np.mean(amae)),
            'RMSE': float(np.mean(armse)),
            'MAPE': float(np.mean(amape)),
        }
        metrics_csv = args.save + '_metrics.csv'
        with open(metrics_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['horizon', 'MAE', 'RMSE', 'MAPE'])
            writer.writeheader()
            for horizon in ('3', '6', '12', 'Average'):
                if horizon in selected_metrics:
                    writer.writerow({'horizon': horizon, **selected_metrics[horizon]})
        print(f'Saved selected test metrics to {metrics_csv}', flush=True)


        save_model(engine_inst, final_save_path)



    if is_distributed:

        dist.barrier()

        dist.destroy_process_group()





def main():

    device_ids = parse_device_ids(args.devices)

    use_cuda = torch.cuda.is_available()

    world_size = len(device_ids) if use_cuda else 1



    t1 = time.time()

    if use_cuda and world_size > 1:

        mp.spawn(run, nprocs=world_size, args=(device_ids,))

    else:

        run(0, device_ids)

    t2 = time.time()

    if world_size == 1:

        print("Total time spent: {:.4f}".format(t2 - t1))





if __name__ == "__main__":

    main()
