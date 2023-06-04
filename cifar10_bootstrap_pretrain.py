"""
    xkp: 2023/6/2 pretrain stage of mae by applying bootstrapping on cifar10
"""

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch

from main_pretrain import get_args_parser

def set_cifar10_args(args):
    # modify the args to be trained on cifar10
    args.model = 'mae_vit_tiny_patch4'
    args.input_size = 32
    args.data_path = './datasets/cifar10'
    # args.epochs = 200
    return args

def get_addition_parser():
    parser = get_args_parser()
    parser.add_argument('-k', '--k', default=5, type=int, metavar='N', help='number of bootstrap iterations')
    parser.add_argument('--gpu_id', default=0, type=int, metavar='N', help='gpu id')

    # ema 
    parser.add_argument('--ema', action='store_true', help='use EMA')
    parser.add_argument('--ema_beta', default=0.9, type=float, metavar='M', help='ema beta for model update')

    return parser


# train_one_epoch: modified from engine_pretrain/train_one_epoch
# add compute loss using encoder output when old_model is not None
import math
import sys
from typing import Iterable
import util.misc as misc
import util.lr_sched as lr_sched
def train_one_epoch(boot_strap_iter: int,
                    model: torch.nn.Module,
                    last_model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Iter-Epoch: [{}-{}]'.format(boot_strap_iter, epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # if no old model, normal loss by imgs and pred
        if last_model is None:
            samples = samples.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

            loss_value = loss.item()
        else:
            with torch.cuda.amp.autocast():
                # last_model.eval()
                last_model.to(device)
                # check the type of last_model and model are BootstrapMAE
                from models_mae import BootstrapMAE
                assert isinstance(last_model, BootstrapMAE)
                assert isinstance(model, BootstrapMAE)

                # last model use encoder output, the full images (all patches) are used
                samples = samples.to(device, non_blocking=True)
                
                last_latent = last_model.forward_encoder_all(samples)
                # last_latent = last_latent.to(device, non_blocking=True)

                # this model use decoder output
                loss = model.forward_latent_loss(last_latent, samples, mask_ratio=args.mask_ratio)
            
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning) # ignore the warning of Interpolate because of unmatched version of package
    
    args = set_cifar10_args(args)

    timestamp = time.strftime('%m%d-%H%M')

    # same
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.gpu_id)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # use cifar10 from torchvision instead of ImageFolder
    # simple augmentation
    from torchvision.transforms import InterpolationMode
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])])
    dataset_train = datasets.CIFAR10(root=args.data_path, train=True, transform=transform_train, download=True)
    print(dataset_train)

    # no distributed sampler
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    print("Sampler_train = %s" % str(sampler_train))

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # start to train
    last_model = None
    each_epoch = args.epochs // args.k

    init_time = time.time()
    for i in range(args.k):
        print("Bootstrap iteration %d" % i)
        if i == 0:
            print("Training initial model, loss computed on imgs and pred")

        # init model
        if last_model: # not the first iteration, load encoder weight from last iteration
            print("Load last weight")
            model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, reconstruct_latent=True)
            model.init_encoder(last_model.state_dict())
        else:
            model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, reconstruct_latent=False)
            # print("Model = %s" % str(model))
            
        model.to(device)
        

        # init optimizer
        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        # define optimizer
        param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        print("Optimizer = %s" % str(optimizer))
        loss_scaler = NativeScaler()

        # train
        print(f"Start training for {args.epochs / args.k} epochs")
        start_time = time.time()
        for epoch in range(args.start_epoch, each_epoch):
            train_stats = train_one_epoch(
                i,
                model, last_model,
                data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )
            if args.output_dir and (epoch % 20 == 0 or epoch + 1 == each_epoch):
                misc.save_model(
                    args=args, model=None, model_without_ddp=model, optimizer=optimizer,    # single GPU
                    loss_scaler=loss_scaler, epoch='{}_boot{}_{}'.format(timestamp, i, epoch))

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,}

            if args.output_dir:
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log-" + timestamp + ".txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        # save the weight of the last iteration
        if args.ema and last_model is not None:
            # use ema to compute the weights
            # update the current model 
            for ema_param, param in zip(model.parameters(), last_model.parameters()):
                ema_param.data.mul_(1 - args.ema_beta).add_(param.data, alpha=args.ema_beta)
        
        if last_model is not None:
            import gc
            del last_model
            gc.collect()
        last_model = model
        
    total_time = time.time() - init_time
    print('Total training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))


if __name__ == "__main__":
    args = get_addition_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
