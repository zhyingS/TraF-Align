# ------------------------------------------------------------------------------------

# TrafAlign: Trajectory-aware Feature Alignment for Asynchronous Multi-agent Perception
# Copyright (c) 2024 Tsinghua. All Rights Reserved.
# Licensed: TDG-Attribution-NonCommercial-NoDistrib
# ------------------------------------------------------------------------------------
# Modified from OpenCOOD (https://github.com/DerrickXuNu/OpenCOOD)
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
# ps -ef |grep train.py |grep -v grep |awk '{print $2}'|xargs kill -9

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import argparse
import sys

sys.path.append("/".join(sys.argv[0].split("/")[:-2]))

import statistics
import logging
import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import hypes_yaml.yaml_utils as yaml_utils
from datasets import build_dataset
from utils import train_utils, multi_gpu_utils
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def train_parser():
    parser = argparse.ArgumentParser()
    checkpoint = 0  # whether train from checkpoint
    if checkpoint:
        parser.add_argument(
            "--hypes_yaml",
            type=str,
            default="",
            help="data generation yaml file needed ",
        )
        parser.add_argument(
            "--model_dir",
            default="",
            help="Continued training path",
        )
    else:
        parser.add_argument(
            "--hypes_yaml",
            type=str,
            default="hypes_yaml/dair_v2x_seq/dair_v2x_seq_Trafalign.yaml",
            help="data generation yaml file needed ",
        )
        parser.add_argument("--model_dir", default="", help="Continued training path")

    parser.add_argument(
        "--half", default=0, help="training with half precision, hard to converge"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    cfg = parser.parse_args()
    return cfg


def main():
    cfg = train_parser()
    hypes = yaml_utils.load_yaml(cfg.hypes_yaml, cfg)
    multi_gpu_utils.init_distributed_mode(cfg)
    gpus = 1 if not cfg.distributed else cfg.world_size

    hypes["train_params"]["gpus"] = gpus
    hypes = yaml_utils.check_pillar_params(hypes)

    print("---------------Creating Model------------------")
    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if we want to train from last checkpoint.
    if cfg.model_dir:
        saved_path = cfg.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
    else:
        init_epoch = 0
        saved_path = train_utils.setup_train(hypes)

    print("-----------------Dataset Building------------------")
    train_dataset = build_dataset(hypes, set="train")
    validate_dataset = build_dataset(hypes, set="val")

    shuffle_each_epoch = hypes["dataset"].get("shuffle_each_epoch", False)

    if cfg.distributed:
        sampler_train = DistributedSampler(train_dataset, shuffle=True)
        sampler_val = DistributedSampler(validate_dataset)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes["train_params"]["train_batch_size"], drop_last=True
        )
        batch_sampler_val = torch.utils.data.BatchSampler(
            sampler_val, hypes["train_params"]["val_batch_size"], drop_last=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler_train,
            num_workers=12,
            pin_memory=False,
            collate_fn=train_dataset.collate_batch,
        )
        val_loader = DataLoader(
            validate_dataset,
            batch_sampler=batch_sampler_val,
            num_workers=12,
            pin_memory=False,
            collate_fn=validate_dataset.collate_batch,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=hypes["train_params"]["train_batch_size"],
            num_workers=1,
            collate_fn=train_dataset.collate_batch,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
        )
        val_loader = DataLoader(
            validate_dataset,
            batch_size=hypes["train_params"]["val_batch_size"],
            num_workers=1,
            collate_fn=validate_dataset.collate_batch,
            shuffle=False,
            pin_memory=False,
            drop_last=True,
        )

    logger = logging.getLogger("Train")
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")

    file_handler = logging.FileHandler(saved_path + "/train_logging.log")
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        criterion.to(device)

    model_without_ddp = model

    if cfg.distributed:
        find_unused_params = hypes["dataset"].get("find_unused_params", False)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.gpu], find_unused_parameters=find_unused_params
        )
        model_without_ddp = model.module

    # optimizer setup
    if "AdamW" in hypes["optimizer"]["core_method"]:
        optimizer = torch.optim.AdamW(
            list(model_without_ddp.parameters()) + list(criterion.parameters()),
            eps=1e-3,
            betas=hypes["optimizer"]["betas"],
            weight_decay=hypes["optimizer"]["weight_decay"],
            amsgrad=hypes["optimizer"]["amsgrad"],
        )
    else:
        optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)

    last_epoch = -1
    # lr scheduler setup
    scheduler = train_utils.setup_lr_schedular(
        hypes, optimizer, steps_per_epoch=len(train_loader), epoch=last_epoch
    )

    # record training
    writer = SummaryWriter(saved_path)

    logger.info("Training start")
    epoches = hypes["train_params"]["epoches"]
    scheduler_method = hypes["lr_scheduler"]["core_method"]

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if (
            scheduler_method != "cosineannealwarm"
            and not "OneCycleLR" in scheduler_method
        ):
            scheduler.step(epoch)
        logger.info(
            "At epoch %d, the learning rate is %.7f"
            % (epoch, optimizer.param_groups[0]["lr"])
        )

        if cfg.distributed:
            sampler_train.set_epoch(epoch)

        pbar = tqdm.tqdm(total=len(train_loader), leave=True)
        train_ave_loss = []

        for i, batch_data in enumerate(train_loader):
            model.train()
            model.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            if len(batch_data["processed_lidar"]["voxel_num_points"]) >= 0:
                preds = model(batch_data)
                loss = criterion(batch_data, preds, train=True)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if "OneCycleLR" in scheduler_method:
                    scheduler.step()
                if "cosineannealwarm" in scheduler_method:
                    scheduler.step_update(epoch * len(train_loader) + i)

                writer.add_scalar(
                    "Loss/LR",
                    optimizer.param_groups[0]["lr"],
                    epoch * len(train_loader) + i,
                )

                train_ave_loss.append(loss.item())

                logger.debug(
                    "[epoch %d][%d/%d], || Loss: %.4f "
                    % (epoch, i + 1, len(train_loader), loss.item())
                )
            pbar.update(1)

        train_ave_loss = statistics.mean(train_ave_loss)
        logger.info("At epoch %d, the training loss is %f" % (epoch, train_ave_loss))
        writer.add_scalar("Loss/TrainLoss", train_ave_loss, epoch)

        if (
            epoch % hypes["train_params"]["save_freq"] == 0
            or epoch == hypes["train_params"]["epoches"] - 1
        ):
            torch.save(
                model_without_ddp.state_dict(),
                os.path.join(saved_path, "net_epoch%d.pth" % (epoch + 1)),
            )

        if epoch % hypes["train_params"]["eval_freq"] == 0:
            valid_ave_loss = []
            pbar2 = tqdm.tqdm(total=len(val_loader), leave=True)

            with torch.no_grad():
                logger.info("validating")
                for i, batch_data in enumerate(val_loader):
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()
                    if len(batch_data["processed_lidar"]["voxel_num_points"]) >= 10:
                        batch_data = train_utils.to_device(batch_data, device)
                        preds, _ = model(batch_data)
                        loss = criterion(batch_data, preds)

                        logger.debug(
                            "[epoch %d][%d/%d], || Loss: %.4f "
                            % (epoch, i + 1, len(val_loader), loss.item())
                        )
                        valid_ave_loss.append(loss.item())
                        pbar2.update(1)

                valid_ave_loss = statistics.mean(valid_ave_loss)
                logger.info(
                    "At epoch %d, the validation loss is %f" % (epoch, valid_ave_loss)
                )
                writer.add_scalar("Loss/ValidLoss", valid_ave_loss, epoch)

        if shuffle_each_epoch:
            train_dataset.shuffle_each_epoch()

    print("Training Finished, checkpoints saved.")


if __name__ == "__main__":
    main()
