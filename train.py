#
import argparse
import os
import pickle
import pprint

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from model.model_factory import get_model
from parameters import parser

# from test import *
import test as test
from dataset import CompositionDataset
from utils import *


def setup_distributed():
    """Initialize torch.distributed if launched via torchrun. Returns (is_dist, rank, world_size, local_rank)."""
    if "LOCAL_RANK" in os.environ and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def is_main_process(rank):
    return rank == 0


def train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset,
                is_dist, rank, world_size, local_rank):
    if is_dist:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_sampler = None
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    model.train()
    best_metric = 0
    best_loss = 1e5
    best_epoch = 0
    final_model_state = None

    val_results = []

    scheduler = get_scheduler(optimizer, config, len(train_dataloader))
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()

    train_losses = []

    scaler = torch.cuda.amp.GradScaler()

    inner_model = model.module if is_dist else model

    for i in range(config.epoch_start, config.epochs):
        if is_dist:
            train_sampler.set_epoch(i)

        if is_main_process(rank):
            progress_bar = tqdm.tqdm(
                total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
            )
        else:
            progress_bar = None

        epoch_train_losses = []
        for bid, batch in enumerate(train_dataloader):
            with torch.cuda.amp.autocast():
                predict = model(batch, train_pairs)
                loss = inner_model.loss_calu(predict, batch)
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            scheduler = step_scheduler(scheduler, config, bid, len(train_dataloader))

            epoch_train_losses.append(loss.item())
            if progress_bar is not None:
                progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
                progress_bar.update()

        if progress_bar is not None:
            progress_bar.close()
            progress_bar.write(f"epoch {i+1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))

        if is_dist:
            dist.barrier()

        # Evaluation and checkpointing only on rank 0
        if is_main_process(rank):
            if (i + 1) % config.save_every_n == 0:
                torch.save(inner_model.state_dict(), os.path.join(config.save_path, f"epoch_{i}.pt"))

            print("Evaluating val dataset:")
            val_result = evaluate(inner_model, val_dataset, config)
            val_results.append(val_result)

            if config.val_metric == 'best_loss' and val_result[config.val_metric] < best_loss:
                best_loss = val_result['best_loss']
                best_epoch = i
                torch.save(inner_model.state_dict(), os.path.join(
                    config.save_path, "val_best.pt"))
            if config.val_metric != 'best_loss' and val_result[config.val_metric] > best_metric:
                best_metric = val_result[config.val_metric]
                best_epoch = i
                torch.save(inner_model.state_dict(), os.path.join(
                    config.save_path, "val_best.pt"))

            final_model_state = inner_model.state_dict()
            if i + 1 == config.epochs:
                print("--- Evaluating test dataset on Closed World ---")
                inner_model.load_state_dict(torch.load(os.path.join(
                    config.save_path, "val_best.pt"
                )))
                evaluate(inner_model, test_dataset, config)

        # Switch back to train mode on all ranks (eval ran only on rank 0 but
        # safe to call train() on every rank).
        model.train()

        if is_dist:
            dist.barrier()

    if is_main_process(rank) and config.save_final_model:
        torch.save(final_model_state, os.path.join(config.save_path, f'final_model.pt'))


def evaluate(model, dataset, config):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
            model, dataset, config)
    test_stats = test.test(
            dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )
    test_saved_results = dict()
    result = ""
    key_set = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    for key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
        test_saved_results[key] = round(test_stats[key], 4)
    print(result)
    test_saved_results['loss'] = loss_avg
    return test_saved_results



if __name__ == "__main__":
    config = parser.parse_args()
    if config.yml_path:
        load_args(config.yml_path, config)

    is_dist, rank, world_size, local_rank = setup_distributed()

    if is_main_process(rank):
        print(config)
        print(f"[DDP] is_dist={is_dist} world_size={world_size} rank={rank} local_rank={local_rank}")

    # Per-rank seed offset so dataloader shuffling differs across ranks
    set_seed(config.seed + rank)

    dataset_path = config.dataset_path

    train_dataset = CompositionDataset(dataset_path,
                                       phase='train',
                                       split='compositional-split-natural',
                                       same_prim_sample=config.same_prim_sample)

    val_dataset = CompositionDataset(dataset_path,
                                     phase='val',
                                     split='compositional-split-natural')

    test_dataset = CompositionDataset(dataset_path,
                                       phase='test',
                                       split='compositional-split-natural')

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)

    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()

    if is_dist:
        # broadcast_buffers=True ensures prototype queues stay in sync across ranks.
        # Since every rank performs the same all-gathered prototype update, the
        # broadcast is essentially a no-op safeguard.
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
        )

    optimizer = get_optimizer(model.module if is_dist else model, config)

    if is_main_process(rank):
        os.makedirs(config.save_path, exist_ok=True)

    if is_dist:
        dist.barrier()

    train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset,
                is_dist, rank, world_size, local_rank)

    if is_main_process(rank):
        with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
            pickle.dump(config, fp)
        write_json(os.path.join(config.save_path, "config.json"), vars(config))
        print("done!")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()
