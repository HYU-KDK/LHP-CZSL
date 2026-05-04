"""Sanity check: build LHP-CZSL v1, run 1 train step, check no NaN/crash."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
from torch.utils.data import DataLoader
from collections import Counter

from parameters import parser
from utils import load_args, set_seed
from model.model_factory import get_model
from dataset import CompositionDataset


def main():
    config = parser.parse_args(['--yml_path', 'config/lhp_czsl_v1_mit_l14.yml'])
    load_args(config.yml_path, config)
    set_seed(config.seed)

    train_dataset = CompositionDataset(
        config.dataset_path, phase='train',
        split='compositional-split-natural',
        same_prim_sample=config.same_prim_sample,
    )

    attrs = [a.replace('.', ' ').lower() for a in train_dataset.attrs]
    classes = [c.replace('.', ' ').lower() for c in train_dataset.objs]
    offset = len(attrs)

    model = get_model(config, attributes=attrs, classes=classes, offset=offset).cuda()

    print(f"[v1] attr_k dist: {Counter(model.attr_k).most_common()}")
    print(f"[v1] obj_k  dist: {Counter(model.obj_k).most_common()}")
    print(f"[v1] attr_sem_mask True count: {model.attr_sem_mask.sum().item()}")
    print(f"[v1] obj_sem_mask  True count: {model.obj_sem_mask.sum().item()}")

    loader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                        shuffle=True, num_workers=0)
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx
    train_pairs = torch.tensor([(attr2idx[a], obj2idx[o])
                                for a, o in train_dataset.train_pairs]).cuda()

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=config.lr)
    scaler = torch.cuda.amp.GradScaler()
    model.train()

    batch = next(iter(loader))
    with torch.cuda.amp.autocast():
        predict = model(batch, train_pairs)
        loss = model.loss_calu(predict, batch)

    print(f"[v1] step1 loss: {loss.item():.4f}")
    if torch.isnan(loss) or torch.isinf(loss):
        print("[v1] FAIL: loss is NaN/Inf"); sys.exit(1)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    nan_grads = sum(
        1 for p in model.parameters()
        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
    )
    print(f"[v1] params with NaN/Inf grad: {nan_grads}")
    if nan_grads > 0:
        print("[v1] FAIL"); sys.exit(1)

    # also test val_forward path
    model.eval()
    val_dataset = CompositionDataset(
        config.dataset_path, phase='val', split='compositional-split-natural')
    val_loader = DataLoader(val_dataset, batch_size=config.eval_batch_size,
                            shuffle=False, num_workers=0)
    val_batch = next(iter(val_loader))
    with torch.no_grad(), torch.cuda.amp.autocast():
        val_predict = model(val_batch, train_pairs)
    print(f"[v1] val_forward output shapes: {[t.shape for t in val_predict]}")

    print("[v1] SANITY OK")


if __name__ == '__main__':
    main()
