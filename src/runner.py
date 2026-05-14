"""
runner.py  –  Training and evaluation runner.

Usage
-----
python src/runner.py --settings settings.yaml
python src/runner.py --settings settings.yaml --test_only --resume saved_models/best.pt
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from architecture import NarrativePredictionModel, NarrativeLoss
from helpers import (fix_randomness, pick_device, load_settings,
                     store_model, restore_model, plot_training_curves,
                     score_bleu, score_l1)


def run_epoch(net, loader, optimiser, criterion, device, epoch, cfg):
    net.train()
    total = 0.0
    for step, batch in enumerate(loader):
        frames   = batch["frames"].to(device)
        captions = batch["captions"].to(device)
        gt_frame = batch["gt_frame"].to(device)
        gt_caps  = batch["gt_captions"].to(device)

        optimiser.zero_grad()
        out    = net(frames, captions, gt_caps, sampling_rate=0.5)
        losses = criterion(out["pred_frame"], gt_frame,
                           out["caption_logits"], gt_caps)
        losses["total"].backward()
        nn.utils.clip_grad_norm_(net.parameters(),
                                  cfg["optimisation"]["clip_norm"])
        optimiser.step()
        total += losses["total"].item()

        if step % 20 == 0:
            print(f"  Ep{epoch:02d} Bt{step:03d}/{len(loader)} "
                  f"loss={losses['total'].item():.4f} "
                  f"(frame={losses['frame'].item():.4f} "
                  f"caption={losses['caption'].item():.4f})")
    return total / len(loader)


@torch.no_grad()
def evaluate(net, loader, criterion, device):
    net.eval()
    total = 0.0
    for batch in loader:
        frames   = batch["frames"].to(device)
        captions = batch["captions"].to(device)
        gt_frame = batch["gt_frame"].to(device)
        gt_caps  = batch["gt_captions"].to(device)
        out      = net(frames, captions, gt_caps, sampling_rate=0.0)
        losses   = criterion(out["pred_frame"], gt_frame,
                             out["caption_logits"], gt_caps)
        total   += losses["total"].item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings",  default="settings.yaml")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--resume",    default=None)
    args = parser.parse_args()

    cfg    = load_settings(args.settings)
    device = pick_device()
    fix_randomness(cfg["optimisation"]["random_seed"])

    print(f"[Setup] Device: {device}")

    net = NarrativePredictionModel(cfg).to(device)
    print(f"[Model] Parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")

    criterion  = NarrativeLoss(
        frame_w   = cfg["optimisation"]["frame_loss_w"],
        caption_w = cfg["optimisation"]["caption_loss_w"],
    )
    optimiser  = torch.optim.Adam(
        net.parameters(),
        lr           = cfg["optimisation"]["lr"],
        weight_decay = cfg["optimisation"]["l2_penalty"],
    )
    scheduler  = StepLR(
        optimiser,
        step_size = cfg["optimisation"]["lr_step_size"],
        gamma     = cfg["optimisation"]["lr_gamma"],
    )

    start_ep, best_loss = 0, float("inf")
    if args.resume:
        start_ep, best_loss = restore_model(args.resume, net, optimiser, device)

    os.makedirs(cfg["io"]["ckpt_dir"],    exist_ok=True)
    os.makedirs(cfg["io"]["output_dir"],  exist_ok=True)

    if args.test_only:
        print("[Test mode] Skipping training.")
        return

    train_hist, val_hist = [], []

    for ep in range(start_ep + 1, cfg["optimisation"]["num_epochs"] + 1):
        tr = run_epoch(net, train_loader, optimiser, criterion, device, ep, cfg)
        vl = evaluate(net, val_loader, criterion, device)
        scheduler.step()

        train_hist.append(tr)
        val_hist.append(vl)

        print(f">>> Ep {ep:03d} | Train {tr:.4f} | Val {vl:.4f}")

        if vl < best_loss:
            best_loss = vl
            store_model({
                "epoch":     ep,
                "net_state": net.state_dict(),
                "opt_state": optimiser.state_dict(),
                "best_loss": best_loss,
                "cfg":       cfg,
            }, os.path.join(cfg["io"]["ckpt_dir"], "best.pt"))
            print(f"    ✓ Saved best (val={best_loss:.4f})")

    plot_training_curves(
        train_hist, val_hist,
        save_to=os.path.join(cfg["io"]["output_dir"], "loss_curves.png")
    )
    print("Training finished.")


if __name__ == "__main__":
    main()
