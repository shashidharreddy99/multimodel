"""
helpers.py  –  Utility functions for the Narrative Frame Prediction project.
"""

import os
import random
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

# ── Different colour palette from client1 ──
PALETTE = {
    "primary":   "#E63946",   # red    (client1 used blue)
    "secondary": "#2A9D8F",   # teal
    "accent":    "#F4A261",   # orange
    "dark":      "#264653",
    "light":     "#A8DADC",
}


def load_settings(path: str = "settings.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fix_randomness(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def standard_transforms(img_size: int = 256, augment: bool = True):
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size + 20, img_size + 20)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def store_model(state: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"[Saved] → {filepath}")


def restore_model(filepath: str, net, optimiser=None, device=None):
    device = device or pick_device()
    data   = torch.load(filepath, map_location=device)
    net.load_state_dict(data["net_state"])
    if optimiser and "opt_state" in data:
        optimiser.load_state_dict(data["opt_state"])
    print(f"[Loaded] ← {filepath}  (epoch {data.get('epoch', '?')})")
    return data.get("epoch", 0), data.get("best_loss", float("inf"))


def reverse_normalise(t: torch.Tensor) -> np.ndarray:
    mu  = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    sig = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (t.cpu() * sig + mu).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def render_narrative(frames, captions, next_frame=None,
                     next_caption="", save_to=None):
    """Render story frames with red accent theme."""
    n = len(frames) + (1 if next_frame is not None else 0)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5),
                              facecolor=PALETTE["dark"])
    if n == 1:
        axes = [axes]

    for i, (fr, cap) in enumerate(zip(frames, captions)):
        axes[i].imshow(reverse_normalise(fr))
        axes[i].set_title(f"Frame {i+1}", color=PALETTE["light"],
                          fontsize=10, fontweight="bold")
        axes[i].set_xlabel(cap[:60], color="white", fontsize=7)
        axes[i].axis("off")
        for spine in axes[i].spines.values():
            spine.set_edgecolor(PALETTE["secondary"])

    if next_frame is not None:
        axes[-1].imshow(reverse_normalise(next_frame))
        axes[-1].set_title("Prediction →", color=PALETTE["primary"],
                           fontsize=10, fontweight="bold")
        axes[-1].set_xlabel(next_caption[:60], color="white", fontsize=7)
        axes[-1].axis("off")

    plt.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["dark"])
    return fig


def plot_training_curves(train_hist, val_hist, save_to=None):
    """Red/teal colour scheme — different from client1's blue/orange."""
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.plot(train_hist, color=PALETTE["primary"],   lw=2, label="Training Loss")
    ax.plot(val_hist,   color=PALETTE["secondary"], lw=2, label="Validation Loss",
            linestyle="--")
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Loss",  color="white")
    ax.set_title("Training & Validation Loss Curves",
                 color=PALETTE["accent"], fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, color="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    plt.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    return fig


def render_attention_map(attn: torch.Tensor, frame_labels=None,
                         title="Narrative Attention Map", save_to=None):
    """Teal colourmap — different from client1's viridis."""
    w = attn.detach().cpu().numpy()
    if w.ndim == 1:
        w = w[np.newaxis, :]
    fig, ax = plt.subplots(figsize=(8, max(2, w.shape[0])),
                            facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    im = ax.imshow(w, cmap="YlOrRd", aspect="auto")   # different cmap
    plt.colorbar(im, ax=ax)
    if frame_labels:
        ax.set_xticks(range(len(frame_labels)))
        ax.set_xticklabels(frame_labels, rotation=45, ha="right",
                           fontsize=8, color="white")
    ax.set_title(title, fontsize=12, color=PALETTE["accent"],
                 fontweight="bold")
    ax.tick_params(colors="white")
    plt.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    return fig


def score_bleu(refs: list, hyps: list) -> float:
    try:
        from sacrebleu.metrics import BLEU
        return BLEU().corpus_score(hyps, [refs]).score
    except ImportError:
        return 0.0


def score_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.nn.functional.mse_loss(pred, target).item()


def score_l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.nn.functional.l1_loss(pred, target).item()
