"""
xai.py  –  Explainability techniques for Narrative Frame Prediction.

Methods
───────
1. Integrated Gradients  – primary technique (different from client1's rollout)
2. Grad-CAM++            – enhanced version of standard Grad-CAM
3. Attention Visualisation – recency-biased attention maps
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .helpers import PALETTE, reverse_normalise


# ══════════════════════════════════════════════════════════════
# 1. Integrated Gradients
# ══════════════════════════════════════════════════════════════

def integrated_gradients(model, frames, captions, num_steps=50, gt_captions=None):

    model.eval()  # keep eval for BatchNorm

    baseline = torch.zeros_like(frames)
    attributions = torch.zeros(frames.shape[1], device=frames.device)

    with torch.backends.cudnn.flags(enabled=False):  # ✅ FIX

        for step in range(num_steps):
            alpha = step / num_steps
            interpolated = (baseline + alpha * (frames - baseline)).requires_grad_(True)

            out = model(interpolated, captions, gt_captions, sampling_rate=0.0)

            score = out["pred_frame"].mean() + out["caption_logits"].mean()
            model.zero_grad(set_to_none=True)
            score.backward()

            if interpolated.grad is not None:
                grad_mag = interpolated.grad[0].abs().sum(dim=(1, 2, 3))
                attributions += grad_mag.detach()

    delta = (frames - baseline)[0].abs().sum(dim=(1, 2, 3))
    attributions = attributions * delta / num_steps

    attributions = (attributions - attributions.min()) / \
                   (attributions.max() - attributions.min() + 1e-8)

    return attributions.cpu().numpy()


def plot_integrated_gradients(attributions: np.ndarray,
                               save_to: str = None):
    """Bar chart with red theme."""
    K      = len(attributions)
    labels = [f"Frame {k+1}" for k in range(K)]
    colors = [PALETTE["primary"] if a > 0.5 else PALETTE["secondary"]
              for a in attributions]

    fig, ax = plt.subplots(figsize=(max(6, K * 1.2), 4),
                            facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.bar(labels, attributions, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Attribution Score", color="white")
    ax.set_title("Integrated Gradients – Per-Frame Attribution",
                 color=PALETTE["accent"], fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, color="white")
    plt.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        print(f"[XAI] Integrated gradients saved → {save_to}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 2. Grad-CAM++ on ImageFeatureExtractor
# ══════════════════════════════════════════════════════════════

class GradCAMPlusPlus:
    """
    Grad-CAM++ (Chattopadhay et al., 2018) — improved version of Grad-CAM
    that better handles multiple instances of the same class in an image
    by using second-order gradients for more accurate localisation.
    """

    def __init__(self, img_extractor):
        self.extractor   = img_extractor
        self._acts       = None
        self._grads      = None
        target_layer     = list(img_extractor.feature_net.children())[-2]
        self._fwd_handle = target_layer.register_forward_hook(self._capture_acts)
        self._bwd_handle = target_layer.register_full_backward_hook(self._capture_grads)

    def _capture_acts(self, module, inp, out):
        self._acts = out.detach()

    def _capture_grads(self, module, grad_in, grad_out):
        self._grads = grad_out[0].detach()

    def compute(self, frame: torch.Tensor) -> np.ndarray:
        """
        Args:
            frame : (1, C, H, W)
        Returns:
            heatmap : (H, W) in [0,1]
        """
        frame = frame.requires_grad_(True)
        feat  = self.extractor(frame)
        score = feat.mean()
        self.extractor.zero_grad()
        score.backward()

        # Grad-CAM++ weights using second-order approximation
        grads  = self._grads                              # (1, C, h, w)
        acts   = self._acts                               # (1, C, h, w)
        grads2 = grads ** 2
        grads3 = grads ** 3
        denom  = 2 * grads2 + acts * grads3
        denom  = torch.where(denom != 0, denom,
                             torch.ones_like(denom))
        alpha  = grads2 / denom
        weights = (alpha * F.relu(grads)).mean(dim=[2, 3], keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        H, W = frame.shape[2], frame.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def release(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def plot_gradcam_pp(frame: torch.Tensor, heatmap: np.ndarray,
                    title: str = "Grad-CAM++", save_to: str = None):
    img_np = reverse_normalise(frame)
    colored = cm.hot(heatmap)[..., :3]              # hot colormap (different)
    overlay = 0.55 * img_np + 0.45 * colored

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
    axes[0].imshow(img_np);            axes[0].set_title("Original",  color="white")
    axes[1].imshow(heatmap, cmap="hot"); axes[1].set_title("Grad-CAM++", color=PALETTE["primary"])
    axes[2].imshow(overlay);           axes[2].set_title("Overlay",   color=PALETTE["secondary"])
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=12, color=PALETTE["accent"], fontweight="bold")
    plt.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        print(f"[XAI] Grad-CAM++ saved → {save_to}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 3. Attention Map Visualisation
# ══════════════════════════════════════════════════════════════

def extract_frame_scores(attn_weights: torch.Tensor) -> np.ndarray:
    """
    Extract per-frame importance from narrative attention weights.
    Args:
        attn_weights : (1, heads, K, K)
    Returns:
        scores : (K,) numpy array
    """
    if attn_weights.dim() == 4:
        attn = attn_weights.mean(dim=1)     # (1, K, K)
    else:
        attn = attn_weights
    scores = attn[0].mean(dim=0)            # (K,)
    scores = scores.detach().cpu().numpy()
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores


def plot_frame_scores(scores: np.ndarray, save_to: str = None):
    K      = len(scores)
    labels = [f"Frame {k+1}" for k in range(K)]
    fig, ax = plt.subplots(figsize=(max(6, K * 1.2), 4), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    bars = ax.bar(labels, scores,
                  color=[plt.cm.YlOrRd(s) for s in scores],
                  edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Attention Score", color="white")
    ax.set_title("Narrative Attention – Frame Importance",
                 color=PALETTE["accent"], fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, color="white")
    plt.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        print(f"[XAI] Frame scores saved → {save_to}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 4. Run all XAI
# ══════════════════════════════════════════════════════════════

def run_all_xai(model, batch: dict, device: torch.device,
                out_dir: str = "outputs/xai", vocab_inv: dict = None):
    model.eval()
    frames   = batch["frames"][:1].to(device)
    captions = batch["captions"][:1].to(device)
    gt_caps  = batch["gt_captions"][:1].to(device)

    with torch.no_grad():
        out = model(frames, captions, gt_caps, sampling_rate=0.0)

    # 1. Integrated Gradients
    attrs = integrated_gradients(model, frames, captions, gt_captions=gt_caps)
    plot_integrated_gradients(attrs,
        save_to=os.path.join(out_dir, "integrated_gradients.png"))

    # 2. Grad-CAM++
    gcam    = GradCAMPlusPlus(model.img_extractor)
    heatmap = gcam.compute(frames[0, 0].unsqueeze(0))
    gcam.release()
    plot_gradcam_pp(frames[0, 0].cpu(), heatmap,
        title="Grad-CAM++ – Context Frame 1",
        save_to=os.path.join(out_dir, "gradcam_pp_frame1.png"))

    # 3. Attention scores
    scores = extract_frame_scores(out["attn_map"])
    plot_frame_scores(scores,
        save_to=os.path.join(out_dir, "narrative_attention_scores.png"))

    print("[XAI] All figures saved to", out_dir)
