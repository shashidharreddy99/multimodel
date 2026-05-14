"""
architecture.py  –  Narrative Frame Prediction Model.

Pipeline summary
────────────────
ImageFeatureExtractor  → ResNet-34 backbone  →  256-d frame embedding
CaptionFeatureExtractor→ GRU encoder         →  256-d caption embedding
AdditiveAttentionFusion→ Additive attention  →  256-d fused embedding  [INNOVATION #1]
TemporalContextEncoder → Stacked GRU         →  256-d sequence repr    [INNOVATION #2]
NarrativeAttention     → Multi-head self-attn with decay bias          [INNOVATION #3]
FrameDecoder           → Upsample+Conv decoder
CaptionDecoder         → GRU with scheduled sampling

Innovation justification
────────────────────────
1. Additive attention fusion (Bahdanau et al., 2015) uses a learnable
   compatibility function rather than dot-product, making it more robust
   when image and text embeddings occupy different subspaces.
2. Stacked GRU replaces LSTM for sequence modelling — GRUs have fewer
   parameters (no separate cell state) and have been shown to converge
   faster on medium-sized datasets (Chung et al., 2014).
3. Narrative attention applies an exponential recency bias so that frames
   closer to the prediction target receive higher baseline attention,
   reflecting the temporal locality of causal narrative structure.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ══════════════════════════════════════════════════════════════
# 1. Image Feature Extractor
# ══════════════════════════════════════════════════════════════

class ImageFeatureExtractor(nn.Module):
    """ResNet-34 based frame encoder."""

    def __init__(self, embed_size: int = 256, use_pretrained: bool = True,
                 finetune: bool = True):
        super().__init__()
        base = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if use_pretrained else None
        )
        # Strip classifier; keep up to avgpool → 512-d
        self.feature_net = nn.Sequential(*list(base.children())[:-1])
        if not finetune:
            for p in self.feature_net.parameters():
                p.requires_grad = False

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embed_size),
            nn.BatchNorm1d(embed_size),
            nn.Tanh(),
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame : (N, C, H, W)
        Returns:
            (N, embed_size)
        """
        raw = self.feature_net(frame)           # (N, 512, 1, 1)
        return self.embedding_head(raw)


# ══════════════════════════════════════════════════════════════
# 2. Caption Feature Extractor
# ══════════════════════════════════════════════════════════════

class CaptionFeatureExtractor(nn.Module):
    """GRU-based caption encoder."""

    def __init__(self, vocab_size: int = 8000, word_embed: int = 128,
                 hidden_size: int = 256, num_layers: int = 2,
                 dropout_rate: float = 0.4, out_size: int = 256):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_embed, padding_idx=0)
        self.gru = nn.GRU(
            word_embed, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,                # unidirectional GRU
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, out_size),
            nn.BatchNorm1d(out_size),
            nn.Tanh(),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids : (N, T)
        Returns:
            (N, out_size)
        """
        embedded = self.word_embedding(token_ids)   # (N, T, word_embed)
        _, h_n   = self.gru(embedded)               # h_n: (layers, N, hidden)
        last_h   = h_n[-1]                          # (N, hidden)
        return self.projection(last_h)


# ══════════════════════════════════════════════════════════════
# 3. Additive Attention Fusion  [INNOVATION #1]
# ══════════════════════════════════════════════════════════════

class AdditiveAttentionFusion(nn.Module):
    """
    INNOVATION #1 – Additive Attention Fusion.

    Uses a learned compatibility function f(v, t) = w^T tanh(W_v*v + W_t*t)
    to score visual-textual alignment, producing an attended fused embedding.
    Unlike dot-product attention, additive attention learns a separate
    compatibility function which is more expressive when the two modalities
    have different geometric properties in embedding space.

    Reference: Bahdanau et al. (2015) Neural Machine Translation by
    Jointly Learning to Align and Translate.
    """

    def __init__(self, embed_size: int = 256):
        super().__init__()
        self.W_img  = nn.Linear(embed_size, embed_size, bias=False)
        self.W_cap  = nn.Linear(embed_size, embed_size, bias=False)
        self.score  = nn.Linear(embed_size, 1, bias=False)
        self.merge  = nn.Linear(embed_size * 2, embed_size)
        self.norm   = nn.LayerNorm(embed_size)

    def forward(self, img_emb: torch.Tensor,
                cap_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_emb : (N, D)
            cap_emb : (N, D)
        Returns:
            fused   : (N, D)
        """
        # Additive compatibility
        energy = torch.tanh(self.W_img(img_emb) + self.W_cap(cap_emb))  # (N, D)
        alpha  = torch.sigmoid(self.score(energy))                        # (N, 1)

        # Weighted combination
        attended = alpha * img_emb + (1 - alpha) * cap_emb               # (N, D)
        fused    = self.merge(torch.cat([attended, cap_emb], dim=-1))     # (N, D)
        return self.norm(fused)


# ══════════════════════════════════════════════════════════════
# 4. Temporal Context Encoder – Stacked GRU  [INNOVATION #2]
# ══════════════════════════════════════════════════════════════

class TemporalContextEncoder(nn.Module):
    """
    INNOVATION #2 – Stacked GRU Temporal Encoder.

    GRUs have a simpler gating mechanism than LSTMs (reset + update gates
    vs input + forget + output gates), making them faster to train with
    fewer parameters. On medium-sized datasets like StoryReasoning, GRUs
    often achieve comparable or superior performance to LSTMs while
    converging in fewer epochs (Chung et al., 2014).

    A two-layer stacked architecture captures both low-level temporal
    transitions (layer 1) and higher-level narrative structure (layer 2).
    """

    def __init__(self, input_size: int = 256, hidden_size: int = 256,
                 depth: int = 2, dropout_rate: float = 0.4):
        super().__init__()
        self.stacked_gru = nn.GRU(
            input_size, hidden_size,
            num_layers=depth,
            batch_first=True,
            dropout=dropout_rate if depth > 1 else 0.0,
        )
        self.normaliser = nn.LayerNorm(hidden_size)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence : (N, K, input_size)
        Returns:
            encoded  : (N, K, hidden_size)
        """
        out, _ = self.stacked_gru(sequence)     # (N, K, hidden_size)
        return self.normaliser(out)


# ══════════════════════════════════════════════════════════════
# 5. Narrative Attention with Recency Bias  [INNOVATION #3]
# ══════════════════════════════════════════════════════════════

class NarrativeAttention(nn.Module):
    """
    INNOVATION #3 – Narrative Self-Attention with Exponential Recency Bias.

    Standard self-attention treats all positions equally. In narrative
    sequences, recent frames are causally more relevant to the next frame
    than distant ones. We add an exponential recency bias:
        bias[i,j] = -lambda * |i - j|
    where lambda is a learnable scalar. This gives the model a soft
    inductive prior towards local temporal attention while retaining the
    flexibility to attend globally when necessary.
    """

    def __init__(self, embed_size: int = 256, num_heads: int = 4,
                 max_seq_len: int = 20):
        super().__init__()
        self.attention  = nn.MultiheadAttention(embed_size, num_heads,
                                                batch_first=True)
        # Learnable recency decay scalar (initialised to small positive value)
        self.decay_rate = nn.Parameter(torch.tensor(0.1))
        self.max_len    = max_seq_len
        self.layer_norm = nn.LayerNorm(embed_size)

    def _recency_bias(self, seq_len: int,
                      device: torch.device) -> torch.Tensor:
        """Compute (seq_len, seq_len) exponential recency bias matrix."""
        positions = torch.arange(seq_len, device=device).float()
        dist      = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        bias      = -torch.abs(self.decay_rate) * dist   # (K, K)
        return bias

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x : (N, K, D)
        Returns:
            out          : (N, K, D)
            attn_weights : (N, K, K)
        """
        N, K, D  = x.shape
        bias     = self._recency_bias(K, x.device)       # (K, K)

        out, attn_weights = self.attention(
            x, x, x,
            attn_mask=bias,
            average_attn_weights=False
        )
        out = self.layer_norm(out + x)
        return out, attn_weights


# ══════════════════════════════════════════════════════════════
# 6. Frame Decoder (Upsample + Conv)
# ══════════════════════════════════════════════════════════════

class FrameDecoder(nn.Module):
    """Upsample + Conv decoder → (N, 3, 256, 256)."""

    def __init__(self, input_size: int = 256):
        super().__init__()
        self.fc = nn.Linear(input_size, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),                          # 8→16
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2),                          # 16→32
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2),                          # 32→64
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),  nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2),                          # 64→128
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),  nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2),                          # 128→256
            nn.Conv2d(16,  3, 3, padding=1),
            nn.Sigmoid(),                                          # output in [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 256, 8, 8)
        return self.decoder(x)                                    # (N, 3, 256, 256)


# ══════════════════════════════════════════════════════════════
# 7. Caption Decoder
# ══════════════════════════════════════════════════════════════

class CaptionDecoder(nn.Module):
    """GRU caption decoder with scheduled sampling."""

    def __init__(self, vocab_size: int = 8000, word_embed: int = 128,
                 hidden_size: int = 256, num_layers: int = 2,
                 max_len: int = 50):
        super().__init__()
        self.max_len    = max_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.token_embed = nn.Embedding(vocab_size, word_embed, padding_idx=0)
        self.gru         = nn.GRU(word_embed, hidden_size, num_layers,
                                  batch_first=True,
                                  dropout=0.4 if num_layers > 1 else 0.0)
        self.h_init      = nn.Linear(hidden_size, hidden_size * num_layers)
        self.vocab_proj  = nn.Linear(hidden_size, vocab_size)

    def _init_state(self, ctx: torch.Tensor) -> torch.Tensor:
        N  = ctx.size(0)
        h0 = self.h_init(ctx).view(self.num_layers, N, self.hidden_size)
        return h0.contiguous()

    def forward(self, ctx: torch.Tensor,
                gt_tokens: torch.Tensor = None,
                sampling_rate: float = 0.5) -> torch.Tensor:
        """
        Args:
            ctx         : (N, D)
            gt_tokens   : (N, T) ground-truth for scheduled sampling
            sampling_rate : probability of using ground-truth token
        Returns:
            logits : (N, T, vocab_size)
        """
        N  = ctx.size(0)
        h  = self._init_state(ctx)
        T  = gt_tokens.size(1) if gt_tokens is not None else self.max_len
        inp = torch.full((N,), 2, dtype=torch.long, device=ctx.device)  # <SOS>=2

        all_logits = []
        for t in range(T):
            emb  = self.token_embed(inp).unsqueeze(1)       # (N, 1, E)
            out, h = self.gru(emb, h)
            logit  = self.vocab_proj(out.squeeze(1))        # (N, vocab)
            all_logits.append(logit)

            if gt_tokens is not None and torch.rand(1).item() < sampling_rate:
                inp = gt_tokens[:, t]
            else:
                inp = logit.argmax(dim=-1)

        return torch.stack(all_logits, dim=1)               # (N, T, vocab)


# ══════════════════════════════════════════════════════════════
# 8. Full Narrative Prediction Model
# ══════════════════════════════════════════════════════════════

class NarrativePredictionModel(nn.Module):
    """
    End-to-end model for narrative frame prediction.

    Input  : N context frames + captions  →  (N, K, C, H, W), (N, K, T)
    Output : predicted frame K+1 image + caption logits
    """

    def __init__(self, cfg: dict):
        super().__init__()
        ie  = cfg["img_encoder"]
        ce  = cfg["caption_encoder"]
        fu  = cfg["multimodal_fusion"]
        tm  = cfg["temporal"]
        sa  = cfg["self_attention"]
        cd  = cfg["caption_decoder"]
        K   = cfg["data"]["context_frames"]

        self.img_extractor = ImageFeatureExtractor(
            embed_size    = ie["embed_size"],
            use_pretrained = ie["use_pretrained"],
            finetune      = ie["finetune"],
        )
        self.cap_extractor = CaptionFeatureExtractor(
            vocab_size  = ce["vocab_limit"],
            word_embed  = ce["word_embed"],
            hidden_size = ce["gru_hidden"],
            num_layers  = ce["num_gru_layers"],
            dropout_rate = ce["dropout_rate"],
            out_size    = ce["out_size"],
        )
        self.fusion = AdditiveAttentionFusion(embed_size=fu["output_size"])
        self.temporal_enc = TemporalContextEncoder(
            input_size  = fu["output_size"],
            hidden_size = tm["hidden_size"],
            depth       = tm["depth"],
            dropout_rate = tm["dropout_rate"],
        )
        self.narrative_attn = NarrativeAttention(
            embed_size  = tm["hidden_size"],
            num_heads   = sa["heads"],
            max_seq_len = sa["max_len"],
        )
        self.context_aggregator = nn.Linear(tm["hidden_size"], tm["hidden_size"])

        self.frame_dec   = FrameDecoder(input_size=tm["hidden_size"])
        self.caption_dec = CaptionDecoder(
            vocab_size  = ce["vocab_limit"],
            word_embed  = ce["word_embed"],
            hidden_size = cd["hidden_size"],
            num_layers  = cd["depth"],
            max_len     = cd["max_output_len"],
        )

        self._attn_cache = None

    def encode_narrative(self, frames: torch.Tensor,
                         captions: torch.Tensor) -> tuple:
        N, K = frames.shape[:2]

        frame_embs   = []
        caption_embs = []
        for k in range(K):
            frame_embs.append(self.img_extractor(frames[:, k]))
            caption_embs.append(self.cap_extractor(captions[:, k]))

        frame_seq   = torch.stack(frame_embs,   dim=1)   # (N, K, D)
        caption_seq = torch.stack(caption_embs, dim=1)   # (N, K, D)

        fused_seq = []
        for k in range(K):
            fused_seq.append(self.fusion(frame_seq[:, k], caption_seq[:, k]))
        fused_seq = torch.stack(fused_seq, dim=1)         # (N, K, D)

        temporal_out = self.temporal_enc(fused_seq)       # (N, K, D)
        attn_out, attn_w = self.narrative_attn(temporal_out)
        self._attn_cache = attn_w.detach()

        ctx = self.context_aggregator(attn_out.mean(dim=1))  # (N, D)
        return ctx, attn_w

    def forward(self, frames: torch.Tensor, captions: torch.Tensor,
                gt_captions: torch.Tensor = None,
                sampling_rate: float = 0.5) -> dict:
        ctx, attn_w = self.encode_narrative(frames, captions)

        pred_frame   = self.frame_dec(ctx)
        caption_logits = self.caption_dec(ctx, gt_captions, sampling_rate)

        return {
            "pred_frame":      pred_frame,
            "caption_logits":  caption_logits,
            "context_vector":  ctx,
            "attn_map":        attn_w,
        }


# ══════════════════════════════════════════════════════════════
# 9. Combined Loss
# ══════════════════════════════════════════════════════════════

class NarrativeLoss(nn.Module):
    def __init__(self, frame_w: float = 0.8, caption_w: float = 1.2,
                 pad_idx: int = 0):
        super().__init__()
        self.frame_w    = frame_w
        self.caption_w  = caption_w
        self.frame_loss   = nn.L1Loss()                          # L1 instead of MSE
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, pred_frame, gt_frame,
                caption_logits, gt_caption) -> dict:
        fl = self.frame_loss(pred_frame, gt_frame)
        N, T, V = caption_logits.shape
        cl = self.caption_loss(
            caption_logits.reshape(N * T, V),
            gt_caption[:, :T].reshape(N * T)
        )
        total = self.frame_w * fl + self.caption_w * cl
        return {"total": total, "frame": fl, "caption": cl}
