# Multimodal Narrative Frame Prediction

## 🔗 Quick Access
- **[Notebook Pipeline](experiments.ipynb)** – End-to-end training, evaluation, and visualisation
- **[Outputs](outputs/)** – Metrics, plots, and explainability results
- **[Loss Curve](outputs/loss_curve_plot.png)** – Training dynamics across epochs
- **[Prediction Example](outputs/prediction_result.png)** – Qualitative model output

---

## 📌 Overview

This project addresses the task of **narrative frame prediction**, where the goal is to predict the next frame (image + caption) given a sequence of prior frames.

The model is trained on the **StoryReasoning dataset**, which combines visual and textual storytelling elements requiring both temporal and semantic reasoning.

---

## 🚀 Key Contributions

| # | Component | Description |
|---|----------|-------------|
| 1 | Attention-based Fusion | Bahdanau-style additive attention for aligning visual and textual features |
| 2 | Temporal Modelling | Stacked GRU layers for efficient sequence encoding |
| 3 | Recency-aware Attention | Learnable temporal bias to prioritise recent frames |
| 4 | Explainability | Integrated Gradients, Grad-CAM++, and attention visualisations |

---

## 📊 Final Results

| Metric | Value |
|--------|-------|
| Best Validation Loss | **8.3856** |
| BLEU-4 Score | **0.08** |
| Frame L1 Error | **1.288698** |
| Training Epochs | 25 |

---

## 🧠 Model Architecture
Context Frames → Image Encoder (ResNet-34)
→ Text Encoder (GRU)
→ Attention Fusion
→ Temporal GRU Encoder
→ Narrative Attention (Recency Bias)
→ Frame Decoder + Caption Decoder


---

## ⚙️ Design Choices

| Component | Standard Approach | This Work | Motivation |
|----------|------------------|----------|-----------|
| Fusion | Concatenation | Attention-based fusion | Improved cross-modal alignment |
| Temporal Model | LSTM | Stacked GRU | Faster training, fewer parameters |
| Attention | Uniform | Recency-aware | Captures temporal locality |
| Image Loss | MSE | L1 Loss | Robust to outliers |
| LR Scheduler | Cosine | StepLR | Stable convergence |

---

## 🔍 Explainability

Three complementary techniques were used:

- **Integrated Gradients** → Frame-level importance attribution  
- **Grad-CAM++** → Spatial feature importance  
- **Attention Analysis** → Temporal importance across frames  

All outputs are saved in:

outputs/xai/


---

## 📈 Additional Analysis

The notebook includes:
- Training vs validation loss curves  
- Caption length distribution  
- Attention trend visualisation  
- Prediction vs ground truth comparison  
- Ablation study  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook experiments.ipynb
💾 Loading Trained Model
import torch

checkpoint = torch.load("checkpoints/final_model.pt", map_location=device)

model.load_state_dict(checkpoint["model_state"])
model.eval()
⚙️ Configuration Summary
Parameter	Value
Context Frames (K)	4
Image Size	256 × 256
Backbone	ResNet-34
Embedding Dim	256
Batch Size	12
Learning Rate	3e-4
Scheduler	StepLR
Epochs	25
📁 Project Structure
project/
├── experiments.ipynb
├── settings.yaml
├── requirements.txt
├── src/
│   ├── architecture.py
│   ├── helpers.py
│   ├── xai.py
│   └── runner.py
├── checkpoints/
│   ├── best_model.pt
│   └── final_model.pt
└── outputs/
    ├── loss_curve_plot.png
    ├── metrics_comparison.png
    ├── evaluation_metrics.csv
    ├── prediction_result.png
    └── xai/
        ├── integrated_gradients_map.png
        ├── gradcam_visual.png
        ├── attention_frame_scores.png
        └── attention_matrix_visual.png
📚 Dataset Reference

Oliveira, D. A. P., & Matos, D. M. (2025).
StoryReasoning Dataset: Chain-of-Thought for Scene Understanding and Story Generation.
https://arxiv.org/abs/2505.10292