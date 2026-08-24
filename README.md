# Enhanced GuideDepth for Monocular Depth Estimation

This repository implements an enhanced lightweight monocular depth estimation pipeline based on **GuideDepth** (Rudolph et al., ICRA 2022). Our work extends the original guided decoding concept with single-stage semantic skip connections, robust reverse Huber (BerHu) depth loss, multi-scale deep supervision, LocalBins-lite depth quantization, and CutDepth data augmentation.

---

## 🌟 Key Method Highlights

- **Backbone & Decoder**: Built upon DDRNet-23 slim with a single-stage semantic skip connection (`H/4` resolution, reduced to 16 channels) to enrich spatial recovery without adding decoder overhead.
- **Robust Loss Formulation**: `Depth_Loss` combines reverse Huber (`BerHu`, threshold = 0.2), Structural Similarity (`SSIM`), and Image Gradient loss ($\alpha = 0.1, \beta = 1.0, \gamma = 1.0$).
- **Auxiliary Training Supervision**:
  - **Multi-Scale Deep Supervision**: Auxiliary loss heads at decoder stages 1 & 2 (`weight = 0.1`).
  - **LocalBins-Lite**: Discrete depth bin classification auxiliary head (`weight = 0.1`, 16 bins).
  - **CutDepth Augmentation**: Ground truth depth patch overlay on RGB inputs (`probability = 0.75`).
- **Zero Evaluation Overhead**: Auxiliary supervision heads operate strictly during training (`self.training = True`) and are bypassed during inference/evaluation, incurring **0 MACs overhead** at test time.

---

## 📂 Repository Structure

```
.
├── config.yaml                   # Best performing training & model configuration
├── main.py                       # Main training and test evaluation entry point
├── training.py                   # Trainer loop, loss calculation, evaluation logging
├── evaluate.py                   # Standalone test set evaluation module
├── losses.py                     # Loss functions (BerHu, SSIM, Gradient loss)
├── metrics.py                    # Evaluation metrics (RMSE, MAE, Delta1-3, REL, Lg10)
├── profile_params_macs.py        # Model parameter & MAC count profiling script
├── QUICK_RUN.md                  # Quick usage guide for training & evaluation
├── requirements.txt              # Project dependencies
├── model/
│   ├── GuideDepth.py             # Model architecture
│   ├── DDRNet_23_slim.py         # Backbone feature extractor
│   ├── modules.py                # Guided Upsampling Block & LocalBinsLite
│   └── loader.py                 # Weight loader and model builder
├── data/
│   ├── nyu_reduced.py            # NYU Depth V2 dataset loader
│   ├── transforms.py             # Data augmentations (CutDepth, Resize, Flips)
│   └── datasets.py               # DataLoader setup wrapper
└── scripts/
    └── eval_val_test_protocol.py # M0 test-protocol aligned validation diagnostic
```

---

## 🛠️ Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   cd Depth-estimation
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Quick Start

### 1. Training
To train using the optimal configuration defined in `config.yaml`:
```bash
python main.py --train --config config.yaml
```

Override specific arguments via CLI:
```bash
python main.py --train --config config.yaml --batch_size 8 --learning_rate 0.0001 --num_epochs 20
```

### 2. Evaluation
To evaluate a trained checkpoint on the test set:
```bash
python main.py --eval --config config.yaml --weights_path ./results/<TIMESTAMP>/models/best_model.pth
```

Automatically evaluate using the latest checkpoint in `./results/`:
```bash
python main.py --eval --config config.yaml --weights_path latest
```

---

## 📊 Computational Efficiency Profiling (Table 7 Verification)

To measure exact Parameters and MACs of the inference model (with auxiliary heads detached):
```bash
python profile_params_macs.py --resolution 240 320
```

---

## 📖 Citation & Acknowledgments

This implementation extends the baseline model from:
> **Lightweight Monocular Depth Estimation through Guided Decoding**  
> Michael Rudolph, Timothy K. Miller, and Michael D. Zelinsky  
> IEEE International Conference on Robotics and Automation (ICRA), 2022.
