# Satellite Image Super-Resolution using SwinIR
## Klymo Hackathon - Team Alpha
**Team Members:** Mangesh Thale | Kedar Mahajan | Koustubh Gadekar

## Overview

Deep learning solution for 4× satellite image super-resolution (160×160 → 640×640) using memory-optimized SwinIR with temporal fusion. Achieves PSNR 22.55 dB, SSIM 0.573 with only 3.87M parameters and 0.1GB GPU memory.

---

## Architecture

### Data Pipeline
- **Dataset:** WorldStrat satellite imagery (3,929 HR areas, 3,928 LR areas)
- **Input/Output:** 160×160×3 RGB → 640×640×3 RGB (4× upscaling)
- **Temporal Processing:** 8 frames per location with zero-padding for missing frames

### Model Components

```
Input (160×160×27) → Temporal Fusion (27→3 channels) → SwinIR Backbone → Output (640×640×3)
```

**1. Temporal Fusion Module**
- Processes 8 temporal frames (27 input channels: 8×3 RGB + 3 base)
- Convolutional fusion: 27 → 48 → 24 → 3 channels
- Lightweight preprocessing before SwinIR

**2. SwinIR Backbone**
- Pretrained: RealSR BSRGAN DFO SwinIR-M x4
- Window-based self-attention with PixelShuffle upsampler
- 3.87M parameters (memory-optimized)

**3. Memory Optimizations**
- Gradient checkpointing + Mixed precision (FP16)
- Batch size 1 with 8-step gradient accumulation
- ~0.1GB GPU memory during training

### Loss Function

```
Total Loss = L1 + 0.05·MS-SSIM + 0.1·Gradient + 0.02·Perceptual + 0.01·Range
```

| Component | Weight | Purpose |
|-----------|--------|---------|
| L1 Loss | 1.0 | Pixel-wise reconstruction |
| MS-SSIM | 0.05 | Multi-scale structural similarity |
| Gradient (Sobel) | 0.1 | Edge preservation |
| Perceptual (VGG19) | 0.02 | Feature matching at relu3_3 |
| Range Loss | 0.01 | Dynamic range matching |

### Training Configuration

**Optimizer:** AdamW (lr: 2×10⁻⁴, weight decay: 1×10⁻⁴)  
**Scheduler:** Cosine Annealing (T_max: 100, eta_min: 1×10⁻⁶)  
**Epochs:** 61 (early stopped from 100, patience: 15)  
**Effective Batch Size:** 8 (gradient accumulation)  
**Hardware:** NVIDIA Tesla T4

### Data Augmentation

- Random horizontal/vertical flips (p=0.5)
- Random rotation (±15°)
- Shift-Scale-Rotate (shift: 0.0625, scale: 0.1)
- Brightness-Contrast (0.2 each)
- Coarse Dropout (8-12 holes, 8×8px)
- GaussNoise (variance: 10-50)
- Geometric distortions (optical/grid/elastic)

---

## Results

### Performance Metrics (Best @ Epoch 46)

| Metric | Value |
|--------|-------|
| Validation Loss | 0.1844 |
| PSNR | 22.55 dB |
| SSIM | 0.5730 |
| Training Time | 61 epochs |
| Inference Speed | 0.5-1.0 sec/image |
| Model Size | 3.87M parameters |
| GPU Memory | ~0.1GB |

### Training Progress (Baseline to MainModel)

- **Loss Improvement:** 0.53 → 0.18 (66% reduction)
- **PSNR Improvement:** 9.78 → 22.55 dB (+13 dB)
- **SSIM Improvement:** 0.094 → 0.573 (6× increase)
- **Convergence:** Smooth and stable with early stopping at epoch 61

### Key Observations

**Strengths:**
- Efficient 4× upscaling with low computational cost
- Effective temporal fusion reduces noise and improves quality
- Stable training with pretrained weights
- Production-ready inference pipeline

**Baseline Comparisons:**
- **vs Bicubic:** Better detail preservation and sharper edges
- **vs CNNs:** Superior long-range dependencies and texture reconstruction
- **vs GANs:** More stable training with better PSNR/SSIM trade-off

---

## Technical Innovations

1. **Memory-Optimized Architecture:** 0.1GB GPU usage, FP16 training, gradient checkpointing
2. **Temporal Fusion Strategy:** Lightweight 8-frame aggregation before backbone
3. **Pretrained Transfer Learning:** SwinIR RealSR weights adapted for satellite imagery
4. **Hybrid Multi-Loss Function:** Balanced combination of pixel, structural, and perceptual losses
5. **Efficient Training:** Fast convergence (~37 sec/epoch) with early stopping

---

## Key Features

✅ 4× Super-Resolution (160×160 → 640×640)  
✅ 8-Frame Temporal Fusion  
✅ Multi-Component Loss Optimization  
✅ Memory Efficient (0.1GB GPU)  
✅ Lightweight (3.87M parameters)  
✅ Fast Training (38 minutes)  
✅ Pretrained SwinIR Backbone  
✅ Mixed Precision FP16  
✅ Production Ready Pipeline  

---

## Dataset Details

**Training Subset (50 areas):**
- Pairs: 3928
- Train: 35 areas
- Validation: 10 areas
- Test: 5 areas

**Specifications:**
- Temporal frames: 8 per area
- Source: WorldStrat 12-bit satellite imagery
- Data type: Float16

---

## Future Improvements

**Data:** Expand to full 3,928 areas, leverage multi-band spectral data  
**Architecture:** Experiment with SwinIR-Large, refine attention mechanisms  
**Training:** Extended epochs (100+), loss weight tuning, progressive training  
**Optimization:** Hyperparameter search, larger batch sizes, model pruning  

---

## Conclusion

This project demonstrates efficient satellite image super-resolution with extreme memory optimization. The 3.87M parameter model achieves 4× upscaling using temporal fusion and pretrained SwinIR weights, making it suitable for deployment on resource-constrained systems. While metrics show room for improvement with more data and training, the system provides a practical foundation for satellite imagery enhancement in remote sensing and geospatial analysis.

---

## Acknowledgments

- **SwinIR:** Original architecture by Jingyun Liang et al.
- **WorldStrat Dataset:** Satellite imagery for super-resolution research

## License

Uses SwinIR architecture under its original license. Refer to the original repository for details.
