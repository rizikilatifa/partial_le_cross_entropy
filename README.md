# Partial Cross Entropy Loss for Weakly Supervised Segmentation

A TensorFlow/Keras implementation of Partial Cross Entropy Loss for training semantic segmentation models with sparse point annotations instead of full pixel-level masks.

## Overview

This project demonstrates how to train semantic segmentation models using only **point annotations** (sparse labels) instead of expensive full pixel-level annotations. The key innovation is the **Partial Cross Entropy Loss**, which only computes loss on labeled pixels while ignoring unlabeled ones.

### Key Results

| Supervision Type | Labeled Pixels | Best mIoU | Relative Performance |
|------------------|----------------|-----------|---------------------|
| Point (5 pts/class) | ~0.6% | 0.6601 | 99.4% |
| Full Supervision | 100% | 0.6640 | 100% |

**Finding:** Point supervision achieves comparable performance to full supervision while using only ~0.6% labeled pixels!

---

## Reasoning & Theoretical Foundation

### Why Does Point Supervision Work?

At first glance, training a segmentation model with only 0.6% labeled pixels seems impossible. However, several key insights explain why this approach is effective:

#### 1. Spatial Continuity Hypothesis

Natural images (especially remote sensing) exhibit **spatial continuity** — neighboring pixels tend to belong to the same class. This is the fundamental assumption that makes partial supervision viable.

```
┌─────────────────────────────────────┐
│  If pixel (100, 100) is "Forest"    │
│  Then pixels (100, 101), (101, 100) │
│  are likely also "Forest"           │
│                                     │
│  One point label informs            │
│  many surrounding pixels            │
└─────────────────────────────────────┘
```

The model learns to **propagate** class information from sparse points to unlabeled regions through learned spatial priors.

#### 2. Convolutional Neural Networks Learn Smooth Functions

CNNs have an inherent **inductive bias** toward learning smooth, locally consistent functions:

- **Translation equivariance**: What works at one location works at others
- **Local receptive fields**: Predictions depend on local context
- **Hierarchical features**: Low-level features (edges, textures) are shared across regions

This means that even without explicit labels, the network learns to produce consistent predictions across spatially coherent regions.

#### 3. The Loss Function Design

Our **Partial Cross Entropy Loss** is designed to:

```python
# Standard Cross Entropy (full supervision)
L = -Σ log(p(y_i | x_i))    for ALL pixels i

# Partial Cross Entropy (point supervision)
L = -Σ log(p(y_i | x_i))    ONLY for LABELED pixels i
    where y_i ≠ ignore_index
```

**Key insight:** By computing loss only on labeled pixels, we:
- Avoid penalizing the model for "wrong" predictions on unlabeled pixels
- Allow the model to learn from the structure of the image itself
- Rely on the CNN's inductive bias to generalize to unlabeled regions

#### 4. Why So Few Points Suffice

Consider a 128×128 image with 5 classes:

| Total Pixels | Points per Class | Total Points | Labeled Ratio |
|--------------|------------------|--------------|---------------|
| 16,384 | 5 | 25 | 0.15% |
| 16,384 | 10 | 50 | 0.31% |

With 5 points per class, we provide:
- **Class distribution information**: Which classes exist in the image
- **Spatial anchors**: Where each class appears
- **Feature exemplars**: What each class looks like

The CNN then uses these anchors + the image structure to segment the rest.

### Mathematical Foundation

#### The Partial Loss Formulation

Given:
- Image: X ∈ ℝ^(H×W×C)
- Sparse labels: Y_sparse ∈ {-1, 0, 1, ..., K-1}^(H×W) where -1 means unlabeled
- Model: f_θ(X) → logits ∈ ℝ^(H×W×K)

The partial cross-entropy loss is:

```
L_partial(θ) = -1/N_L Σ_{(i,j)∈L} log(p_θ(y_ij | X))

where:
- L = {(i,j) : Y_sparse[i,j] ≠ -1}  (labeled pixels)
- N_L = |L| (number of labeled pixels)
- p_θ(y | X) = softmax(f_θ(X))[y]
```

#### Relationship to Semi-Supervised Learning

This is a form of **semi-supervised learning** where:
- Labeled data: The sparse point annotations
- Unlabeled data: All other pixels

The key difference from traditional semi-supervised learning:
- Unlabeled pixels are **implicit** in the same image (not separate samples)
- Spatial structure provides strong **prior** for label propagation

### When Does Point Supervision Work Best?

Point supervision is most effective when:

| Condition | Why It Helps |
|-----------|--------------|
| **Spatially coherent objects** | Continuity assumption holds |
| **Distinct class appearances** | Easy to distinguish classes |
| **Adequate class sampling** | Each class has at least 1-2 points |
| **Sufficient training data** | Model learns generalizable features |
| **Appropriate model capacity** | Not too simple (underfitting) or complex (overfitting) |

### When Might Point Supervision Struggle?

| Challenge | Mitigation |
|-----------|------------|
| **Fine boundary details** | Use boundary-focused sampling |
| **Many small objects** | Increase points per class |
| **Similar class appearances** | Add more diverse training examples |
| **Class imbalance** | Weight loss by class frequency |

### Why Our Results Show 99.4% Performance

Our synthetic dataset is relatively "easy" for point supervision because:

1. **Large contiguous regions** (Voronoi tessellation creates smooth segments)
2. **Distinct class colors** (each class has unique spectral signature)
3. **Balanced class distribution** (roughly equal area per class)
4. **Consistent textures** (within-class variation is low)

Real-world data may show a larger gap, but typically achieves 85-95% of full supervision performance with appropriate point density.

### Alternative Approaches We Considered

| Approach | Pros | Cons |
|----------|------|------|
| **Pseudo-labeling** | Uses unlabeled pixels | Can propagate errors |
| **Consistency regularization** | Better generalization | More complex implementation |
| **Class activation maps** | No point labels needed | Lower accuracy |
| **Scribble annotations** | More information per class | Higher annotation cost |

We chose **pure point supervision with partial CE loss** for its simplicity and strong empirical results.

---

## Project Structure

```
partial_ce_loss_project/
├── README.md                              # This documentation
├── QUICKSTART.md                          # Quick start guide
├── TECHNICAL_REPORT.md                    # Technical analysis
├── partial_ce_loss.py                     # Core loss implementation
├── partial_ce_loss_segmentation.ipynb     # PyTorch implementation
├── partial_ce_loss_segmentation_tf.ipynb  # TensorFlow implementation
├── partial_ce_loss_segmentation_tf_executed.ipynb  # Executed with outputs
└── requirements.txt                       # Dependencies
```

## Installation

### Requirements

```bash
pip install tensorflow>=2.10.0
pip install numpy scipy matplotlib pillow tqdm
pip install jupyter notebook
```

### For PyTorch Version

```bash
pip install torch torchvision
```

## Quick Start

### 1. Clone the Repository

```bash
git clone git@github.com:rizikilatifa/partial_le_cross_entropy.git
cd partial_le_cross_entropy
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

```bash
jupyter notebook partial_ce_loss_segmentation_tf.ipynb
```

Execute all cells to:
1. Test the Partial CE Loss function
2. Generate point labels from full masks
3. Build the U-Net model
4. Train on synthetic data
5. Visualize predictions

## Core Components

### 1. PartialCrossEntropyLoss

A custom Keras loss function that computes cross-entropy only on labeled pixels.

```python
class PartialCrossEntropyLoss(keras.losses.Loss):
    """
    Args:
        ignore_index: Value used for unlabeled pixels (default: -1)
        from_logits: Whether predictions are logits (default: True)
        label_smoothing: Label smoothing factor (default: 0.0)
    """
    def __init__(self, ignore_index=-1, from_logits=True, label_smoothing=0.0):
        super().__init__(name='partial_crossentropy')
        self.ignore_index = ignore_index
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
```

**Usage:**
```python
loss_fn = PartialCrossEntropyLoss(ignore_index=-1)
loss = loss_fn(y_true, y_pred)  # y_true contains -1 for unlabeled pixels
```

### 2. Point Label Generation

Convert full segmentation masks to sparse point annotations.

```python
def generate_point_labels(mask, num_points_per_class=5, strategy='random'):
    """
    Generate point annotations from a full segmentation mask.

    Args:
        mask: (H, W) - full segmentation mask
        num_points_per_class: Number of points to sample per class
        strategy: 'random', 'grid', or 'boundary'

    Returns:
        point_mask: (H, W) - mask with -1 for unlabeled, class label for labeled
        label_positions: list of (r, c) tuples
    """
```

**Sampling Strategies:**
- `random`: Random sampling from each class
- `grid`: Grid-based sampling for spatial coverage
- `boundary`: Focus on boundary pixels (more informative)

### 3. UNetLite Model

A lightweight U-Net architecture for semantic segmentation (~1.93M parameters).

### 4. SegmentationTrainer

A custom training class with built-in metrics tracking (mIoU, pixel accuracy).

### 5. tf.data Pipeline

Efficient data loading with on-the-fly point generation.

## Experiments

### Experiment 1: Point vs. Full Supervision

**Results:**
- Point Supervision mIoU: 0.6601
- Full Supervision mIoU: 0.6640
- Performance Gap: 0.0039 (0.6%)
- Relative Performance: 99.4%

### Experiment 2: Point Annotation Density

Tests performance with 1, 3, 5, 10, 20 points per class.

**Expected Results:**
- More points → Better performance (diminishing returns)
- 5-10 points per class is often sufficient

## Metrics

### Mean Intersection over Union (mIoU)

```
IoU = Intersection / Union
mIoU = mean(IoU per class)
```

### Pixel Accuracy

```
pixel_accuracy = correct_predictions / total_pixels
```

## Synthetic Dataset

The project uses synthetic remote sensing data for demonstration:

| Class | Name | Color |
|-------|------|-------|
| 0 | Forest | Green |
| 1 | Urban | Tan |
| 2 | Water | Blue |
| 3 | Agricultural | Yellow |
| 4 | Bare Soil | Brown |

Data generation uses Voronoi tessellation to create realistic segmentation masks.

## Comparison: PyTorch vs TensorFlow

| Aspect | PyTorch | TensorFlow |
|--------|---------|------------|
| Loss Class | `nn.Module` | `keras.losses.Loss` |
| Model | `nn.Module` class | Functional API `tf.keras.Model` |
| Data Loading | `DataLoader` + `Dataset` | `tf.data.Dataset` |
| Training Loop | Manual with `optimizer.step()` | Custom trainer with `tf.GradientTape` |
| Device Handling | `.to(device)` | Automatic |
| Metrics | Manual computation | `keras.metrics` |

## Extending the Project

### Using Real Data

Replace `SyntheticRemoteSensingDataset` with your own data:

```python
class RealDataset:
    def __init__(self, image_dir, mask_dir):
        self.images = sorted(glob(f"{image_dir}/*.png"))
        self.masks = sorted(glob(f"{mask_dir}/*.png"))

    def generate_sample(self, idx):
        image = np.array(Image.open(self.images[idx])) / 255.0
        mask = np.array(Image.open(self.masks[idx]))
        return image, mask
```

### Different Model Architectures

```python
# U-Net
model = build_unet_lite(input_shape=(256, 256, 3), num_classes=5)

# DeepLabV3 with ASPP
model = build_deeplabv3_lite(input_shape=(256, 256, 3), num_classes=5)
```

## Troubleshooting

### Common Issues

1. **CUDA Error**: Ensure CUDA drivers are installed for GPU support
2. **Memory Issues**: Reduce `batch_size` or `img_size`
3. **Slow Training**: Use `tf.data.AUTOTUNE` for prefetching

### Performance Tips

- Use smaller image sizes for faster experiments
- Increase `num_points_per_class` for better accuracy
- Enable mixed precision for GPU speedup:

```python
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

## References

- [Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/1603.06980)
- [Point Supervision for Semantic Segmentation](https://arxiv.org/abs/1506.02106)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## License

This project is provided for educational purposes. Feel free to use and modify.

## Author

Created by **rizikilatifa** with Claude Code assistance.
