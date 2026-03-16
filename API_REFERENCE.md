# API Reference

This document provides detailed API documentation for all components in the Partial Cross Entropy Loss project.

---

## Table of Contents

1. [Loss Functions](#loss-functions)
2. [Point Label Generation](#point-label-generation)
3. [Model Architectures](#model-architectures)
4. [Dataset Classes](#dataset-classes)
5. [Training Utilities](#training-utilities)
6. [Metrics](#metrics)
7. [Visualization](#visualization)

---

## Loss Functions

### PartialCrossEntropyLoss

```python
class PartialCrossEntropyLoss(keras.losses.Loss)
```

Custom loss function for weakly supervised segmentation with sparse labels.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ignore_index` | int | -1 | Index value used to mark unlabeled pixels |
| `from_logits` | bool | True | Whether model outputs are raw logits |
| `label_smoothing` | float | 0.0 | Label smoothing factor (0.0 = no smoothing) |
| `name` | str | 'partial_crossentropy' | Name of the loss function |

#### Methods

##### `call(y_true, y_pred)`

Compute the partial cross-entropy loss.

**Parameters:**
- `y_true` (tf.Tensor): Ground truth labels with shape `(B, H, W)`. Unlabeled pixels should have value `ignore_index`.
- `y_pred` (tf.Tensor): Model predictions with shape `(B, H, W, C)` where C is the number of classes.

**Returns:**
- `tf.Tensor`: Scalar loss value.

**Example:**
```python
loss_fn = PartialCrossEntropyLoss(ignore_index=-1, from_logits=True)

# y_true has -1 for unlabeled pixels
y_true = tf.constant([[[0, -1], [-1, 1]]])  # Shape: (1, 2, 2)
y_pred = tf.random.normal([1, 2, 2, 3])     # Shape: (1, 2, 2, 3) for 3 classes

loss = loss_fn(y_true, y_pred)
```

---

## Point Label Generation

### generate_point_labels

```python
def generate_point_labels(mask, num_points_per_class=5, strategy='random')
```

Generate sparse point annotations from a full segmentation mask.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mask` | np.ndarray or tf.Tensor | required | Full segmentation mask with shape `(H, W)` |
| `num_points_per_class` | int | 5 | Number of points to sample per class |
| `strategy` | str | 'random' | Sampling strategy: 'random', 'grid', or 'boundary' |

#### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `point_mask` | np.ndarray | Mask with shape `(H, W)`. Contains class labels for labeled pixels, -1 for unlabeled |
| `label_positions` | list | List of `(row, col)` tuples indicating labeled pixel positions |

#### Sampling Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `random` | Random sampling within each class | General use |
| `grid` | Grid-based sampling for uniform coverage | Large objects |
| `boundary` | Focus on boundary pixels | Small objects, fine details |

**Example:**
```python
# Full mask with 3 classes
mask = np.array([
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [2, 2, 1, 1],
    [2, 2, 1, 1]
])

# Generate 2 points per class
point_mask, positions = generate_point_labels(mask, num_points_per_class=2, strategy='random')

# point_mask will have -1 for unlabeled pixels
# Example output:
# [[ 0, -1,  1, -1],
#  [-1,  0, -1,  1],
#  [ 2, -1, -1, -1],
#  [-1,  2, -1, -1]]
```

---

## Model Architectures

### build_unet_lite

```python
def build_unet_lite(input_shape=(256, 256, 3), num_classes=5)
```

Build a lightweight U-Net architecture for semantic segmentation.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_shape` | tuple | (256, 256, 3) | Input image dimensions `(H, W, C)` |
| `num_classes` | int | 5 | Number of segmentation classes |

#### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `model` | tf.keras.Model | Compiled U-Net model with ~1.93M parameters |

#### Architecture Details

```
Input (H, W, 3)
    │
    ├── Encoder ────────────────────────────────
    │   Level 1: Conv(32) → Conv(32) → Pool
    │   Level 2: Conv(64) → Conv(64) → Pool
    │   Level 3: Conv(128) → Conv(128) → Pool
    │   Level 4: Conv(256) → Conv(256) [Bottleneck]
    │
    ├── Decoder ────────────────────────────────
    │   UpConv(128) → Concat → Conv(128) → Conv(128)
    │   UpConv(64) → Concat → Conv(64) → Conv(64)
    │   UpConv(32) → Concat → Conv(32) → Conv(32)
    │
    └── Output ─────────────────────────────────
        Conv(num_classes) → Logits (H, W, num_classes)
```

**Example:**
```python
model = build_unet_lite(input_shape=(128, 128, 3), num_classes=5)

# Model summary
print(f"Parameters: {model.count_params():,}")  # ~1,929,957

# Forward pass
images = tf.random.normal([2, 128, 128, 3])
logits = model(images)  # Shape: (2, 128, 128, 5)
```

---

### build_deeplabv3_lite

```python
def build_deeplabv3_lite(input_shape=(256, 256, 3), num_classes=5)
```

Build a lightweight DeepLabV3-style model with ASPP (Atrous Spatial Pyramid Pooling).

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_shape` | tuple | (256, 256, 3) | Input image dimensions |
| `num_classes` | int | 5 | Number of segmentation classes |

#### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `model` | tf.keras.Model | Compiled DeepLabV3 model with ~9.4M parameters |

#### ASPP Module

The ASPP (Atrous Spatial Pyramid Pooling) captures multi-scale context:

```
Input Feature Map
    │
    ├── 1x1 Conv ────────────┐
    ├── 3x3 Conv (rate=6)  ──┤
    ├── 3x3 Conv (rate=12) ──┼──→ Concatenate → 1x1 Conv → Output
    ├── 3x3 Conv (rate=18) ──┤
    └── Global Avg Pool ─────┘
```

---

## Dataset Classes

### SyntheticRemoteSensingDataset

```python
class SyntheticRemoteSensingDataset
```

Generates synthetic remote sensing images with segmentation masks using Voronoi tessellation.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_samples` | int | 500 | Number of samples to generate |
| `img_size` | int | 256 | Image dimensions (square) |
| `num_classes` | int | 5 | Number of land cover classes |

#### Class Labels

| Class ID | Name | Color (RGB) |
|----------|------|-------------|
| 0 | Forest | (34, 139, 34) |
| 1 | Urban | (210, 180, 140) |
| 2 | Water | (65, 105, 225) |
| 3 | Agricultural | (240, 230, 140) |
| 4 | Bare Soil | (139, 69, 19) |

#### Methods

##### `generate_sample()`

Generate a single synthetic sample.

**Returns:**
- `image` (np.ndarray): Image with shape `(H, W, 3)`, normalized to [-1, 1]
- `mask` (np.ndarray): Segmentation mask with shape `(H, W)`

---

### create_tf_dataset

```python
def create_tf_dataset(base_dataset, num_points_per_class=5, point_strategy='random',
                      batch_size=16, shuffle=True, include_full_mask=True)
```

Create a tf.data.Dataset pipeline from a base dataset.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_dataset` | SyntheticRemoteSensingDataset | required | Source dataset |
| `num_points_per_class` | int | 5 | Points per class for labels |
| `point_strategy` | str | 'random' | Sampling strategy |
| `batch_size` | int | 16 | Batch size |
| `shuffle` | bool | True | Whether to shuffle data |
| `include_full_mask` | bool | True | Include full masks for evaluation |

#### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `dataset` | tf.data.Dataset | Dataset yielding (images, point_masks, full_masks) |

**Yields:**
- `images`: `tf.Tensor` of shape `(B, H, W, 3)`, dtype=float32
- `point_masks`: `tf.Tensor` of shape `(B, H, W)`, dtype=int32
- `full_masks`: `tf.Tensor` of shape `(B, H, W)`, dtype=int32

**Example:**
```python
train_base = SyntheticRemoteSensingDataset(num_samples=100, img_size=128)
train_dataset = create_tf_dataset(train_base, num_points_per_class=5, batch_size=16)

for images, point_masks, full_masks in train_dataset:
    # images: (16, 128, 128, 3)
    # point_masks: (16, 128, 128) - sparse labels
    # full_masks: (16, 128, 128) - complete masks
    pass
```

---

## Training Utilities

### SegmentationTrainer

```python
class SegmentationTrainer
```

Custom trainer class for segmentation models with partial labels.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | tf.keras.Model | required | Model to train |
| `num_classes` | int | 5 | Number of segmentation classes |
| `learning_rate` | float | 0.001 | Adam optimizer learning rate |
| `use_point_labels` | bool | True | Use partial CE loss if True |

#### Methods

##### `train_step(batch)`

Execute a single training step.

**Parameters:**
- `batch`: Tuple of (images, point_masks, full_masks)

**Returns:**
- `loss` (tf.Tensor): Loss value for this step

---

##### `validate(dataset, full_loss_fn=None)`

Validate the model on a dataset.

**Parameters:**
- `dataset`: tf.data.Dataset to validate on
- `full_loss_fn`: Loss function for full mask evaluation (default: SparseCategoricalCrossentropy)

**Returns:**
```python
{
    'loss': float,           # Average validation loss
    'pixel_accuracy': float, # Pixel-wise accuracy (0-1)
    'miou': float           # Mean Intersection over Union
}
```

---

##### `fit(train_dataset, val_dataset, epochs=10, verbose=True)`

Train the model.

**Parameters:**
- `train_dataset`: Training tf.data.Dataset
- `val_dataset`: Validation tf.data.Dataset
- `epochs`: Number of training epochs
- `verbose`: Print progress

**Returns:**
- `history` (dict): Training history with keys:
  - `'train_loss'`: List of training losses per epoch
  - `'val_loss'`: List of validation losses per epoch
  - `'val_miou'`: List of validation mIoU per epoch
  - `'val_acc'`: List of validation accuracy per epoch
- `best_miou` (float): Best mIoU achieved during training

**Example:**
```python
model = build_unet_lite(input_shape=(128, 128, 3), num_classes=5)
trainer = SegmentationTrainer(model, num_classes=5, learning_rate=0.001)

history, best_miou = trainer.fit(train_dataset, val_dataset, epochs=10)

print(f"Best mIoU: {best_miou:.4f}")
plt.plot(history['val_miou'])
plt.show()
```

---

## Metrics

### compute_iou

```python
def compute_iou(y_true, y_pred, num_classes)
```

Compute Intersection over Union for each class.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | np.ndarray | Ground truth labels, shape `(H, W)` |
| `y_pred` | np.ndarray | Predicted labels, shape `(H, W)` |
| `num_classes` | int | Number of classes |

#### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `iou_per_class` | np.ndarray | IoU for each class, shape `(num_classes,)` |
| `mean_iou` | float | Mean IoU across all classes |

**Formula:**
```
IoU(class) = |Prediction ∩ GroundTruth| / |Prediction ∪ GroundTruth|
mIoU = mean(IoU per class)
```

---

## Visualization

### visualize_point_labels

```python
def visualize_point_labels(image, full_mask, point_mask, title="Point Annotations")
```

Visualize point labels overlaid on the original image.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | np.ndarray or tf.Tensor | Original image `(H, W, 3)` |
| `full_mask` | np.ndarray or tf.Tensor | Full segmentation mask `(H, W)` |
| `point_mask` | np.ndarray or tf.Tensor | Point labels `(H, W)` |
| `title` | str | Plot title |

**Displays:** 3-column figure with original image, full mask, and point annotations.

---

### visualize_predictions

```python
def visualize_predictions(model, dataset, num_samples=3)
```

Visualize model predictions on validation samples.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | tf.keras.Model | Trained model |
| `dataset` | tf.data.Dataset | Dataset to visualize |
| `num_samples` | int | Number of samples to display |

**Displays:** 4-column figure per sample showing:
1. Original image
2. Point labels overlay
3. Ground truth mask
4. Model prediction

---

### plot_training_history

```python
def plot_training_history(histories, labels, metric='val_miou')
```

Plot training curves for comparison.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `histories` | list | List of history dictionaries |
| `labels` | list | Legend labels for each history |
| `metric` | str | Metric to plot ('val_miou', 'val_loss', etc.) |

---

## Constants

### Default Class Colors

```python
CLASS_COLORS = {
    0: [34, 139, 34],    # Forest - green
    1: [210, 180, 140],  # Urban - tan
    2: [65, 105, 225],   # Water - blue
    3: [240, 230, 140],  # Agricultural - yellow
    4: [139, 69, 19],    # Bare soil - brown
}
```

### Default Hyperparameters

```python
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 16
DEFAULT_IMAGE_SIZE = 128
DEFAULT_NUM_CLASSES = 5
DEFAULT_NUM_POINTS = 5
DEFAULT_EPOCHS = 10
```

---

## Error Handling

### Common Exceptions

| Exception | Cause | Solution |
|-----------|-------|----------|
| `ValueError` | Invalid input shape | Ensure images have shape `(B, H, W, 3)` |
| `tf.errors.InvalidArgumentError` | Negative dimensions | Check that `num_points_per_class > 0` |
| `AttributeError` | Missing model method | Ensure model returns logits, not probabilities |

---

## Version Compatibility

| Package | Minimum Version | Tested Version |
|---------|-----------------|----------------|
| TensorFlow | 2.10.0 | 2.21.0 |
| NumPy | 1.20.0 | 1.26.0 |
| Matplotlib | 3.5.0 | 3.8.0 |
| SciPy | 1.7.0 | 1.12.0 |

---

*Last updated: March 2026*
