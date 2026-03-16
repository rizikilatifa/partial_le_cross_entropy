# Quick Start Guide

Get up and running with Partial Cross Entropy Loss in 5 minutes.

## Prerequisites

```bash
# Install dependencies
pip install tensorflow numpy scipy matplotlib tqdm

# For Jupyter notebooks
pip install jupyter
```

## Option 1: Run the Notebook

```bash
# Navigate to project directory
cd partial_ce_loss_project

# Start Jupyter
jupyter notebook

# Open: partial_ce_loss_segmentation_tf.ipynb
# Run all cells (Cell → Run All)
```

## Option 2: Use in Your Own Code

### Minimal Example

```python
import tensorflow as tf
import numpy as np

# 1. Import the loss function
from partial_ce_loss import PartialCrossEntropyLoss, build_unet_lite

# 2. Create model
model = build_unet_lite(input_shape=(256, 256, 3), num_classes=5)

# 3. Prepare your data
# Images: shape (batch, H, W, 3)
# Labels: shape (batch, H, W) with -1 for unlabeled pixels
images = tf.random.normal([4, 256, 256, 3])
labels = np.full((4, 256, 256), -1, dtype=np.int32)

# Add some point labels (example: class 0 at specific locations)
labels[0, 100, 100] = 0
labels[0, 150, 150] = 1
labels[1, 50, 50] = 2

labels = tf.constant(labels)

# 4. Training step
loss_fn = PartialCrossEntropyLoss(ignore_index=-1)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_fn(labels, predictions)

gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))

print(f"Loss: {loss.numpy():.4f}")
```

### Full Training Loop

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

class PartialCrossEntropyLoss(keras.losses.Loss):
    def __init__(self, ignore_index=-1, from_logits=True, name='partial_ce'):
        super().__init__(name=name)
        self.ignore_index = ignore_index
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        label_mask = tf.not_equal(y_true, self.ignore_index)

        num_classes = tf.shape(y_pred)[-1]
        y_pred_flat = tf.reshape(y_pred, [-1, num_classes])
        y_true_flat = tf.reshape(y_true, [-1])
        label_mask_flat = tf.reshape(label_mask, [-1])

        labeled_indices = tf.squeeze(tf.where(label_mask_flat), axis=-1)

        def compute_loss():
            pred_labeled = tf.gather(y_pred_flat, labeled_indices)
            target_labeled = tf.gather(y_true_flat, labeled_indices)
            target_one_hot = tf.one_hot(target_labeled, depth=num_classes)
            log_prob = tf.nn.log_softmax(pred_labeled)
            loss = -tf.reduce_sum(target_one_hot * log_prob, axis=-1)
            return tf.reduce_mean(loss)

        return tf.cond(tf.size(labeled_indices) > 0, compute_loss, lambda: 0.0)


def train_model(train_images, point_labels, val_images, val_labels,
                num_classes=5, epochs=10, batch_size=8):
    """Train a segmentation model with partial labels."""

    # Build model
    inputs = keras.Input(shape=train_images.shape[1:])
    # ... add your model layers here ...
    outputs = keras.layers.Conv2D(num_classes, 1)(inputs)  # logits
    model = keras.Model(inputs, outputs)

    # Setup training
    loss_fn = PartialCrossEntropyLoss(ignore_index=-1)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    # Training loop
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(train_images))
        epoch_loss = 0.0

        for i in range(0, len(train_images), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_images = train_images[batch_idx]
            batch_labels = point_labels[batch_idx]

            with tf.GradientTape() as tape:
                predictions = model(batch_images, training=True)
                loss = loss_fn(batch_labels, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += loss.numpy()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/(len(train_images)//batch_size):.4f}")

    return model


# Usage example
if __name__ == "__main__":
    # Generate dummy data
    train_images = np.random.randn(100, 128, 128, 3).astype(np.float32)
    val_images = np.random.randn(20, 128, 128, 3).astype(np.float32)

    # Create sparse labels (most pixels are -1 / unlabeled)
    train_labels = np.full((100, 128, 128), -1, dtype=np.int32)
    val_labels = np.random.randint(0, 5, (20, 128, 128)).astype(np.int32)

    # Add some point annotations (5 points per image)
    for i in range(100):
        for _ in range(5):
            r, c = np.random.randint(0, 128, 2)
            train_labels[i, r, c] = np.random.randint(0, 5)

    # Train
    model = train_model(train_images, train_labels, val_images, val_labels)

    # Predict
    predictions = model.predict(val_images)
    predicted_classes = np.argmax(predictions, axis=-1)
    print(f"Predictions shape: {predicted_classes.shape}")
```

## Convert Your Data to Point Labels

```python
def mask_to_points(full_mask, num_points_per_class=5):
    """Convert full segmentation mask to point annotations."""
    H, W = full_mask.shape
    point_mask = np.full((H, W), -1, dtype=np.int32)

    for cls in np.unique(full_mask):
        if cls < 0:
            continue

        # Find all pixels of this class
        cls_pixels = np.argwhere(full_mask == cls)
        if len(cls_pixels) == 0:
            continue

        # Random sample
        n = min(num_points_per_class, len(cls_pixels))
        indices = np.random.choice(len(cls_pixels), n, replace=False)
        selected = cls_pixels[indices]

        for r, c in selected:
            point_mask[r, c] = cls

    return point_mask

# Usage
full_mask = np.array(your_full_annotation)  # Shape: (H, W)
point_mask = mask_to_points(full_mask, num_points_per_class=5)

# Labeled pixel ratio
labeled_ratio = (point_mask != -1).sum() / point_mask.size
print(f"Labeled: {labeled_ratio*100:.2f}%")
```

## Evaluate Your Model

```python
def compute_metrics(y_true, y_pred, num_classes):
    """Compute segmentation metrics."""
    y_pred = np.argmax(y_pred, axis=-1)

    # Pixel accuracy
    accuracy = (y_true == y_pred).mean()

    # mIoU
    ious = []
    for cls in range(num_classes):
        pred_cls = (y_pred == cls)
        true_cls = (y_true == cls)
        intersection = (pred_cls & true_cls).sum()
        union = (pred_cls | true_cls).sum()
        if union > 0:
            ious.append(intersection / union)

    miou = np.mean(ious)
    return {'accuracy': accuracy, 'miou': miou}

# Usage
predictions = model.predict(val_images)
metrics = compute_metrics(val_labels, predictions, num_classes=5)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"mIoU: {metrics['miou']:.4f}")
```

## Common Pitfalls

### 1. Using softmax instead of logits

```python
# ❌ Wrong: Model outputs probabilities
outputs = keras.layers.Conv2D(num_classes, 1, activation='softmax')(x)

# ✅ Correct: Model outputs logits
outputs = keras.layers.Conv2D(num_classes, 1)(x)

# Then use from_logits=True in loss
loss_fn = PartialCrossEntropyLoss(from_logits=True)
```

### 2. Wrong label format

```python
# ❌ Wrong: One-hot encoded labels
labels = np.zeros((batch, H, W, num_classes))

# ✅ Correct: Integer labels with -1 for unlabeled
labels = np.full((batch, H, W), -1, dtype=np.int32)
labels[0, 100, 100] = 2  # Class 2 at position (100, 100)
```

### 3. Forgetting to cast labels

```python
# ❌ May cause issues
loss = loss_fn(y_true, y_pred)

# ✅ Explicitly ensure int type
y_true = tf.cast(y_true, tf.int32)
loss = loss_fn(y_true, y_pred)
```

## Next Steps

1. **Read the full documentation:** See [README.md](README.md) and [API_REFERENCE.md](API_REFERENCE.md)
2. **Run experiments:** Try different `num_points_per_class` values
3. **Use your own data:** Replace synthetic dataset with real images
4. **Try different architectures:** Experiment with DeepLabV3 or other models

## Getting Help

- Check the executed notebook: `partial_ce_loss_segmentation_tf_executed.ipynb`
- Review the API reference for detailed parameter documentation
- Ensure TensorFlow version >= 2.10.0
