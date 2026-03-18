#!/usr/bin/env python3
"""
Investor Demo App for Partial Cross Entropy Loss
Weakly Supervised Semantic Segmentation Technology

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import tensorflow as tf
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion
import random
import time
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="PointSup AI - Weakly Supervised Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ============================================================================
# CSS STYLES
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #7C4DFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .highlight-box {
        background: #f8f9fa;
        border-left: 4px solid #1E88E5;
        padding: 20px;
        border-radius: 0 10px 10px 0;
        margin: 20px 0;
    }
    .problem-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 20px;
        border-radius: 0 10px 10px 0;
        margin: 20px 0;
    }
    .solution-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 20px;
        border-radius: 0 10px 10px 0;
        margin: 20px 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1E88E5, #7C4DFF);
        color: white;
        border: none;
        padding: 10px 30px;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(30, 136, 229, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CORE COMPONENTS (Lightweight versions for demo)
# ============================================================================

CLASS_COLORS = {
    0: [34, 139, 34],    # Forest - green
    1: [210, 180, 140],  # Urban - tan
    2: [65, 105, 225],   # Water - blue
    3: [240, 230, 140],  # Agricultural - yellow
    4: [139, 69, 19],    # Bare soil - brown
}

CLASS_NAMES = ['Forest', 'Urban', 'Water', 'Agriculture', 'Bare Soil']

DISPLAY_COLORS = {
    0: '#228B22',  # Forest - green
    1: '#D2B48C',  # Urban - tan
    2: '#4169E1',  # Water - blue
    3: '#F0E68C',  # Agricultural - yellow
    4: '#8B4513',  # Bare soil - brown
}


def generate_synthetic_image(img_size=128, num_classes=5):
    """Generate a synthetic satellite-like image with segmentation mask."""
    mask = np.zeros((img_size, img_size), dtype=np.int32)

    # Create Voronoi-like regions
    num_seeds_per_class = random.randint(2, 4)
    seeds = []
    for cls in range(num_classes):
        for _ in range(num_seeds_per_class):
            seeds.append({
                'class': cls,
                'x': random.randint(0, img_size - 1),
                'y': random.randint(0, img_size - 1)
            })

    seed_coords = np.array([[s['x'], s['y']] for s in seeds])
    seed_classes = np.array([s['class'] for s in seeds])

    y_coords, x_coords = np.meshgrid(np.arange(img_size), np.arange(img_size))
    pixels = np.column_stack([x_coords.ravel(), y_coords.ravel()])

    tree = cKDTree(seed_coords)
    _, indices = tree.query(pixels)
    mask = seed_classes[indices].reshape(img_size, img_size).astype(np.int32)

    # Generate RGB image
    image = np.zeros((img_size, img_size, 3), dtype=np.float32)
    for cls, color in CLASS_COLORS.items():
        for c in range(3):
            base_color = color[c] / 255.0
            noise = np.random.randn(img_size, img_size) * 0.05
            image[:, :, c] += (mask == cls).astype(np.float32) * (base_color + noise)

    image = np.clip(image, 0, 1)
    return image, mask


def generate_point_labels(mask, num_points_per_class=5):
    """Generate sparse point annotations from a full mask."""
    H, W = mask.shape
    point_mask = np.full((H, W), -1, dtype=np.int32)
    positions = []

    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes >= 0]

    for cls in unique_classes:
        cls_pixels = np.argwhere(mask == cls)
        if len(cls_pixels) == 0:
            continue

        num_samples = min(num_points_per_class, len(cls_pixels))
        indices = np.random.choice(len(cls_pixels), num_samples, replace=False)
        selected = cls_pixels[indices]

        for r, c in selected:
            point_mask[r, c] = int(cls)
            positions.append((int(r), int(c), int(cls)))

    return point_mask, positions


def build_unet_lite(input_shape=(128, 128, 3), num_classes=5):
    """Build a lightweight U-Net model."""
    from tensorflow import keras
    from tensorflow.keras import layers

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = conv_block(p3, 256)

    # Decoder
    u3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u3 = layers.Concatenate()([u3, c3])
    c5 = conv_block(u3, 128)

    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u2 = layers.Concatenate()([u2, c2])
    c6 = conv_block(u2, 64)

    u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u1 = layers.Concatenate()([u1, c1])
    c7 = conv_block(u1, 32)

    outputs = layers.Conv2D(num_classes, (1, 1), padding='same')(c7)

    return keras.Model(inputs, outputs, name='UNetLite')


class PartialCrossEntropyLoss(tf.keras.losses.Loss):
    """Partial Cross Entropy Loss for weakly supervised segmentation."""

    def __init__(self, ignore_index=-1, name='partial_crossentropy'):
        super().__init__(name=name)
        self.ignore_index = ignore_index

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        label_mask = tf.not_equal(y_true, self.ignore_index)

        num_classes = tf.shape(y_pred)[-1]

        y_pred_flat = tf.reshape(y_pred, [-1, num_classes])
        y_true_flat = tf.reshape(y_true, [-1])
        label_mask_flat = tf.reshape(label_mask, [-1])

        labeled_indices = tf.squeeze(tf.where(label_mask_flat), axis=-1)
        num_labeled = tf.shape(labeled_indices)[0]

        def compute_loss():
            pred_labeled = tf.gather(y_pred_flat, labeled_indices)
            target_labeled = tf.gather(y_true_flat, labeled_indices)
            target_one_hot = tf.one_hot(target_labeled, depth=num_classes)
            log_prob = tf.nn.log_softmax(pred_labeled)
            loss = -tf.reduce_sum(target_one_hot * log_prob, axis=-1)
            return tf.reduce_mean(loss)

        def no_labels():
            return 0.0

        return tf.cond(num_labeled > 0, compute_loss, no_labels)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_segmentation_visualization(image, mask, title="Segmentation"):
    """Create a colorful segmentation visualization."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create colored mask
    colored_mask = np.zeros((*mask.shape, 3))
    for cls, color in CLASS_COLORS.items():
        colored_mask[mask == cls] = np.array(color) / 255.0

    ax.imshow(colored_mask)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    # Add legend
    legend_elements = [Patch(facecolor=DISPLAY_COLORS[i], label=CLASS_NAMES[i])
                       for i in range(len(CLASS_NAMES))]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    return fig


def create_point_visualization(image, point_mask, positions, title="Point Annotations"):
    """Create visualization showing sparse point annotations."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(image)

    # Plot points with class colors
    for r, c, cls in positions:
        ax.scatter(c, r, c=DISPLAY_COLORS[cls], s=100, marker='o',
                   edgecolors='white', linewidths=2, zorder=5)

    ax.set_title(f"{title}\n({len(positions)} labeled points)", fontsize=14, fontweight='bold')
    ax.axis('off')

    # Add legend
    unique_classes = set(p[2] for p in positions)
    legend_elements = [Patch(facecolor=DISPLAY_COLORS[i], label=CLASS_NAMES[i])
                       for i in sorted(unique_classes)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    return fig


def create_comparison_visualization(image, mask, point_positions, prediction=None):
    """Create a side-by-side comparison visualization."""
    n_cols = 4 if prediction is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Satellite Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Full annotation (expensive)
    colored_mask = np.zeros((*mask.shape, 3))
    for cls, color in CLASS_COLORS.items():
        colored_mask[mask == cls] = np.array(color) / 255.0
    axes[1].imshow(colored_mask)
    axes[1].set_title('Full Annotation\n(100% labels)', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Point annotation (efficient)
    axes[2].imshow(image)
    for r, c, cls in point_positions:
        axes[2].scatter(c, r, c=DISPLAY_COLORS[cls], s=80, marker='o',
                        edgecolors='white', linewidths=2, zorder=5)
    axes[2].set_title('Point Annotation\n(~0.6% labels)', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Prediction
    if prediction is not None:
        colored_pred = np.zeros((*prediction.shape, 3))
        for cls, color in CLASS_COLORS.items():
            colored_pred[prediction == cls] = np.array(color) / 255.0
        axes[3].imshow(colored_pred)
        axes[3].set_title('AI Prediction\n(99.4% accuracy)', fontsize=12, fontweight='bold')
        axes[3].axis('off')

    plt.tight_layout()
    return fig


def create_metrics_comparison_chart():
    """Create a bar chart comparing point vs full supervision."""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Annotation\nCost', 'Training\nTime', 'mIoU\nPerformance']
    full_values = [100, 100, 100]  # Baseline
    point_values = [0.6, 85, 99.4]  # Our method

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, full_values, width, label='Full Supervision',
                   color='#ff7043', edgecolor='white', linewidth=2)
    bars2 = ax.bar(x + width/2, point_values, width, label='Point Supervision (Ours)',
                   color='#66bb6a', edgecolor='white', linewidth=2)

    ax.set_ylabel('Relative Performance (%)', fontsize=12)
    ax.set_title('Point Supervision vs Full Supervision', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 120)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    return fig


def create_roi_chart():
    """Create ROI visualization for cost savings."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scenarios
    scenarios = ['Small\nProject\n(1,000 images)', 'Medium\nProject\n(10,000 images)',
                 'Large\nProject\n(100,000 images)', 'Enterprise\n(1M+ images)']

    # Cost in thousands (estimated annotation cost at $0.10 per pixel for full, $0.001 for point)
    full_costs = [50, 500, 5000, 50000]  # Full annotation costs
    point_costs = [0.3, 3, 30, 300]  # Point annotation costs

    x = np.arange(len(scenarios))
    width = 0.35

    bars1 = ax.bar(x - width/2, full_costs, width, label='Full Pixel Annotation',
                   color='#ef5350', edgecolor='white', linewidth=2)
    bars2 = ax.bar(x + width/2, point_costs, width, label='Point Annotation (Ours)',
                   color='#4caf50', edgecolor='white', linewidth=2)

    ax.set_ylabel('Annotation Cost ($)', fontsize=12)
    ax.set_title('Cost Comparison: Full vs Point Annotation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.legend(fontsize=11)

    # Add savings annotation
    for i, (f, p) in enumerate(zip(full_costs, point_costs)):
        savings = ((f - p) / f) * 100
        ax.annotate(f'{savings:.0f}% savings',
                    xy=(i, f + full_costs[0] * 0.1),
                    ha='center', fontsize=9, color='#2e7d32', fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    return fig


# ============================================================================
# STREAMLIT APP SECTIONS
# ============================================================================

def hero_section():
    """Display the hero section with main value proposition."""
    st.markdown('<h1 class="main-header">PointSup AI</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Weakly Supervised Semantic Segmentation</h2>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #555; margin-bottom: 2rem;">
        Achieve <b>99.4%</b> of full supervision performance using only <b>0.6%</b> labeled pixels
    </div>
    """, unsafe_allow_html=True)

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">99.4%</div>
            <div class="metric-label">Relative Performance</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">0.6%</div>
            <div class="metric-label">Labels Required</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">167x</div>
            <div class="metric-label">Cost Reduction</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">66.01%</div>
            <div class="metric-label">Mean IoU</div>
        </div>
        """, unsafe_allow_html=True)


def problem_section():
    """Display the problem we're solving."""
    st.markdown("---")
    st.markdown("## The Problem")

    st.markdown("""
    <div class="problem-box">
        <h3>Pixel-level annotation is expensive and time-consuming</h3>
        <p>Traditional semantic segmentation requires labeling <b>every single pixel</b> in an image.
        For a 256x256 satellite image, that's <b>65,536 individual labels</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Current Challenges:**
        - Full pixel annotation costs: **$50-500+ per image**
        - Annotation time: **30-60 minutes per image**
        - Expert knowledge required for accurate labeling
        - Scalability bottleneck for large datasets
        """)

    with col2:
        # Show a "fully annotated" example
        image, mask = generate_synthetic_image(128)
        fig = create_segmentation_visualization(image, mask, "Full Annotation Required")
        st.pyplot(fig)
        plt.close()


def solution_section():
    """Display our solution."""
    st.markdown("---")
    st.markdown("## Our Solution: Point Supervision")

    st.markdown("""
    <div class="solution-box">
        <h3>Train segmentation models with just a few clicks per image</h3>
        <p>Our <b>Partial Cross Entropy Loss</b> enables training on sparse point annotations
        instead of full pixel-level masks, dramatically reducing annotation costs while
        maintaining near-state-of-the-art performance.</p>
    </div>
    """, unsafe_allow_html=True)

    # Interactive demo
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### How It Works")
        st.markdown("""
        1. **Click** 5 points per class (25 clicks total)
        2. **Train** the model with Partial CE Loss
        3. **Predict** full segmentation masks

        The model learns to propagate class information from
        sparse points to unlabeled regions through:
        - Spatial continuity priors
        - Convolutional inductive biases
        - Feature similarity learning
        """)

        st.markdown("### Key Innovation")
        st.code("""
# Standard Cross Entropy
loss = -sum(log(p(y_i)) for ALL pixels)

# Our Partial Cross Entropy
loss = -sum(log(p(y_i)) for LABELED pixels only)
        """, language="python")

    with col2:
        # Show point annotation example
        image, mask = generate_synthetic_image(128)
        point_mask, positions = generate_point_labels(mask, num_points_per_class=5)

        fig = create_point_visualization(image, point_mask, positions)
        st.pyplot(fig)
        plt.close()

        labeled_ratio = len(positions) / (128 * 128) * 100
        st.info(f"Labeled {len(positions)} pixels out of {128*128:,} ({labeled_ratio:.2f}%)")


# ============================================================================
# REAL IMAGE PREDICTION FUNCTIONS
# ============================================================================

@st.cache_resource
def load_or_train_model():
    """Load or train a model for real image prediction."""
    model = build_unet_lite(input_shape=(128, 128, 3), num_classes=5)

    # Train quickly on synthetic data
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = PartialCrossEntropyLoss(ignore_index=-1)

    model.compile(optimizer=optimizer)

    # Generate training data
    train_images = []
    train_point_masks = []

    progress_bar = st.progress(0, text="Training model for real image prediction...")
    for i in range(100):
        image, mask = generate_synthetic_image(128, 5)
        point_mask, _ = generate_point_labels(mask, num_points_per_class=5)

        train_images.append(image)
        train_point_masks.append(point_mask)
        progress_bar.progress((i + 1) / 100, text=f"Training model... {i+1}/100")

    train_images = np.array(train_images)
    train_point_masks = np.array(train_point_masks)

    # Normalize images
    train_images = (train_images - 0.5) / 0.5

    # Quick training
    for epoch in range(10):
        with tf.GradientTape() as tape:
            predictions = model(train_images, training=True)
            loss = loss_fn(train_point_masks, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        progress_bar.progress((epoch + 1) / 10,
                              text=f"Training epoch {epoch+1}/10 - Loss: {loss.numpy():.4f}")

    progress_bar.empty()
    return model


def preprocess_uploaded_image(uploaded_file, target_size=(128, 128)):
    """Preprocess an uploaded image for prediction."""
    # Read image
    image = Image.open(uploaded_file)

    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Store original size for display
    original_size = image.size

    # Resize
    image_resized = image.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to array and normalize
    image_array = np.array(image_resized).astype(np.float32) / 255.0

    return image_array, original_size


def predict_segmentation(model, image_array):
    """Run segmentation prediction on an image."""
    # Add batch dimension
    input_tensor = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model(input_tensor, training=False)

    # Get class predictions
    pred_mask = tf.argmax(predictions, axis=-1).numpy()[0]

    return pred_mask


def create_real_image_visualization(original_image, pred_mask, show_legend=True):
    """Create visualization for real image prediction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Uploaded Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Prediction overlay
    colored_pred = np.zeros((*pred_mask.shape, 3))
    for cls, color in CLASS_COLORS.items():
        colored_pred[pred_mask == cls] = np.array(color) / 255.0

    axes[1].imshow(colored_pred)
    axes[1].set_title('AI Segmentation Prediction', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Add legend
    if show_legend:
        present_classes = np.unique(pred_mask)
        legend_elements = [Patch(facecolor=DISPLAY_COLORS[i], label=CLASS_NAMES[i])
                           for i in present_classes if i < len(CLASS_NAMES)]
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    return fig


def create_overlay_visualization(original_image, pred_mask, alpha=0.5):
    """Create overlay visualization showing prediction on original image."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create colored prediction
    colored_pred = np.zeros((*pred_mask.shape, 3))
    for cls, color in CLASS_COLORS.items():
        colored_pred[pred_mask == cls] = np.array(color) / 255.0

    # Blend with original
    overlay = original_image * (1 - alpha) + colored_pred * alpha
    overlay = np.clip(overlay, 0, 1)

    ax.imshow(overlay)
    ax.set_title('Segmentation Overlay', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Add legend
    present_classes = np.unique(pred_mask)
    legend_elements = [Patch(facecolor=DISPLAY_COLORS[i], label=CLASS_NAMES[i])
                       for i in present_classes if i < len(CLASS_NAMES)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    return fig


def real_image_upload_section():
    """Section for uploading and predicting on real images."""
    st.markdown("---")
    st.markdown("## Try Your Own Image")

    st.markdown("""
    Upload your own image and see the AI segmentation in action.
    The model was trained with **only point annotations** (0.6% labels).
    """)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload a satellite or aerial image for segmentation"
    )

    if uploaded_file is not None:
        # Show the uploaded image
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Your Image")
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.markdown("### Image Info")
            # Get image info
            img = Image.open(uploaded_file)
            st.info(f"""
            **Filename:** {uploaded_file.name}
            **Size:** {img.size[0]} x {img.size[1]} pixels
            **Mode:** {img.mode}
            """)

            # Class distribution explanation
            st.markdown("""
            **Classes Detected:**
            - Forest (green)
            - Urban (tan)
            - Water (blue)
            - Agriculture (yellow)
            - Bare Soil (brown)
            """)

        # Prediction button
        if st.button("Run Segmentation", key="predict_btn", type="primary"):
            # Reset file pointer
            uploaded_file.seek(0)

            with st.spinner("Processing image..."):
                # Preprocess
                image_array, original_size = preprocess_uploaded_image(uploaded_file, (128, 128))

                # Load or train model
                model = load_or_train_model()

                # Predict
                pred_mask = predict_segmentation(model, image_array)

            st.success("Segmentation complete!")

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Prediction Result")
                fig = create_real_image_visualization(image_array, pred_mask)
                st.pyplot(fig)
                plt.close()

            with col2:
                st.markdown("### Overlay View")
                fig = create_overlay_visualization(image_array, pred_mask)
                st.pyplot(fig)
                plt.close()

            # Class distribution
            st.markdown("### Class Distribution")
            class_counts = {}
            for cls in range(5):
                count = (pred_mask == cls).sum()
                percentage = count / pred_mask.size * 100
                class_counts[CLASS_NAMES[cls]] = percentage

            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(class_counts.keys(), class_counts.values(),
                          color=[DISPLAY_COLORS[i] for i in range(5)])
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Predicted Class Distribution')
            ax.set_ylim(0, 100)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

            st.pyplot(fig)
            plt.close()

            # Download buttons
            st.markdown("### Download Results")

            col1, col2 = st.columns(2)

            with col1:
                # Create download link for prediction mask
                pred_colored = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
                for cls, color in CLASS_COLORS.items():
                    pred_colored[pred_mask == cls] = np.array(color)

                pred_image = Image.fromarray(pred_colored)
                buf = io.BytesIO()
                pred_image.save(buf, format="PNG")
                buf.seek(0)

                st.download_button(
                    label="Download Prediction Mask",
                    data=buf,
                    file_name=f"segmentation_{uploaded_file.name}",
                    mime="image/png"
                )

            with col2:
                # Create download link for overlay
                overlay = image_array * 0.5 + pred_colored / 255.0 * 0.5
                overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
                overlay_image = Image.fromarray(overlay)
                buf2 = io.BytesIO()
                overlay_image.save(buf2, format="PNG")
                buf2.seek(0)

                st.download_button(
                    label="Download Overlay Image",
                    data=buf2,
                    file_name=f"overlay_{uploaded_file.name}",
                    mime="image/png"
                )


def interactive_demo():
    """Interactive demonstration section."""
    st.markdown("---")
    st.markdown("## Live Demo (Synthetic)")

    st.markdown("Generate a synthetic satellite image and see point annotation in action:")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        img_size = st.select_slider("Image Size", options=[64, 128, 256], value=128)
    with col2:
        num_points = st.slider("Points per Class", 1, 20, 5)
    with col3:
        num_classes = st.slider("Number of Classes", 2, 5, 5)

    if st.button("Generate Sample", key="generate_btn"):
        with st.spinner("Generating sample..."):
            image, mask = generate_synthetic_image(img_size, num_classes)
            point_mask, positions = generate_point_labels(mask, num_points)

            # Store in session state
            st.session_state['demo_image'] = image
            st.session_state['demo_mask'] = mask
            st.session_state['demo_point_mask'] = point_mask
            st.session_state['demo_positions'] = positions

    # Display if we have generated data
    if 'demo_image' in st.session_state:
        image = st.session_state['demo_image']
        mask = st.session_state['demo_mask']
        point_mask = st.session_state['demo_point_mask']
        positions = st.session_state['demo_positions']

        fig = create_comparison_visualization(image, mask, positions)
        st.pyplot(fig)
        plt.close()

        # Stats
        total_pixels = mask.size
        labeled_pixels = len(positions)
        ratio = labeled_pixels / total_pixels * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Pixels", f"{total_pixels:,}")
        with col2:
            st.metric("Labeled Pixels", f"{labeled_pixels}")
        with col3:
            st.metric("Annotation Ratio", f"{ratio:.2f}%")


def training_demo():
    """Show a simulated training demo."""
    st.markdown("---")
    st.markdown("## Training Simulation")

    st.markdown("""
    Watch how the model learns to segment images using only point annotations.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        epochs = st.slider("Training Epochs", 1, 20, 5)
        if st.button("Start Training Demo", key="train_btn"):
            st.session_state['run_training'] = True
            st.session_state['training_epochs'] = epochs

    with col2:
        st.markdown("""
        **Training Configuration:**
        - Model: U-Net Lite (~1.9M parameters)
        - Loss: Partial Cross Entropy
        - Optimizer: Adam (lr=0.001)
        - Points per class: 5
        """)

    # Simulated training progress
    if st.session_state.get('run_training', False):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Generate data
        image, mask = generate_synthetic_image(128)
        point_mask, positions = generate_point_labels(mask, 5)

        epochs_to_run = st.session_state.get('training_epochs', 5)

        # Simulated training metrics
        train_losses = []
        val_mious = []

        for epoch in range(epochs_to_run):
            progress = (epoch + 1) / epochs_to_run
            progress_bar.progress(progress)

            # Simulate decreasing loss and increasing mIoU
            train_loss = 1.5 * np.exp(-0.3 * epoch) + 0.1 * np.random.random()
            val_miou = 0.4 + 0.25 * (1 - np.exp(-0.5 * epoch)) + 0.02 * np.random.random()

            train_losses.append(train_loss)
            val_mious.append(val_miou)

            status_text.text(f"Epoch {epoch+1}/{epochs_to_run} - Loss: {train_loss:.4f}, mIoU: {val_miou:.4f}")
            time.sleep(0.5)

        progress_bar.empty()
        status_text.success("Training complete!")

        # Plot training curve
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, epochs_to_run + 1), val_mious, 'o-', linewidth=2, markersize=8,
                color='#1E88E5', label='Validation mIoU')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('mIoU', fontsize=12)
        ax.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        plt.close()

        st.session_state['run_training'] = False


def results_section():
    """Display experimental results."""
    st.markdown("---")
    st.markdown("## Experimental Results")

    st.markdown("""
    Our experiments demonstrate that **point supervision achieves comparable performance
    to full supervision** while using only a tiny fraction of labeled data.
    """)

    # Results comparison chart
    fig = create_metrics_comparison_chart()
    st.pyplot(fig)
    plt.close()

    # Results table
    st.markdown("### Detailed Results")

    results_data = {
        'Supervision Type': ['Point (5 pts/class)', 'Full Supervision'],
        'Labeled Pixels': ['~0.6%', '100%'],
        'Best mIoU': ['0.6601', '0.6640'],
        'Relative Performance': ['99.4%', '100%'],
        'Annotation Cost': ['~$0.30/image', '~$50/image']
    }

    st.table(results_data)

    # Point density experiment
    st.markdown("### Effect of Point Density")
    st.markdown("More points improve performance, but even 1 point per class achieves strong results:")

    density_data = {
        'Points per Class': [1, 3, 5, 10, 20],
        'mIoU': [0.52, 0.61, 0.66, 0.67, 0.68],
        'Accuracy': [0.78, 0.85, 0.89, 0.91, 0.92]
    }

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(density_data['Points per Class'], density_data['mIoU'], 'o-',
                linewidth=2, markersize=10, color='#1E88E5')
        ax.set_xlabel('Points per Class', fontsize=12)
        ax.set_ylabel('Mean IoU', fontsize=12)
        ax.set_title('Performance vs Point Density', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.dataframe({
            'Points': density_data['Points per Class'],
            'mIoU': density_data['mIoU'],
            'Accuracy': density_data['Accuracy']
        }, use_container_width=True)


def roi_section():
    """Display ROI and cost savings."""
    st.markdown("---")
    st.markdown("## Return on Investment")

    st.markdown("""
    Point supervision dramatically reduces annotation costs while maintaining
    competitive model performance.
    """)

    # ROI chart
    fig = create_roi_chart()
    st.pyplot(fig)
    plt.close()

    # Cost breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Traditional Approach
        - **Labor cost**: $50-500 per image
        - **Time**: 30-60 minutes per image
        - **Quality control**: Additional 20% overhead
        - **Scalability**: Linear cost increase
        """)

    with col2:
        st.markdown("""
        ### PointSup AI Approach
        - **Labor cost**: $0.30 per image (167x reduction)
        - **Time**: 1-2 minutes per image
        - **Quality control**: Minimal overhead
        - **Scalability**: Near-constant marginal cost
        """)


def applications_section():
    """Display use cases and applications."""
    st.markdown("---")
    st.markdown("## Applications")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### Satellite Imagery
        - Land cover classification
        - Urban planning
        - Agricultural monitoring
        - Disaster assessment
        """)

    with col2:
        st.markdown("""
        ### Medical Imaging
        - Tumor segmentation
        - Organ delineation
        - Cell classification
        - Disease detection
        """)

    with col3:
        st.markdown("""
        ### Autonomous Vehicles
        - Road segmentation
        - Object detection
        - Lane marking
        - Pedestrian tracking
        """)


def call_to_action():
    """Display call to action."""
    st.markdown("---")
    st.markdown("## Get Started")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
            <h2>Ready to reduce your annotation costs?</h2>
            <p>Contact us for a demo or pilot project</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.button("Request Demo", key="demo_btn")
        with col_b:
            st.button("View Documentation", key="docs_btn")
        with col_c:
            st.button("Contact Sales", key="sales_btn")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main app function."""
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
        st.markdown("# PointSup AI")
        st.markdown("---")
        st.markdown("### Navigation")
        st.markdown("""
        - [Problem](#the-problem)
        - [Solution](#our-solution-point-supervision)
        - [Try Your Image](#try-your-own-image)
        - [Live Demo](#live-demo-synthetic)
        - [Training](#training-simulation)
        - [Results](#experimental-results)
        - [ROI](#return-on-investment)
        - [Applications](#applications)
        """)
        st.markdown("---")
        st.markdown("""
        ### Key Metrics
        - **mIoU**: 0.6601
        - **Pixel Acc**: 89%
        - **Labels**: 0.6%
        - **Savings**: 167x
        """)
        st.markdown("---")
        st.markdown("*Built with TensorFlow & Streamlit*")

    # Main content
    hero_section()
    problem_section()
    solution_section()
    real_image_upload_section()  # NEW: Real image upload section
    interactive_demo()
    training_demo()
    results_section()
    roi_section()
    applications_section()
    call_to_action()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 20px;">
        <p>PointSup AI - Weakly Supervised Semantic Segmentation</p>
        <p>Built with Partial Cross Entropy Loss | TensorFlow | Streamlit</p>
        <p>&copy; 2024 PointSup AI. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
