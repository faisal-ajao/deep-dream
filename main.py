"""📦 Import Required Libraries"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import utils
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.models import Model

# %matplotlib inline   # (Useful in Jupyter Notebook, not required in scripts)

"""⚙️ Parameters & Configuration"""

base_image_path = "inputs/cat.jpg"
result_prefix = "cat_dream"

# Path to input image and prefix for saving results
layer_settings = {
    "mixed0": 0.5,
    "mixed2": 1.0,
    "mixed3": 1.5,
    "mixed5": 2.0,
}

# DeepDream hyperparameters
step = 0.02  # Gradient ascent step size
iterations = 30  # Number of gradient ascent iterations per octave
num_octave = 4  # How many scales (octaves) to run
octave_scale = 1.3  # Scale factor between octaves
max_loss = 8.0  # Prevents runaway loss for stability

"""🖼️ View Base Image"""

plt.figure(figsize=(6, 6))
plt.imshow(plt.imread(base_image_path))
plt.axis("off")
plt.show()

"""🔄 Preprocessing & Deprocessing Utilities"""


def preprocess_image(image_path):
    """
    Load and preprocess the image for InceptionV3.
    - Converts to array
    - Adds batch dimension
    - Applies InceptionV3 preprocessing
    """
    img = utils.load_img(image_path)
    img = utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    """
    Convert a processed tensor back into a valid RGB image.
    - Undo InceptionV3 preprocessing
    - Clip values to [0, 255]
    """
    x = x.reshape((x.shape[1], x.shape[2], 3))
    # Undo inception v3 preprocessing
    x /= 2.0
    x += 0.5
    x *= 255.0
    x = np.clip(x, 0, 255).astype("uint8")
    return x


"""🤖 Load Pretrained Model & Feature Extractor"""

# Load InceptionV3 pretrained on ImageNet (exclude classification head)
model = inception_v3.InceptionV3(weights="imagenet", include_top=False)

# Extract outputs of specific layers for dreaming
output_dicts = dict(
    [
        (layer.name, layer.output)
        for layer in [model.get_layer(name) for name in layer_settings.keys()]
    ]
)

# Model that maps input → selected layer activations
feature_extractor = Model(inputs=model.inputs, outputs=output_dicts)

"""📉 Define the Loss Function"""


def compute_loss(input_image):
    """
    Compute DeepDream loss:
    - Forward pass through selected layers
    - Weighted sum of squared activations (excluding borders to reduce artifacts)
    """
    features = feature_extractor(input_image)
    loss = tf.zeros(shape=())  # Initialize total loss
    for name in features.keys():
        coeff = layer_settings[name]
        activation = features[name]
        scaling = tf.reduce_prod(tf.cast(tf.shape(activation), dtype="float32"))
        loss += coeff * tf.reduce_sum(tf.square(activation[:, 2:-2, 2:-2, :])) / scaling
    return loss


"""🔼 Gradient Ascent Loop (Single Octave)"""


@tf.function
def gradient_ascent_step(img, learning_rate):
    """
    Perform a single gradient ascent step:
    - Compute gradients of loss wrt image
    - Normalize gradients to avoid exploding updates
    - Update the image in the direction of maximum activation
    """
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img)
    grads = tape.gradient(loss, img)
    grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-6)  # Normalize
    img += learning_rate * grads
    return loss, img


def gradient_ascent_loop(img, iterations, learning_rate, max_loss=None):
    """
    Run multiple gradient ascent steps on an image.
    Stops early if `max_loss` is exceeded.
    """
    for i in range(iterations):
        loss, img = gradient_ascent_step(img, learning_rate)
        if max_loss is not None and loss > max_loss:
            break
        print(f"... Loss value at step {i}: {loss:.2f}")
    return img


"""🚀 Run the DeepDream Training Loop"""

# Preprocess the original image
original_img = preprocess_image(base_image_path)
original_shape = original_img.shape[1:3]

# Compute shapes for each octave (small → large)
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale**i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]  # Smallest first

# Initialize shrunk version of the original image
shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])
img = tf.identity(original_img)  # Make a working copy

# Process each octave
for i, shape in enumerate(successive_shapes):
    print(f"Processing octave {i} with shape {shape}")
    img = tf.image.resize(img, shape)
    img = gradient_ascent_loop(
        img, iterations=iterations, learning_rate=step, max_loss=max_loss
    )
    # Recover lost detail by blending with original
    upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
    same_size_original = tf.image.resize(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = tf.image.resize(original_img, shape)

# Convert final tensor back to RGB and save
deprocessed_img = deprocess_image(img.numpy())
utils.save_img(path=f"outputs/{result_prefix}.png", x=deprocessed_img)

# Display result
plt.figure(figsize=(6, 6))
plt.imshow(deprocessed_img)
plt.axis("off")
plt.show()
