# vit_demo.py
import tensorflow as tf
from transformers import TFViTForImageClassification, ViTConfig
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image, image_array

def main():
    parser = argparse.ArgumentParser(description="Vision Transformer Demo")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()
    
    # Load and process the image
    original_image, image_array = load_image(args.image_path)
    
    # Create a simple ViT configuration (smaller than the full model for quick testing)
    config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_labels=1000,  # ImageNet classes
        hidden_size=384,
        num_hidden_layers=4,
        num_attention_heads=6,
    )
    
    # Initialize a model from the config
    print("Loading model...")
    model = TFViTForImageClassification(config)
    print("Model loaded!")
    
    # For a real test, you would load pretrained weights, but for a quick test:
    print("Running inference...")
    outputs = model(image_array, training=False)
    predicted_class = tf.argmax(outputs.logits, axis=-1)[0].numpy()
    
    # Display results
    plt.figure(figsize=(6, 6))
    plt.imshow(original_image)
    plt.title(f"Test: Class {predicted_class}")
    plt.axis("off")
    plt.savefig("prediction_result.png")
    plt.show()
    
    print(f"Test completed successfully! Check 'prediction_result.png'")

if __name__ == "__main__":
    main()
