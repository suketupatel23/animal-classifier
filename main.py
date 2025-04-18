import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import matplotlib.pyplot as plt
import numpy as np
import argparse

def load_image(image_path):
    """Load and preprocess an image."""
    image = Image.open(image_path).convert("RGB")
    return image

def predict(model, processor, image):
    """Make a prediction using the vision transformer model."""
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get prediction
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    return model.config.id2label[predicted_class_idx], logits

def main():
    parser = argparse.ArgumentParser(description="Vision Transformer Image Classification")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()
    
    # Load pre-trained ViT model and processor
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    
    # Load and process the image
    image = load_image(args.image_path)
    
    # Make prediction
    predicted_class, logits = predict(model, processor, image)
    
    # Display results
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")
    
    # Get top 5 predictions
    top5_probabilities, top5_indices = torch.topk(logits.softmax(dim=-1), 5)
    
    plt.subplot(1, 2, 2)
    labels = [model.config.id2label[idx.item()] for idx in top5_indices[0]]
    probabilities = top5_probabilities[0].tolist()
    
    plt.barh(np.arange(5), probabilities)
    plt.yticks(np.arange(5), labels)
    plt.xlabel("Probability")
    plt.title("Top 5 Predictions")
    
    plt.tight_layout()
    plt.savefig("prediction_result.png")
    plt.show()
    
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {logits.softmax(dim=-1).max().item():.4f}")

if __name__ == "__main__":
    main()
