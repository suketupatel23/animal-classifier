# example.py
import torch
import os
from PIL import Image
import requests
from io import BytesIO
from transformers import ViTForImageClassification, ViTImageProcessor
import matplotlib.pyplot as plt

def test_vision_transformer():
    # Download an example image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    
    # Load pre-trained ViT model and processor
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process results
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]
    
    # Display results
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f"Prediction: {predicted_class}")
    plt.axis("off")
    plt.savefig("example_prediction.png")
    plt.show()
    
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {logits.softmax(dim=-1).max().item():.4f}")

if __name__ == "__main__":
    test_vision_transformer()
