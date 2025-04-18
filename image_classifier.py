import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse

def load_image(image_path):
    """Load an image from a file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Could not find image at {image_path}")
    return Image.open(image_path).convert('RGB')

def predict_image(image_path):
    """Predict the class of an image using a pre-trained ResNet model."""
    # Load the pre-trained model
    print("Loading pre-trained ResNet model...")
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.eval()
    
    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load and preprocess the image
    print(f"Loading image from {image_path}...")
    image = load_image(image_path)
    
    # Display the original image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Transform the image and add batch dimension
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    
    # Perform inference
    print("Running inference...")
    with torch.no_grad():
        output = model(input_batch)
    
    # The output has unnormalized scores. To get probabilities, run a softmax on it
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get ImageNet class mappings
    with open('imagenet_classes.txt', 'r') as f:
        categories = [s.strip() for s in f.readlines()]
    
    # Show top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    # Print results
    print("\nTop 5 predictions:")
    for i in range(5):
        print(f"{categories[top5_catid[i]]}: {top5_prob[i].item()*100:.2f}%")
    
    # Plot the results
    plt.subplot(1, 2, 2)
    y_pos = range(5)
    plt.barh(y_pos, top5_prob.cpu().numpy())
    plt.yticks(y_pos, [categories[idx] for idx in top5_catid])
    plt.xlabel('Probability')
    plt.title('Top 5 Predictions')
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()
    
    return categories[top5_catid[0]], top5_prob[0].item()

def main():
    parser = argparse.ArgumentParser(description='Image Classification with Pre-trained ResNet')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    args = parser.parse_args()
    
    # Download ImageNet class labels if they don't exist
    if not os.path.exists('imagenet_classes.txt'):
        print("Downloading ImageNet class labels...")
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        urllib.request.urlretrieve(url, "imagenet_classes.txt")
    
    # Predict the image
    top_class, confidence = predict_image(args.image)
    print(f"\nFinal prediction: {top_class} with {confidence*100:.2f}% confidence")

if __name__ == "__main__":
    main()
