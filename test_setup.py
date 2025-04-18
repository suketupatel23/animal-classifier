# test_setup.py
import torch
import torchvision
from transformers import ViTFeatureExtractor, ViTForImageClassification

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("GPU available:", torch.cuda.is_available())

# Try to load a small model to test transformers
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

print("Successfully loaded ViT model!")
