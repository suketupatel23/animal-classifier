# simple_test.py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

def main():
    # Print environment information
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    
    # Create a simple test image
    img_array = np.random.rand(100, 100, 3)
    img = Image.fromarray((img_array * 255).astype('uint8'))
    
    # Display the image
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title("Random Test Image")
    plt.axis("off")
    plt.savefig("test_image.png")
    
    print("Test completed successfully!")
    print("A random test image has been saved as 'test_image.png'")
    print("If you can see this message and the image was created, your basic environment is working!")

if __name__ == "__main__":
    main()
