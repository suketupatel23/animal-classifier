import tkinter as tk
from tkinter import filedialog, ttk
import torch
from torchvision import models, transforms
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import urllib.request
import threading

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")
        self.root.geometry("1000x700")  # Larger window size
        self.root.minsize(800, 600)     # Set minimum window size
        self.root.configure(bg="#f0f0f0")
        
        # Initialize model
        self.model = None
        self.categories = []
        
        # Create GUI elements
        self.create_widgets()
        
        # Load model in a separate thread
        threading.Thread(target=self.load_model).start()
    
    def create_widgets(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#4a7abc")
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            header_frame,
            text="Image Classification with ResNet",
            font=("Arial", 18, "bold"),
            fg="white",
            bg="#4a7abc",
            pady=10
        )
        title_label.pack()
        
        # Main content area
        content_frame = tk.Frame(self.root, bg="#f0f0f0")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Image display
        left_panel = tk.Frame(content_frame, bg="#f0f0f0", width=450)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.image_frame = tk.Frame(left_panel, bg="white", bd=1, relief=tk.SOLID)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.image_label = tk.Label(self.image_frame, bg="white", text="No image selected")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        select_btn = tk.Button(
            left_panel,
            text="Select Image",
            command=self.select_image,
            bg="#4a7abc",
            fg="white",
            font=("Arial", 12),
            padx=10,
            pady=5
        )
        select_btn.pack(fill=tk.X)
        
        # Right panel - Results
        right_panel = tk.Frame(content_frame, bg="#f0f0f0", width=450)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        results_label = tk.Label(
            right_panel,
            text="Classification Results",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0"
        )
        results_label.pack(pady=(0, 10))
        
        self.results_frame = tk.Frame(right_panel, bg="white", bd=1, relief=tk.SOLID)
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.status_label = tk.Label(
            self.root,
            text="Loading model...",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_model(self):
        # Download ImageNet class labels if they don't exist
        if not os.path.exists('imagenet_classes.txt'):
            self.update_status("Downloading ImageNet class labels...")
            url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            urllib.request.urlretrieve(url, "imagenet_classes.txt")
        
        # Load categories
        with open('imagenet_classes.txt', 'r') as f:
            self.categories = [s.strip() for s in f.readlines()]
        
        # Load the pre-trained model
        self.update_status("Loading pre-trained ResNet model...")
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.model.eval()
        
        self.update_status("Ready")
    
    def update_status(self, message):
        # Update status bar with a message
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def select_image(self):
        # Open file dialog to select image
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            self.process_image(file_path)
    
    def process_image(self, image_path):
        if self.model is None:
            self.update_status("Model not loaded yet. Please wait...")
            return
        
        self.update_status(f"Processing image: {os.path.basename(image_path)}")
        
        # Load and display the image
        image = Image.open(image_path).convert('RGB')
        self.display_image(image)
        
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
        
        # Transform the image and add batch dimension
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)
        
        # Perform inference
        with torch.no_grad():
            output = self.model(input_batch)
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        # Display results
        self.display_results(top5_catid, top5_prob)
        
        self.update_status("Ready")
    
    def display_image(self, image):
        # Clear the current image
        self.image_label.config(image=None)
        
        # Resize image to fit in the frame while maintaining aspect ratio
        display_image = image.copy()
        
        # Get frame dimensions
        frame_width = self.image_frame.winfo_width()
        frame_height = self.image_frame.winfo_height()
        
        # If the frame hasn't been drawn yet, use default values
        if frame_width <= 1:
            frame_width = 450
        if frame_height <= 1:
            frame_height = 350
        
        # Calculate new dimensions
        img_width, img_height = display_image.size
        ratio = min(frame_width/img_width, frame_height/img_height)
        new_width = int(img_width * ratio * 0.9)  # 90% of available space
        new_height = int(img_height * ratio * 0.9)
        
        # Resize image
        display_image = display_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage and display
        photo = ImageTk.PhotoImage(display_image)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # Keep a reference to prevent garbage collection
    
    def display_results(self, top_indices, top_probs):
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Create figure for matplotlib
        fig = plt.Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Get shortened category names to fit better
        categories = [self.categories[idx] for idx in top_indices]
        # Shorten category names if needed (first word only)
        short_names = [cat.split(',')[0] for cat in categories]
        
        # Format labels with probability percentages
        labels = [f"{name} ({prob:.1%})" for name, prob in zip(short_names, top_probs)]
        
        # Create horizontal bar chart
        y_pos = range(len(top_indices))
        bars = ax.barh(y_pos, top_probs.numpy(), align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Probability')
        ax.set_title('Top 5 Predictions')
        
        # Ensure y-axis labels are visible by adjusting margins
        fig.tight_layout(pad=2.0)
        
        # Add bar labels for better readability
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{width:.2f}", ha='left', va='center')
        
        # Embed plot in tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create a scrollable text area for detailed results
        result_text = tk.Text(self.results_frame, height=8, bg="white", wrap=tk.WORD)
        result_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(result_text, command=result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        result_text.config(yscrollcommand=scrollbar.set)
        
        # Insert detailed results
        result_text.insert(tk.END, f"Top prediction: {self.categories[top_indices[0]]}\n")
        result_text.insert(tk.END, f"Confidence: {top_probs[0]:.2%}\n\n")
        result_text.insert(tk.END, "All top 5 predictions:\n")
        
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            result_text.insert(tk.END, f"{i+1}. {self.categories[idx]} ({prob:.2%})\n")
        
        # Make text read-only
        result_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    # Create and run the app
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
