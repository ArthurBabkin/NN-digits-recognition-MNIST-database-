# Import dependencies
import torch
from PIL import Image, ImageFilter
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

# Checking background and foreground relation
# We need a white digit with a black background!
def ensure_white_digit_on_black_background(img):
    # Convert to NumPy array
    img_array = np.array(img)

    # Check background and digit colors
    is_black_background = np.mean(img_array) > 128

    # Invert the image if colors do not match expectations
    if is_black_background:
        img_array = 255 - img_array

    # Convert back to Image
    result_img = Image.fromarray(img_array.astype(np.uint8))

    return result_img

# Load image
def change_pixel_color(img):
    img = img.convert("L")
    img_array = np.array(img)
    img = increase_sharpness(img)

    height, width = 28, 28

    def get_pixel_color(x, y):
        return img_array[x, y] if 0 <= x < width and 0 <= y < height else None

    for y in range(height):
        for x in range(width):
            current_color = get_pixel_color(x, y)
            if current_color > 240:
                current_color = 255
            elif current_color < 15:
                current_color = 0
            if current_color is not None:
                neighbors = [
                    get_pixel_color(x - 1, y),
                    get_pixel_color(x + 1, y),
                    get_pixel_color(x, y - 1),
                    get_pixel_color(x, y + 1),
                ]
                black = neighbors.count(0)
                white = neighbors.count(255)
                if white >= 3 and get_pixel_color(x, y) < 240:
                    img_array[x, y] = 255
                elif black >= 3 and get_pixel_color(x, y) > 10:
                    img_array[x, y] = 0

    result_img = Image.fromarray(img_array.astype(np.uint8))
    return result_img

def increase_sharpness(img):
    # Apply multiple iterations of unsharp mask to increase sharpness
    for _ in range(5):
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return img

# Get data
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
data = DataLoader(train, 32)

# Image Classifier Neural Network
class Image_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, x):
        return self.model(x)

# Instance of neural network, loss, optimizer
clf = Image_Classifier().to('mps')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

if __name__ == "__main__":
    # Load pre-trained model
    with open('model_state1.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    # Load and preprocess the input image
    img = Image.open('img7.png')
    img.show()

    img = img.convert("RGB")
    img = increase_sharpness(img)
    img = ensure_white_digit_on_black_background(img)
    # img.show()

    img = increase_sharpness(img)
    img = change_pixel_color(img)
    img = increase_sharpness(img)
    img = change_pixel_color(img)
    img = img.convert("L")
    img.show()

    # Convert the image to a PyTorch tensor
    img_tensor = ToTensor()(img).unsqueeze(0).to('mps')

    # Make a prediction using the neural network
    print(torch.argmax(clf(img_tensor)))
