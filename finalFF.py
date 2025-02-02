import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import psutil
import pynvml
import time
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Hyperparameters
opacity = 0.3
k_size = 5
p_size = 2

# GLOBAL VARS
saved_images_count = 0

# Utility Functions
# Utility Function: Save Image
def save_image(tensor, label, index, prefix="original"):
    os.makedirs("InputImage", exist_ok=True)
    # Ensure mean and std are on the same device as the tensor
    mean = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(3, 1, 1)  # CIFAR-10 normalization mean
    std = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(3, 1, 1)   # CIFAR-10 normalization std
    tensor = tensor * std + mean  # Reverse normalization

    # Convert from CHW to HWC and scale to 0-255
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)  # Scale to 0-255

    file_path = f"InputImage/{prefix}_label_{label}_index_{index}.png"
    Image.fromarray(image).save(file_path)

# Utility Function: Apply Spatial Labeling

# Initialize NVML for GPU stats
pynvml.nvmlInit()

def get_cpu_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

def get_gpu_memory_usage():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.used / (1024 * 1024)  # Convert to MB

def get_cpu_usage():
    return psutil.cpu_percent(interval=None)

def get_gpu_power_usage():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert from milliwatts to watts
    return power


# Utility Functions


def apply_spatial_labeling(images, labels, num_classes):
    global saved_images_count  # Declare as global to persist across function calls

    batch_size, channels, height, width = images.size()
    label_patterns = torch.zeros_like(images).to(images.device)  # Ensure label_patterns is on the same device as images

    for label in range(num_classes):
        freq = 1 + label % 4  # Frequency patterns
        phase = label % 2
        for b in range(batch_size):
            if labels[b] == label:
                for c in range(channels):
                    x = torch.linspace(0, 2 * np.pi, steps=width, device=images.device)
                    y = torch.linspace(0, 2 * np.pi, steps=height, device=images.device)
                    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
                    pattern = torch.sin(freq * grid_x + phase * grid_y)
                    label_patterns[b, c] += pattern

    # Save exactly 3 images across all batches
    if saved_images_count < 3:
        remaining_to_save = 3 - saved_images_count  # Calculate how many more images to save
        random_indices = random.sample(range(batch_size), min(remaining_to_save, batch_size))
        for idx in random_indices:
            # Save the original image
            save_image(images[idx], labels[idx].item(), idx, prefix="original")
            # Save the label-encoded image
            save_image(images[idx] + opacity * label_patterns[idx], labels[idx].item(), idx, prefix="labeled")
            saved_images_count += 1
            if saved_images_count >= 3:  # Stop saving once 3 images are saved
                break

    # Combine image and label pattern
    images = images + opacity * label_patterns  # Adjust contribution of labels
    images = torch.clamp(images, 0, 1)      # Normalize back to [0, 1]
    return images


# Define Auxiliary Layer for Goodness Computation
class AuxiliaryLayer(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(AuxiliaryLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, num_classes, kernel_size=1)

    def forward(self, x):
        goodness_tensor = self.conv(x)  # Compute goodness tensor
        goodness_scores = goodness_tensor.mean(dim=(2, 3))  # Pool spatial dimensions
        return goodness_scores

# Define a Block with Backpropagation
class Block(nn.Module):
    def __init__(self, block_layers, aux_input_channels, num_classes):
        super(Block, self).__init__()
        self.block_layers = nn.Sequential(*block_layers)
        self.aux_layer = AuxiliaryLayer(aux_input_channels, num_classes)

    def forward(self, x):
        features = self.block_layers(x)
        goodness_scores = self.aux_layer(features)
        return features, goodness_scores

# Define the Full Model
class HybridFFModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super(HybridFFModel, self).__init__()
        self.blocks = nn.ModuleList()
        for block in backbone:
            self.blocks.append(Block(block['layers'], block['aux_input_channels'], num_classes))

    def forward(self, x):
        all_goodness_scores = []
        for block in self.blocks:
            x, goodness_scores = block(x)
            all_goodness_scores.append(goodness_scores)
        return torch.stack(all_goodness_scores, dim=1)  # Stack goodness scores for all blocks

# Loss Function
def hybrid_loss(goodness_scores, target):
    batch_size, num_blocks, num_classes = goodness_scores.size()
    positive_goodness = []

    for block_idx in range(num_blocks):
        # Extract the target class for the current block
        target_for_block = target
        # Gather the goodness scores for the target class in the current block
        block_goodness = goodness_scores[:, block_idx, :].gather(1, target_for_block.unsqueeze(1)).squeeze(1)
        positive_goodness.append(block_goodness)

    # Stack the positive goodness scores for all blocks
    positive_goodness = torch.stack(positive_goodness, dim=1).mean(dim=1)  # Mean across blocks

    # Compute the log-sum-exp across all classes
    log_sum_exp = torch.logsumexp(goodness_scores, dim=2).mean(dim=1)  # Mean across blocks
    loss = -torch.mean(positive_goodness - log_sum_exp)  # Maximize positive goodness
    return loss



# Training Loop
def train_hybrid_model(model, dataloader, num_classes, device, epochs=10, lr=1e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        start_time = time.time()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = apply_spatial_labeling(inputs, labels, num_classes)
            optimizer.zero_grad()
            goodness_scores = model(inputs)
            loss = hybrid_loss(goodness_scores, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_time = time.time() - start_time
        cpu_memory = get_cpu_memory_usage()
        gpu_memory = get_gpu_memory_usage() if device.type == 'cuda' else 0
        cpu_usage = get_cpu_usage()
        power_usage = get_gpu_power_usage()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}, "
              f"Time: {epoch_time:.2f}s, CPU Mem: {cpu_memory:.2f} MB, GPU Mem: {gpu_memory:.2f} MB, CPU Usage: {cpu_usage:.2f}%, GPU Power: {power_usage:.2f} W")

# Testing Loop
def test_hybrid_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = apply_spatial_labeling(inputs, labels, 10)
            goodness_scores = model(inputs)
            predictions = goodness_scores.sum(dim=1).argmax(dim=1)  # Sum goodness scores across blocks
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    return accuracy


# CIFAR-10 Dataset and Dataloader
def prepare_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader

# Define Backbone (ResNet18-like structure divided into blocks)
def create_backbone():
    backbone = [
        {'layers': [nn.Conv2d(3, 64, kernel_size=k_size, padding=p_size), nn.ReLU(), nn.MaxPool2d(2)],
         'aux_input_channels': 64},
        {'layers': [nn.Conv2d(64, 128, kernel_size=k_size, padding=p_size), nn.ReLU(), nn.MaxPool2d(2)],
         'aux_input_channels': 128},
        {'layers': [nn.Conv2d(128, 256, kernel_size=k_size, padding=p_size), nn.ReLU(), nn.MaxPool2d(2)],
         'aux_input_channels': 256},
        {'layers': [nn.Conv2d(256, 512, kernel_size=k_size, padding=p_size), nn.ReLU(), nn.AdaptiveAvgPool2d(1)],
         'aux_input_channels': 512},
    ]
    return backbone

# Main Function
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = prepare_dataloader(batch_size=64)
    backbone = create_backbone()
    model = HybridFFModel(backbone, num_classes=10).to(device)

    print("Starting Training...")
    print(f"Dataset: CIFAR-10 | Algo: FF | Kernel: {k_size} | Pad: {p_size } | Opacity {opacity} | MaxPooling: TRUE")
    print(f"Channels: 3 → 64 → 128 → 256 → 512  Spatial Dimensions: 32 x 32 → 16 x 16 → 8 x 8 → 4 x 4 → 1 x 1")
    train_hybrid_model(model, trainloader, num_classes=10, device=device, epochs=50, lr=1e-3)

    print("Starting Testing...")
    test_hybrid_model(model, testloader, device=device)
