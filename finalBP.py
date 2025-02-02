
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import psutil
import pynvml
import time

# Hyperparameters
k_size = 5
p_size = 2

# GLOBAL VARS
saved_images_count = 0


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


# Define a Basic CNN Model
class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super(BasicCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=k_size, padding=p_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=k_size, padding=p_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=k_size, padding=p_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=k_size, padding=p_size),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Training Loop
def train_backprop_model(model, dataloader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        start_time = time.time()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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
def test_backprop_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
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

# Main Function
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = prepare_dataloader(batch_size=64)
    model = BasicCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    print("Starting Training...")
    print(f"Dataset: CIFAR-10 | Algo: BP | Kernel: {k_size} | Pad: {p_size } | MaxPooling: TRUE")
    print(f"Channels: 3 → 64 → 128 → 256 → 512 → 1 (via AdaptiveAvgPool) → num_classes")
    train_backprop_model(model, trainloader, criterion, optimizer, device, epochs=50)

    print("Starting Testing...")
    test_backprop_model(model, testloader, device)

