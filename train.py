import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

import matplotlib.pyplot as plt
import time

from models.lenet5 import LeNet5


# DEVICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# TRANSFORMS

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# DATASETS

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)


# DATALOADERS

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# MODEL

model = LeNet5().to(device)

print(model)


# LOSS FUNCTION

criterion = nn.CrossEntropyLoss()


# OPTIMIZER

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# LEARNING RATE SCHEDULER

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)


# PARAMETER COUNT

total_params = sum(
    p.numel() for p in model.parameters()
)

print("Total Parameters:", total_params)


# TRAINING VARIABLES

epochs = 10

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []

epoch_times = []


# TRAINING LOOP

for epoch in range(epochs):

    start_time = time.time()

    model.train()

    running_loss = 0

    correct = 0
    total = 0

    loop = tqdm(train_loader)

    for images, labels in loop:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")

        loop.set_postfix(loss=loss.item())

    train_loss = running_loss / len(train_loader)

    train_accuracy = 100 * correct / total

    train_losses.append(train_loss)

    train_accuracies.append(train_accuracy)


    # VALIDATION

    model.eval()

    val_loss = 0

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(test_loader)

    val_accuracy = 100 * val_correct / val_total

    val_losses.append(val_loss)

    val_accuracies.append(val_accuracy)


    # STEP LR

    scheduler.step()


    # TIME

    epoch_time = time.time() - start_time

    epoch_times.append(epoch_time)


    # PRINT

    print(f"\nEpoch {epoch+1}")

    print(f"Train Loss: {train_loss:.4f}")

    print(f"Train Accuracy: {train_accuracy:.2f}%")

    print(f"Validation Loss: {val_loss:.4f}")

    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    print(f"Time per epoch: {epoch_time:.2f} seconds")


# FINAL TEST ACCURACY

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total

print("\nFinal Test Accuracy:", test_accuracy)


# LOSS PLOT

plt.figure(figsize=(10, 5))

plt.plot(train_losses, label='Train Loss')

plt.plot(val_losses, label='Validation Loss')

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Loss Curve")

plt.legend()

plt.show()


# ACCURACY PLOT

plt.figure(figsize=(10, 5))

plt.plot(train_accuracies, label='Train Accuracy')

plt.plot(val_accuracies, label='Validation Accuracy')

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Accuracy Curve")

plt.legend()

plt.show()


# FILTER VISUALIZATION

filters = model.features[0].weight.data.cpu()

fig, axes = plt.subplots(8, 4, figsize=(8, 16))

for i, ax in enumerate(axes.flat):

    filter_img = filters[i][0]

    ax.imshow(filter_img, cmap='gray')

    ax.axis('off')

plt.tight_layout()

plt.show()


# COMPARISON TABLE

print("\n===== COMPARISON =====")

print(f"LeNet-5 Test Accuracy: {test_accuracy:.2f}%")

print(f"LeNet-5 Parameters: {total_params}")

print(f"Average Time per Epoch: {sum(epoch_times)/len(epoch_times):.2f} sec")