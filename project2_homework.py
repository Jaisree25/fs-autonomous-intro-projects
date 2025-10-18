# Adding two integers given handwritten MNIST integers
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import random

# Load MNIST
transform = transforms.ToTensor()
train_mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_mnist = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Dataset of image pairs and their sum
def get_batch(mnist, batch_size=64):
    imgs = []
    labels = []
    for i in range(batch_size):
        img1, label1 = random.choice(mnist)
        img2, label2 = random.choice(mnist)
        combined = torch.cat((img1, img2), dim=2)
        imgs.append(combined)
        labels.append(label1 + label2)
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels)
    return imgs, labels

# Neural network
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(1 * 28 * 56, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 19) 
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Training started...\n")
for epoch in range(20):
    model.train()
    total_loss = 0
    for i in range(400):  
        imgs, labels = get_batch(train_mnist)
        optimizer.zero_grad()
        out = model(imgs)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss = {total_loss/400:.4f}")

# test
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for i in range(100):
        imgs, labels = get_batch(test_mnist)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f"\nTest accuracy: {100 * correct / total:.2f}%")
