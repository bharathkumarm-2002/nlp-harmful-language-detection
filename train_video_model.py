import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Parameters
data_dir = "datasets/video_frames"
num_epochs = 10
batch_size = 16
learning_rate = 0.0001
model_save_path = "models/video_model.pth"

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load pretrained model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Training started...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    true_labels, pred_labels = [], []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

    acc = accuracy_score(true_labels, pred_labels)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Accuracy: {acc:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"✅ Model saved to {model_save_path}")
