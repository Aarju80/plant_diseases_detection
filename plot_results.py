# 📦 Imports
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import os
import pandas as pd
from tqdm import tqdm


# === Paths ===
MODEL_PATH = "model/resnet_crop_disease_best.pth"
LABEL_MAP_PATH = "model/label_map.json"
DATA_DIR = "PlantVillage"  # root dataset folder (with train/test subfolders)

# === Load label map ===
with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)
idx_to_class = {v: k for k, v in label_map.items()}

# === Transform for validation/test data ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Load dataset (assumes subfolders by class) ===
test_data = datasets.ImageFolder(DATA_DIR, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(label_map))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded successfully!")

# === Evaluate Model ===
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Evaluating"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap="Greens")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# === Classification Report ===
report = classification_report(y_true, y_pred, target_names=list(label_map.keys()), output_dict=True)
report_df = pd.DataFrame(report).transpose()
plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Classification Report (Precision, Recall, F1-score)")
plt.show()

# === Sample Predictions Visualization ===
import random
from PIL import Image

plt.figure(figsize=(12, 12))
sample_indices = random.sample(range(len(test_data)), 9)

for i, idx in enumerate(sample_indices):
    img_path, true_label_idx = test_data.samples[idx]
    img = Image.open(img_path)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred_label_idx = torch.argmax(output, 1).item()

    true_label = idx_to_class[true_label_idx]
    pred_label = idx_to_class[pred_label_idx]
    color = "green" if true_label == pred_label else "red"

    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.title(f"T: {true_label}\nP: {pred_label}", color=color)
    plt.axis("off")

plt.suptitle("Sample Predictions from ResNet-50 Model", fontsize=16)
plt.show()
