import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import medmnist
from medmnist import INFO
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

os.makedirs("output", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}", flush=True)

data_flag = "pathmnist"
info = INFO[data_flag]
n_classes = len(info["label"])
label_names = list(info["label"].values())
print(f"数据集: {data_flag}, 类别数: {n_classes}", flush=True)

mean_std = info.get("mean_std", None)
if mean_std:
    mean, std = mean_std
else:
    mean, std = [0.5], [0.5]

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

train_dataset = medmnist.PathMNIST(split="train", transform=train_transform, download=True)
val_dataset = medmnist.PathMNIST(split="val", transform=test_transform, download=True)
test_dataset = medmnist.PathMNIST(split="test", transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}", flush=True)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, n_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

start_epoch = 1
best_acc = 0
checkpoint_path = "output/best_model.pth"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"已加载已有模型: {checkpoint_path}", flush=True)

def train_epoch(loader, epoch, num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    total_batches = len(loader)
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.squeeze().long().to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        if (batch_idx + 1) % 200 == 0:
            current_acc = correct / total if total > 0 else 0
            print(f"  第{epoch}/{num_epochs}轮 - 批次 {batch_idx+1}/{total_batches} | 损失: {loss.item():.4f} | 实时准确率: {current_acc:.4f}", flush=True)
    return total_loss / total, correct / total

@torch.no_grad()
def eval_epoch(loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.squeeze().long().to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels

train_losses, train_accs = [], []
val_losses, val_accs = [], []
num_epochs = 20

if os.path.exists(checkpoint_path):
    _, current_acc, _, _ = eval_epoch(val_loader)
    best_acc = current_acc
    print(f"当前模型验证准确率: {best_acc:.4f} ({best_acc*100:.2f}%)，以此作为最佳基准", flush=True)

for epoch in range(start_epoch, num_epochs + 1):
    t_loss, t_acc = train_epoch(train_loader, epoch, num_epochs)
    v_loss, v_acc, _, _ = eval_epoch(val_loader)
    train_losses.append(t_loss)
    train_accs.append(t_acc)
    val_losses.append(v_loss)
    val_accs.append(v_acc)
    scheduler.step(v_acc)

    print(f"第{epoch:2d}/{num_epochs}轮 | "
          f"训练损失: {t_loss:.4f} 准确率: {t_acc:.4f} | "
          f"验证损失: {v_loss:.4f} 准确率: {v_acc:.4f}", flush=True)

    if v_acc > best_acc:
        best_acc = v_acc
        torch.save(model.state_dict(), "output/best_model.pth")
        print(f"  -> 已保存最佳模型 (验证准确率={v_acc:.4f})", flush=True)

model.load_state_dict(torch.load("output/best_model.pth"))
test_loss, test_acc, test_preds, test_labels = eval_epoch(test_loader)
final_test_acc = test_acc
print(f"\n最终测试集准确率: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, 'o-', label='Train Loss')
axes[0].plot(val_losses, 's-', label='Val Loss')
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].set_title("Training & Validation Loss")
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_accs, 'o-', label='Train Acc')
axes[1].plot(val_accs, 's-', label='Val Acc')
axes[1].axhline(y=best_acc, color='green', linestyle='--', alpha=0.5,
                label=f'Best Val Acc ({best_acc*100:.2f}%)')
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].set_title("Training & Validation Accuracy")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/training_curves.png", dpi=150)
plt.close()
print("已保存: output/training_curves.png", flush=True)

cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title(f"Confusion Matrix (Test Set - {final_test_acc*100:.2f}%)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("output/confusion_matrix.png", dpi=150)
plt.close()
print("已保存: output/confusion_matrix.png", flush=True)

report = classification_report(test_labels, test_preds,
                                target_names=label_names, digits=4)
print("\n分类报告:")
print(report, flush=True)
with open("output/classification_report.txt", "w") as f:
    f.write(f"Final Test Accuracy: {final_test_acc*100:.2f}%\n")
    f.write(f"Best Val Accuracy: {best_acc*100:.2f}%\n\n")
    f.write(report)

print(f"\n所有输出已保存至 'output/' 目录")
print(f"最佳验证准确率: {best_acc*100:.2f}%")
print(f"最终测试准确率: {final_test_acc*100:.2f}%")