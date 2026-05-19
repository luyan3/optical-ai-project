import torch
import medmnist
from medmnist import INFO, PathMNIST
from torchvision import models, transforms

print("正在测试数据集下载...")
info = INFO["pathmnist"]
train_dataset = PathMNIST(split="train", transform=transforms.ToTensor(), download=True)
print(f"数据集已加载: {len(train_dataset)} 个样本, {len(info['label'])} 个类别")
print(f"标签: {info['label']}")

print("正在测试模型创建...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Linear(model.fc.in_features, len(info["label"]))
print(f"模型: ResNet18, 参数数量: {sum(p.numel() for p in model.parameters()):,}")

x, y = train_dataset[0]
print(f"样本形状: {x.shape}, 标签: {y}")
print("所有检查通过！")