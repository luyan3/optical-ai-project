import torch
import torch.nn as nn
from torchvision import models, transforms
import medmnist
from medmnist import INFO
import gradio as gr
import numpy as np
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_flag = "pathmnist"
info = INFO[data_flag]
n_classes = len(info["label"])
label_names = list(info["label"].values())
label_cn = {
    "adipose": "脂肪组织",
    "background": "背景",
    "debris": "细胞碎片",
    "lymphocytes": "淋巴细胞",
    "mucus": "黏液",
    "smooth muscle": "平滑肌",
    "normal colon mucosa": "正常结肠黏膜",
    "cancer-associated stroma": "癌相关间质",
    "colorectal adenocarcinoma epithelium": "结直肠腺癌上皮"
}

mean_std = info.get("mean_std", None)
if mean_std:
    mean, std = mean_std
else:
    mean, std = [0.5], [0.5]

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

model = models.resnet18()
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, n_classes)

model_path = "output/best_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("模型加载成功")
else:
    print(f"模型文件不存在: {model_path}，请先运行 train.py 训练模型")
    model = None

def predict(img):
    if model is None:
        return {"错误": 1.0}
    img_pil = Image.fromarray(img).convert("RGB")
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    result = {}
    for i in range(n_classes):
        en_name = label_names[i]
        cn_name = label_cn.get(en_name, en_name)
        result[f"{cn_name}"] = float(probs[i])
    return result

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="上传病理图像"),
    outputs=gr.Label(num_top_classes=3, label="分类结果"),
    title="病理图像智能分类系统",
    description="上传一张病理组织图像，模型将自动识别其所属的9种组织类型之一。",
    examples=None,
)

if __name__ == "__main__":
    demo.launch(share=False)