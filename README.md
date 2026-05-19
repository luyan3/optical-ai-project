# 病理图像智能分类系统

基于 PyTorch 和 ResNet18 的光学显微图像多分类系统，在 PathMNIST 数据集（9类、约10万张图像）上实现 **验证准确率 99.22%，测试准确率 89.00%**。

## 功能

- 图像分类：识别9种病理组织类型
- Gradio Web界面：上传图像即可实时预测
- 训练可视化：损失曲线、混淆矩阵

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 训练模型
python train.py

# 启动演示界面
python app.py
```

## 项目结构

```
optical_ai_project/
├── train.py              # 训练脚本
├── app.py                # Gradio 演示界面
├── setup.bat             # Windows 一键安装
├── test_setup.py         # 环境检查
├── requirements.txt      # 依赖清单
├── sample_images/        # 测试样本图片
├── output/               # 训练输出
│   ├── best_model.pth    # 预训练模型 (99.22%)
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
└── .gitignore
```

## 模型性能

| 指标 | 结果 |
|------|------|
| 最佳验证准确率 | **99.22%** |
| 最终测试准确率 | **89.00%** |
| 训练轮数 | 18/20 |
| 模型架构 | ResNet18 (预训练) |

## 技术栈

Python · PyTorch · ResNet18 · Gradio · scikit-learn · Matplotlib · MedMNIST

## 数据集

使用 [MedMNIST](https://medmnist.com/) 中的 PathMNIST 数据集，包含 9 种结直肠癌病理组织分类。