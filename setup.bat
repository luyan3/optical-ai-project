@echo off
chcp 65001 >nul
echo ============================================
echo  光学图像智能分类系统 - 环境安装脚本
echo ============================================
echo.
echo [1/3] 正在安装 PyTorch（CPU版本）...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if %ERRORLEVEL% NEQ 0 (
    echo PyTorch 安装失败，尝试备用方式...
    pip install torch torchvision
)
echo.
echo [2/3] 正在安装其他依赖...
pip install medmnist scikit-learn matplotlib seaborn gradio pillow numpy
echo.
echo [3/3] 正在验证安装...
python -c "import torch; import torchvision; import medmnist; import sklearn; import gradio; print('所有依赖安装成功！')"
echo.
echo ============================================
echo  安装完成！
echo  运行 'python train.py' 开始训练模型
echo  运行 'python app.py' 启动演示界面
echo ============================================
pause