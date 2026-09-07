"""
数据探索脚本：随机展示训练集中每个类别的5张图片（共35张）
"""

import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# 项目根目录（src/data/show.py -> 项目根目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'dataset_data', 'fer2013')
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')

# 类别名称（与数据集目录一致）
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

random.seed(42)

fig, axes = plt.subplots(len(CLASSES), 5, figsize=(12, 15))
fig.suptitle('FER2013 Training Samples - 5 per class', fontsize=16)

for row, cls in enumerate(CLASSES):
    cls_dir = os.path.join(TRAIN_DIR, cls)
    imgs = [f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    sample = random.sample(imgs, 5)
    for col, img_name in enumerate(sample):
        img_path = os.path.join(cls_dir, img_name)
        img = Image.open(img_path).convert('L')  # 灰度图，更直观体现图像质量
        ax = axes[row][col]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(cls, fontsize=12)

plt.tight_layout()
plt.show()