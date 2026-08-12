import timm
import torch

# 配置
MODEL_NAME = "mobilevit_xs"   # 要导出的模型
NUM_CLASSES = 7
IMAGE_SIZE = 224
CKPT_PATH = f"../models/{MODEL_NAME}_best.pth"
ONNX_PATH = f"../models/{MODEL_NAME}.onnx"

