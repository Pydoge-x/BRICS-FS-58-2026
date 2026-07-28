import timm
import torch

MODEL_NAME = "mobilevit_xs"   # 要导出的模型
NUM_CLASSES = 7
IMAGE_SIZE = 224
CKPT_PATH = f"../models/{MODEL_NAME}_best.pth"
ONNX_PATH = f"../models/{MODEL_NAME}.onnx"

# 1. 重建模型结构并加载训练好的参数
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
state_dict = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(state_dict)          
model.eval()                    

# 2. 构造假输入，形状应为 [batch, channel, H, W]
dummy_input = torch.randn(1, 3, 224, 224)   

# 3. 导出ONNX
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},  # batch维动态
    opset_version=17,
    dynamo=False,  # 使用经典TorchScript导出器，兼容dynamic_axes
)
print(f"已导出: {ONNX_PATH}")