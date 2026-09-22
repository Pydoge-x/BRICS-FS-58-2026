from pathlib import Path

from src.models.factory import create_fer_model, count_parameters


models = [
    ("VGG-16", "vgg", "vgg16", "vgg"),
    ("ResNet-50", "resnet", "resnet50", "resnet"),
    ("MobileNetV3-Small", "mobilenet", "mobilenetv3_small", "mobilenet"),
    ("MobileViT-XS", "mobilevit", "mobilevit_xs", "mobilevit"),
]

model_dir = Path("models")

for display_name, model_type, model_name, file_name in models:
    model = create_fer_model(
        model_type=model_type,
        model_name=model_name,
        num_classes=7,
        pretrained=False,
    )

    parameters = count_parameters(model)
    model_path = model_dir / f"fer_{file_name}.onnx"

    if model_path.exists():
        file_size_mb = model_path.stat().st_size / 1024 / 1024
        file_info = f"{file_size_mb:.2f} MB"
    else:
        file_info = "ONNX 文件不存在"

    print(
        f"{display_name:<20} "
        f"参数量: {parameters:,} "
        f"({parameters / 1_000_000:.2f}M), "
        f"文件大小: {file_info}"
    )