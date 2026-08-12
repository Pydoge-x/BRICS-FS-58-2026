"""检查项目依赖是否完整"""
import sys

modules = [
    ('torch', 'PyTorch'),
    ('timm', 'timm'),
    ('cv2', 'OpenCV'),
    ('pandas', 'Pandas'),
    ('flask', 'Flask'),
    ('flask_cors', 'Flask-CORS'),
    ('onnxruntime', 'ONNXRuntime'),
    ('sklearn', 'Scikit-learn'),
    ('tqdm', 'tqdm'),
    ('yaml', 'PyYAML'),
    ('matplotlib', 'Matplotlib'),
]

print("=" * 50)
print("检查项目依赖")
print("=" * 50)

all_ok = True
for module, name in modules:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"[OK] {name}: {version}")
    except ImportError:
        print(f"[MISSING] {name}")
        all_ok = False

print("=" * 50)
if all_ok:
    print("所有依赖已安装！")
else:
    print("部分依赖缺失，请运行: pip install -r requirements.txt")
print("=" * 50)
