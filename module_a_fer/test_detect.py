"""测试人脸检测：对比 InsightFace 和 OpenCV"""
import cv2
import numpy as np
import os
from src.inference.detector import FaceDetector

# 模型根目录 = 项目 models/
MODEL_ROOT = os.path.join(os.path.dirname(__file__), 'models')

img = cv2.imread('test/face-fers.jpg')
print(f'图像尺寸: {img.shape}')

# 测试 InsightFace
print('\n=== InsightFace 检测 ===')
try:
    det_if = FaceDetector(detector_type='insightface', insightface_root=MODEL_ROOT)
    faces_if = det_if.detect(img)
    print(f'检测到 {len(faces_if)} 个人脸')
    for i, f in enumerate(faces_if):
        bbox = f['bbox']
        conf = f['confidence']
        print(f'  [{i+1}] bbox=[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}], conf={conf:.3f}')
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'InsightFace 失败: {e}')

# 测试 OpenCV
print('\n=== OpenCV 检测 (min_size=20, scale=1.1) ===')
det_cv = FaceDetector(detector_type='opencv', min_face_size=20, scale_factor=1.1)
faces_cv = det_cv.detect(img)
print(f'检测到 {len(faces_cv)} 个人脸')
for i, f in enumerate(faces_cv):
    bbox = f['bbox']
    print(f'  [{i+1}] bbox=[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]')

# 绘制结果
if faces_if:
    result = img.copy()
    for fi in faces_if:
        x, y, w, h = fi['bbox']
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(result, f'{fi["confidence"]:.2f}', (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite('test/detected_insightface.jpg', result)
    print('\n已保存: test/detected_insightface.jpg')

if faces_cv:
    result2 = img.copy()
    for fc in faces_cv:
        x, y, w, h = fc['bbox']
        cv2.rectangle(result2, (x, y), (x+w, y+h), (0, 0, 255), 3)
    cv2.imwrite('test/detected_opencv.jpg', result2)
    print('已保存: test/detected_opencv.jpg')

print(f'\n期望: 16 个人脸 | InsightFace: {len(faces_if)} | OpenCV: {len(faces_cv)}')
