"""
Flask API路由
表情识别服务接口
"""

import base64
import io
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify


def setup_routes(app: Flask):
    """
    设置API路由
    
    Args:
        app: Flask应用
    """
    
# 人脸检测器已在 app.py 中预初始化（app.face_detector）
    
    @app.route('/api/health', methods=['GET'])
    def health():
        """健康检查"""
        return jsonify({
            'status': 'healthy',
            'model_loaded': app.model_manager.get_current_model() is not None,
            'current_model': app.model_manager.get_current_model()
        })
    
    @app.route('/api/models', methods=['GET'])
    def get_models():
        """
        获取可用模型列表
        
        TODO: 考生需要实现此方法 (考点16/24: 获取模型列表)
        
        要求：
        1. 调用 app.model_manager.get_available_models() 获取可用模型字典
        2. 返回 JSON，格式: {
             'success': True,
             'data': {
                 'models': <模型字典>,
                 'current_model': app.model_manager.get_current_model()
             }
           }
        """
        # TODO: 实现获取模型列表
        raise NotImplementedError("TODO: 请实现 get_models() 接口")
    
    @app.route('/api/model/switch', methods=['POST'])
    def switch_model():
        """
        切换模型
        
        TODO: 考生需要实现此方法 (考点17/24: 模型切换)
        
        要求：
        1. 从 request.json 获取 model_name
        2. 如果未指定，返回 400: {'success': False, 'message': '未指定模型'}
        3. 调用 app.model_manager.switch_model(model_name)
        4. 返回: {'success': True, 'message': f'已切换到模型: {model_name}', 'current_model': model_name}
        5. 用 try/except 包裹，异常时返回 500
        """
        # TODO: 实现模型切换
        raise NotImplementedError("TODO: 请实现 switch_model() 接口")
    
    @app.route('/api/recognize', methods=['POST'])
    def recognize():
        """
        单图表情识别（含人脸检测可视化）
        
        TODO: 考生需要实现此方法 (考点18/24: 单图识别)
        
        要求：
        1. 获取图像：
           - 如果有 'image' in request.files：Image.open(image_file).convert('RGB')
           - 否则如果有 'image_base64' in request.json：base64解码后 Image.open
           - 否则 400 错误
        2. 检查模型：app.model_manager.get_inferrer()，为 None 则 500
        3. 人脸检测：
           - image_np = np.array(image_pil)
           - image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
           - faces = app.face_detector.detect(image_bgr)
        4. 构建 response_data 结构
        5. 如果 faces 为空：用 inferrer.predict(image_pil) 整体推理
        6. 如果有 faces：遍历每张人脸：
           - bbox = [int(v) for v in face_info['bbox']]
           - face_crop_bgr = app.face_detector.extract_face(image_bgr, bbox)
           - face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
           - result = inferrer.predict(face_crop_rgb)
           - 构建 face_data = {'bbox', 'confidence', 'expression', 'label', 'confidence_emotion', 'probabilities', 'keypoints'}
           - 追加到 response_data['data']['faces']
        7. 返回 JSON，异常返回 500
        
        关键提示：
        - face's bbox: [int(v) for v in face_info['bbox']]
        - face_info['confidence']: float
        - keypoints 需要将 numpy 类型转为 Python 原生类型
        - 检测器输入 BGR，推理器输入 RGB
        """
        # TODO: 实现单图表情识别
        raise NotImplementedError("TODO: 请实现 recognize() 接口")
    
    @app.route('/api/recognize/batch', methods=['POST'])
    def batch_recognize():
        """
        批量表情识别
        
        TODO: 考生需要实现此方法 (考点19/24: 批量识别)
        
        要求：
        1. 从 request.json 获取 images 列表
        2. 如果为空，返回 400
        3. 检查模型：app.model_manager.get_inferrer()，为 None 则 500
        4. 遍历解码图像：
           img_data = base64.b64decode(img_info.get('data', img_info.get('image_base64', '')))
           image = Image.open(io.BytesIO(img_data)).convert('RGB')
           追加到 images 列表
        5. 调用 inferrer.batch_predict(images) 批量推理
        6. 返回 JSON: {'success': True, 'data': {'results': [...], 'total': ...}}
        7. 异常返回 500
        """
        # TODO: 实现批量表情识别
        raise NotImplementedError("TODO: 请实现 batch_recognize() 接口")
    
    @app.route('/api/detect', methods=['POST'])
    def detect_faces():
        """
        人脸检测
        
        TODO: 考生需要实现此方法 (考点20/24: 人脸检测)
        
        要求：
        1. 获取图像（同 recognize 接口的获取方式）
        2. image_np = np.array(image)
        3. faces = app.face_detector.detect(image_np)
        4. 返回: {'success': True, 'data': {'faces': [{'bbox', 'confidence', 'keypoints'}, ...], 'total': len(faces)}}
        5. 异常返回 500
        """
        # TODO: 实现人脸检测
        raise NotImplementedError("TODO: 请实现 detect_faces() 接口")
    
    @app.route('/api/detector', methods=['GET'])
    def get_detector():
        """
        获取当前检测器信息
        
        TODO: 考生需要实现此方法 (考点21/24: 检测器信息)
        
        要求：
        1. 从 app.face_detector 获取 detector_type
        2. 返回: {'success': True, 'data': {'detector_type': det.detector_type}}
        """
        # TODO: 实现获取检测器信息
        raise NotImplementedError("TODO: 请实现 get_detector() 接口")
    
    @app.route('/api/detector/switch', methods=['POST'])
    def switch_detector():
        """
        切换人脸检测器
        
        TODO: 考生需要实现此方法 (考点22/24: 检测器切换)
        
        要求：
        1. from src.inference.detector import FaceDetector
        2. 从 request.json 获取 detector_type（默认 'insightface'）
        3. 校验类型只支持 'insightface' 或 'opencv'，不支持则 400
        4. 创建 FaceDetector(detector_type=det_type, insightface_root=app.config['ONNX_DIR'])
        5. app.face_detector = new_det
        6. 返回: {'success': True, 'message': ..., 'detector_type': det_type}
        7. 异常返回 500
        """
        # TODO: 实现检测器切换
        raise NotImplementedError("TODO: 请实现 switch_detector() 接口")
    
    @app.route('/api/benchmark', methods=['GET'])
    def benchmark():
        """
        性能测试
        
        TODO: 考生需要实现此方法 (考点23/24: 性能测试)
        
        要求：
        1. 获取推理器：app.model_manager.get_inferrer()，为 None 则 500
        2. 读取 iterations 参数（默认 100）：request.args.get('iterations', 100, type=int)
        3. 调用 inferrer.benchmark(num_iterations)
        4. 返回: {'success': True, 'data': results}
        5. 异常返回 500
        """
        # TODO: 实现性能测试
        raise NotImplementedError("TODO: 请实现 benchmark() 接口")
    
    @app.route('/api/docs', methods=['GET'])
    def docs():
        """
        API文档
        
        TODO: 考生需要实现此方法 (考点24/24: API文档)
        
        要求：
        返回 JSON，包含所有已实现接口的路径、方法、描述和参数信息。
        接口列表: health, models, model/switch, recognize, recognize/batch, detect, detector, detector/switch, benchmark
        """
        # TODO: 实现API文档
        raise NotImplementedError("TODO: 请实现 docs() 接口")