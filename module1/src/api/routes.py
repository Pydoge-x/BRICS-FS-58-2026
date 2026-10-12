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
        models = app.model_manager.get_available_models()
        return jsonify({
            'success': True,
            'data': {
                'models': models,
                'current_model': app.model_manager.get_current_model()
            }
        })
    
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
        try:
            data = request.json or {}
            model_name = data.get('model_name')
            
            if not model_name:
                return jsonify({'success': False, 'message': '未指定模型'}), 400
            
            app.model_manager.switch_model(model_name)
            
            return jsonify({
                'success': True,
                'message': f'已切换到模型: {model_name}',
                'current_model': model_name
            })
        
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
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
        try:
            # 获取图像
            if 'image' in request.files:
                image_file = request.files['image']
                image_pil = Image.open(image_file).convert('RGB')
            elif request.json and 'image_base64' in request.json:
                img_data = base64.b64decode(request.json['image_base64'])
                image_pil = Image.open(io.BytesIO(img_data)).convert('RGB')
            else:
                return jsonify({'success': False, 'message': '未提供图像'}), 400
            
            # 检查模型
            inferrer = app.model_manager.get_inferrer()
            if inferrer is None:
                return jsonify({'success': False, 'message': '模型未加载'}), 500
            
            # 人脸检测（所有检测器统一使用 BGR 输入）
            image_np = np.array(image_pil)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            try:
                faces = app.face_detector.detect(image_bgr)
            except Exception as det_err:
                print(f"人脸检测异常，使用全局推理: {det_err}")
                faces = []
            
            response_data = {
                'success': True,
                'data': {
                    'faces_detected': len(faces),
                    'faces': [],
                    'process_time': 0,
                    'model': inferrer.model_type
                }
            }
            
            if len(faces) == 0:
                # 无人脸时仍进行整体推理
                result = inferrer.predict(image_pil)
                response_data['data'].update({
                    'faces_detected': 0,
                    'expression': result['expression'],
                    'label': result['label'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities'],
                    'process_time': result['process_time']
                })
                return jsonify(response_data)
            
            # 对每张检测到的人脸进行识别
            for face_info in faces:
                bbox = [int(v) for v in face_info['bbox']]
                face_crop_bgr = app.face_detector.extract_face(image_bgr, bbox)
                # 推理器需要 RGB 输入
                face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                result = inferrer.predict(face_crop_rgb)
                
                # 将 keypoints 中的 numpy 类型转为 Python 原生类型
                keypoints = None
                if face_info.get('keypoints'):
                    keypoints = {
                        k: [int(v[0]), int(v[1])]
                        for k, v in face_info['keypoints'].items()
                    }
                
                face_data = {
                    'bbox': bbox,
                    'confidence': float(face_info['confidence']),
                    'expression': result['expression'],
                    'label': result['label'],
                    'confidence_emotion': result['confidence'],
                    'probabilities': result['probabilities'],
                    'keypoints': keypoints
                }
                response_data['data']['faces'].append(face_data)
            
            response_data['data']['process_time'] = result.get('process_time', 0)
            
            return jsonify(response_data)
        
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
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
        try:
            data = request.json or {}
            images_data = data.get('images', [])
            
            if not images_data:
                return jsonify({'success': False, 'message': '未提供图像'}), 400
            
            # 检查模型
            inferrer = app.model_manager.get_inferrer()
            if inferrer is None:
                return jsonify({'success': False, 'message': '模型未加载'}), 500
            
            # 解码图像
            images = []
            for img_info in images_data:
                img_data = base64.b64decode(img_info.get('data', img_info.get('image_base64', '')))
                image = Image.open(io.BytesIO(img_data)).convert('RGB')
                images.append(image)
            
            # 批量推理
            results = inferrer.batch_predict(images)
            
            return jsonify({
                'success': True,
                'data': {
                    'results': [
                        {
                            'id': images_data[i].get('id', i),
                            'expression': r['expression'],
                            'confidence': r['confidence'],
                            'probabilities': r['probabilities']
                        }
                        for i, r in enumerate(results)
                    ],
                    'total': len(results),
                    'batch_time': results[0]['batch_time'],
                    'avg_time': results[0]['avg_time']
                }
            })
        
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
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
        try:
            # 获取图像
            if 'image' in request.files:
                image_file = request.files['image']
                image = Image.open(image_file)
            elif request.json and 'image_base64' in request.json:
                img_data = base64.b64decode(request.json['image_base64'])
                image = Image.open(io.BytesIO(img_data))
            else:
                return jsonify({'success': False, 'message': '未提供图像'}), 400
            
            image_np = np.array(image)
            
            # 人脸检测
            faces = app.face_detector.detect(image_np)
            
            return jsonify({
                'success': True,
                'data': {
                    'faces': [
                        {
                            'bbox': f['bbox'],
                            'confidence': f['confidence'],
                            'keypoints': f.get('keypoints')
                        }
                        for f in faces
                    ],
                    'total': len(faces)
                }
            })
        
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    @app.route('/api/detector', methods=['GET'])
    def get_detector():
        """
        获取当前检测器信息
        
        TODO: 考生需要实现此方法 (考点21/24: 检测器信息)
        
        要求：
        1. 从 app.face_detector 获取 detector_type
        2. 返回: {'success': True, 'data': {'detector_type': det.detector_type}}
        """
        det = app.face_detector
        return jsonify({
            'success': True,
            'data': {
                'detector_type': det.detector_type
            }
        })
    
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
        try:
            from src.inference.detector import FaceDetector
            data = request.json or {}
            det_type = data.get('detector_type', 'insightface')
            
            if det_type not in ('insightface', 'opencv'):
                return jsonify({'success': False, 'message': '不支持的检测器类型'}), 400
            
            try:
                new_det = FaceDetector(
                    detector_type=det_type,
                    insightface_root=app.config['ONNX_DIR']
                )
            except Exception as e:
                return jsonify({'success': False, 'message': f'初始化失败: {e}'}), 500
            
            app.face_detector = new_det
            print(f"人脸检测器已切换为: {det_type}")
            
            return jsonify({
                'success': True,
                'message': f'已切换到检测器: {det_type}',
                'detector_type': det_type
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
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
        try:
            inferrer = app.model_manager.get_inferrer()
            if inferrer is None:
                return jsonify({'success': False, 'message': '模型未加载'}), 500
            
            num_iterations = request.args.get('iterations', 100, type=int)
            results = inferrer.benchmark(num_iterations)
            
            return jsonify({
                'success': True,
                'data': results
            })
        
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    @app.route('/api/docs', methods=['GET'])
    def docs():
        """
        API文档
        
        TODO: 考生需要实现此方法 (考点24/24: API文档)
        
        要求：
        返回 JSON，包含所有已实现接口的路径、方法、描述和参数信息。
        接口列表: health, models, model/switch, recognize, recognize/batch, detect, detector, detector/switch, benchmark
        """
        return jsonify({
            'title': 'Face Expression Recognition API',
            'version': '1.0.0',
            'endpoints': [
                {
                    'path': '/api/health',
                    'method': 'GET',
                    'description': '健康检查'
                },
                {
                    'path': '/api/models',
                    'method': 'GET',
                    'description': '获取可用模型列表'
                },
                {
                    'path': '/api/model/switch',
                    'method': 'POST',
                    'description': '切换模型',
                    'params': {'model_name': '模型名称'}
                },
                {
                    'path': '/api/recognize',
                    'method': 'POST',
                    'description': '单图表情识别',
                    'params': {'image': '图像文件或base64'}
                },
                {
                    'path': '/api/recognize/batch',
                    'method': 'POST',
                    'description': '批量表情识别',
                    'params': {'images': '图像列表'}
                },
                {
                    'path': '/api/detect',
                    'method': 'POST',
                    'description': '人脸检测',
                    'params': {'image': '图像文件或base64'}
                },
                {
                    'path': '/api/detector',
                    'method': 'GET',
                    'description': '获取当前检测器信息'
                },
                {
                    'path': '/api/detector/switch',
                    'method': 'POST',
                    'description': '切换人脸检测器',
                    'params': {'detector_type': 'insightface 或 opencv'}
                },
                {
                    'path': '/api/benchmark',
                    'method': 'GET',
                    'description': '性能测试',
                    'params': {'iterations': '测试次数'}
                }
            ]
        })
