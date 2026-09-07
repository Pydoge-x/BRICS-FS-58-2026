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
        """获取可用模型列表"""
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
        """切换模型"""
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
        """单图表情识别（含人脸检测可视化）"""
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
        """批量表情识别"""
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
        """人脸检测"""
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
        """获取当前检测器信息"""
        det = app.face_detector
        return jsonify({
            'success': True,
            'data': {
                'detector_type': det.detector_type
            }
        })
    
    @app.route('/api/detector/switch', methods=['POST'])
    def switch_detector():
        """切换人脸检测器"""
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
        """性能测试"""
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
        """API文档"""
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
