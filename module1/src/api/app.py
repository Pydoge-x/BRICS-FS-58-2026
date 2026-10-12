"""
Flask应用主文件
表情识别推理服务 + 前端管理界面
"""

import os
from flask import Flask, send_from_directory
from flask_cors import CORS

from .routes import setup_routes
from .model_manager import ModelManager


def create_app(config: dict = None) -> Flask:
    """
    创建Flask应用
    
    Args:
        config: 配置字典
    
    Returns:
        Flask应用
    """
    app = Flask(__name__, static_folder='../../frontend', static_url_path='/static')
    
    # CORS配置
    CORS(app)
    
    # 默认配置
    app.config.update({
        'MODEL_DIR': 'checkpoints',
        'ONNX_DIR': 'models',
        'DEFAULT_MODEL': 'resnet',
        'NUM_THREADS': 4,
        'MAX_BATCH_SIZE': 32,
        'SECRET_KEY': 'fer-secret-key-2026'
    })
    
    # 更新配置
    if config:
        app.config.update(config)
    
    # 初始化模型管理器
    app.model_manager = ModelManager(
        model_dir=app.config['MODEL_DIR'],
        onnx_dir=app.config['ONNX_DIR']
    )
    
    # 预转换所有模型为ONNX
    app.model_manager.pre_convert_all()
    
    # 预先初始化 InsightFace 人脸检测器（服务启动时完成，避免首次请求用低精度 OpenCV）
    from src.inference.detector import FaceDetector
    try:
        app.face_detector = FaceDetector(
            detector_type='insightface',
            insightface_root=app.config['ONNX_DIR']  # 模型下载到 models/ 目录
        )
        print(f"人脸检测器: InsightFace 已就绪 (模型目录: {app.config['ONNX_DIR']})")
    except Exception as e:
        print(f"InsightFace 不可用 ({e})，回退到 OpenCV")
        app.face_detector = FaceDetector(detector_type='opencv')
    
    # 加载默认模型
    try:
        available = app.model_manager.available_models
        if app.config['DEFAULT_MODEL'] in available:
            app.model_manager.load_model(app.config['DEFAULT_MODEL'])
        elif available:
            first_model = list(available.keys())[0]
            print(f"默认模型 {app.config['DEFAULT_MODEL']} 不可用，使用 {first_model}")
            app.model_manager.load_model(first_model)
    except Exception as e:
        print(f"警告: 无法加载默认模型: {e}")
    
    # 前端页面路由
    @app.route('/')
    def serve_frontend():
        """管理前端首页"""
        return send_from_directory(app.static_folder, 'index.html')
    
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        """静态文件"""
        return send_from_directory(app.static_folder, filename)
    
    # 设置API路由
    setup_routes(app)
    
    return app


def run_app(
    host: str = '0.0.0.0',
    port: int = 5000,
    debug: bool = False,
    config: dict = None
):
    """
    运行Flask应用
    
    Args:
        host: 主机地址
        port: 端口
        debug: 是否调试模式
        config: 配置字典
    """
    app = create_app(config)
    
    print(f"\n{'='*60}")
    print(f"  表情识别服务")
    print(f"  管理界面: http://{host}:{port}")
    print(f"  API文档:  http://{host}:{port}/api/docs")
    print(f"  健康检查:  http://{host}:{port}/api/health")
    print(f"{'='*60}\n")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_app()
