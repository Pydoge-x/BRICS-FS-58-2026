"""
Flask API服务模块
表情识别推理服务
"""

from .app import create_app, run_app
from .routes import setup_routes

__all__ = ['create_app', 'run_app', 'setup_routes']