"""启动表情识别 API 服务"""
import sys
sys.path.insert(0, '.')
from src.api.app import run_app
run_app(port=5000, debug=False)
