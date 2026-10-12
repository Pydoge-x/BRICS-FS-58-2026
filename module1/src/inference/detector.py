"""
人脸检测模块
支持 InsightFace（默认）、MTCNN 和 OpenCV 三种检测器
InsightFace 提供高精度人脸检测 + 5点关键点定位
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
import os
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    人脸检测器
    默认使用 InsightFace (buffalo_l) 提供高精度检测
    同时兼容 MTCNN 和 OpenCV Haar Cascade
    """
    
    # InsightFace 5点关键点映射
    INSIGHTFACE_KP_NAMES = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
    
    def __init__(
        self,
        detector_type: str = 'insightface',
        model_path: Optional[str] = None,
        min_face_size: int = 20,
        scale_factor: float = 0.709,
        det_size: Tuple[int, int] = (640, 640),
        ctx_id: int = 0,
        insightface_root: Optional[str] = None
    ):
        """
        初始化人脸检测器
        
        Args:
            detector_type: 检测器类型 ('insightface', 'opencv', 'mtcnn')
            model_path: 模型路径（OpenCV使用）
            min_face_size: 最小人脸尺寸（OpenCV使用）
            scale_factor: 缩放因子（OpenCV使用）
            det_size: InsightFace 检测输入尺寸
            ctx_id: InsightFace 设备ID（0=GPU, -1=CPU）
            insightface_root: InsightFace 模型根目录（默认 ~/.insightface）
        """
        self.detector_type = detector_type
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self.det_size = det_size
        self.insightface_root = insightface_root
        
        if detector_type == 'insightface':
            self._init_insightface(ctx_id)
        elif detector_type == 'opencv':
            self._init_opencv()
        elif detector_type == 'mtcnn':
            self._init_mtcnn()
        else:
            raise ValueError(f"不支持的检测器类型: {detector_type}")
    
    def _init_insightface(self, ctx_id: int = 0):
        """初始化 InsightFace 检测器"""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            # 设置模型根目录
            root = self.insightface_root if self.insightface_root else '~/.insightface'
            
            self.insightface_app = FaceAnalysis(
                name='buffalo_l',
                root=root,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.insightface_app.prepare(ctx_id=ctx_id, det_size=self.det_size)
            logger.info(f"InsightFace 初始化成功 (buffalo_l, root={root}, det_size={self.det_size})")
            
        except ImportError:
            logger.warning("insightface 库未安装，回退到 OpenCV 检测器")
            self.detector_type = 'opencv'
            self._init_opencv()
        except Exception as e:
            logger.warning(f"InsightFace 初始化失败 ({e})，回退到 OpenCV 检测器")
            self.detector_type = 'opencv'
            self._init_opencv()
    
    def _init_opencv(self):
        """初始化 OpenCV 检测器"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        # OpenCV 要求 scaleFactor > 1.0
        if self.scale_factor <= 1.0:
            self.scale_factor = 1.1
        self.detector_type = 'opencv'
        logger.info("OpenCV Haar Cascade 检测器初始化成功")
    
    def _init_mtcnn(self):
        """初始化 MTCNN 检测器"""
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
            logger.info("MTCNN 检测器初始化成功")
        except ImportError:
            logger.warning("mtcnn 库未安装，回退到 InsightFace")
            self._init_insightface()
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        检测人脸
        
        Args:
            image: 输入图像 (BGR 或 RGB numpy数组)
        
        Returns:
            人脸信息列表，每项包含 bbox, confidence, keypoints
        """
        if self.detector_type == 'insightface':
            return self._detect_insightface(image)
        elif self.detector_type == 'opencv':
            return self._detect_opencv(image)
        elif self.detector_type == 'mtcnn':
            return self._detect_mtcnn(image)
    
    def _detect_insightface(self, image: np.ndarray) -> List[Dict]:
        """InsightFace 人脸检测（buffalo_l 模型）
        
        InsightFace 返回的 bbox 格式为 [x1, y1, x2, y2]
        关键点 kps 为 (5, 2) 数组:
          [0]=left_eye, [1]=right_eye, [2]=nose, [3]=mouth_left, [4]=mouth_right
        """
        # InsightFace 需要 RGB 图像
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        faces = self.insightface_app.get(rgb_image)
        
        results = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int).tolist()
            w = x2 - x1
            h = y2 - y1
            
            # 转换关键点格式 (numpy -> dict)
            keypoints = None
            if hasattr(face, 'kps') and face.kps is not None:
                keypoints = {}
                for i, name in enumerate(self.INSIGHTFACE_KP_NAMES):
                    if i < len(face.kps):
                        keypoints[name] = tuple(int(v) for v in face.kps[i])
            
            results.append({
                'bbox': [x1, y1, w, h],
                'confidence': float(face.det_score),
                'keypoints': keypoints
            })
        
        return results
    
    def _detect_opencv(self, image: np.ndarray) -> List[Dict]:
        """OpenCV Haar Cascade 人脸检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        results = []
        for (x, y, w, h) in faces:
            results.append({
                'bbox': [x, y, w, h],
                'confidence': 1.0,
                'keypoints': None
            })
        
        return results
    
    def _detect_mtcnn(self, image: np.ndarray) -> List[Dict]:
        """MTCNN人脸检测"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        faces = self.detector.detect_faces(rgb_image)
        
        results = []
        for face in faces:
            results.append({
                'bbox': face['box'],
                'confidence': face['confidence'],
                'keypoints': face.get('keypoints', None)
            })
        
        return results
    
    def extract_face(
        self,
        image: np.ndarray,
        bbox: List[int],
        target_size: Tuple[int, int] = (224, 224),
        margin: float = 0.2
    ) -> np.ndarray:
        """
        提取人脸区域
        
        Args:
            image: 输入图像
            bbox: 人脸边界框 [x, y, w, h]
            target_size: 目标尺寸
            margin: 边缘扩展比例
        
        Returns:
            人脸图像
        """
        x, y, w, h = bbox
        
        # 扩展边界
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        # 提取人脸
        face = image[y1:y2, x1:x2]
        
        # 调整大小
        face = cv2.resize(face, target_size)
        
        return face
    
    def align_face(
        self,
        image: np.ndarray,
        keypoints: Dict
    ) -> np.ndarray:
        """
        人脸对齐
        
        Args:
            image: 输入图像
            keypoints: 关键点
        
        Returns:
            对齐后的人脸图像
        """
        if keypoints is None:
            return image
        
        # 使用眼睛关键点进行对齐
        left_eye = keypoints.get('left_eye')
        right_eye = keypoints.get('right_eye')
        
        if left_eye is None or right_eye is None:
            return image
        
        # 计算旋转角度
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # 计算旋转中心
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        
        # 旋转图像
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        return aligned
    
    def detect_and_extract(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
        align: bool = True
    ) -> List[np.ndarray]:
        """
        检测并提取人脸
        
        Args:
            image: 输入图像
            target_size: 目标尺寸
            align: 是否对齐
        
        Returns:
            人脸图像列表
        """
        faces_info = self.detect(image)
        
        extracted_faces = []
        for face_info in faces_info:
            bbox = face_info['bbox']
            keypoints = face_info.get('keypoints')
            
            # 提取人脸
            face = self.extract_face(image, bbox, target_size)
            
            # 对齐
            if align and keypoints:
                face = self.align_face(face, keypoints)
            
            extracted_faces.append(face)
        
        return extracted_faces
    
    def draw_faces(
        self,
        image: np.ndarray,
        faces_info: List[Dict],
        draw_keypoints: bool = True
    ) -> np.ndarray:
        """
        在图像上绘制人脸检测结果
        
        Args:
            image: 输入图像
            faces_info: 人脸信息
            draw_keypoints: 是否绘制关键点
        
        Returns:
            绘制后的图像
        """
        result = image.copy()
        
        for face_info in faces_info:
            bbox = face_info['bbox']
            x, y, w, h = bbox
            
            # 绘制边界框
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制关键点
            if draw_keypoints and face_info.get('keypoints'):
                keypoints = face_info['keypoints']
                for name, point in keypoints.items():
                    cv2.circle(result, point, 2, (0, 0, 255), -1)
        
        return result