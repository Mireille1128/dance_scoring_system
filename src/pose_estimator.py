# src/pose_estimator.py
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

try:
    from config import MEDIAPIPE_CONFIG
except ImportError:
    # 默认配置
    MEDIAPIPE_CONFIG = {
        'static_image_mode': False,
        'model_complexity': 1,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5
    }


class PoseEstimator:
    """姿态估计器 - 核心模块（优化版）"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """初始化姿态估计模型"""
        self.config = config or MEDIAPIPE_CONFIG

        # 初始化MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 创建模型实例
        self.pose = self.mp_pose.Pose(**self.config)

        # 缓存
        self._last_results = None
        self._frame_count = 0

    def process_frame(self, image: np.ndarray) -> Optional[np.ndarray]:
        """处理单帧图像，提取姿态关键点"""
        if image is None:
            print("❌ 输入图像为空")
            return None

        self._frame_count += 1

        try:
            # 1. 预处理图像
            processed_image = self._preprocess_image(image)

            # 2. 转换颜色空间
            image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False  # 提高性能

            # 3. 姿态估计
            results = self.pose.process(image_rgb)

            # 4. 处理结果
            if results.pose_landmarks:
                landmarks = self._landmarks_to_array(results.pose_landmarks.landmark)
                self._last_results = (landmarks, results)

                # 调试信息（每100帧打印一次）
                if self._frame_count % 100 == 0:
                    print(f"✅ 已处理 {self._frame_count} 帧，检测到姿态")

                return landmarks
            else:
                if self._frame_count % 100 == 0:
                    print(f"⚠️ 第 {self._frame_count} 帧未检测到姿态")
                return None

        except Exception as e:
            print(f"❌ 姿态估计错误 (第{self._frame_count}帧): {e}")
            return None

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        # 调整大小
        resized = self._resize_image(image, target_width=640)

        # 可以添加其他预处理（如对比度增强、降噪等）
        # 但注意：MediaPipe对输入图像质量要求不高

        return resized

    def _resize_image(self, image: np.ndarray, target_width: int = 640) -> np.ndarray:
        """调整图像大小，保持宽高比"""
        h, w = image.shape[:2]

        # 如果宽度已经小于目标宽度，不调整
        if w <= target_width:
            return image

        scale = target_width / w
        new_width = target_width
        new_height = int(h * scale)

        # 确保高度是偶数（某些编解码器要求）
        if new_height % 2 != 0:
            new_height += 1

        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def _landmarks_to_array(self, landmarks) -> np.ndarray:
        """将MediaPipe landmarks转换为numpy数组 [33, 4]"""
        landmarks_array = []
        for landmark in landmarks:
            landmarks_array.append([
                landmark.x,  # 归一化x坐标
                landmark.y,  # 归一化y坐标
                landmark.z,  # 归一化z坐标
                landmark.visibility  # 可见性得分
            ])
        return np.array(landmarks_array)

    def get_xyz_coordinates(self, landmarks_array: np.ndarray) -> np.ndarray:
        """获取x,y,z坐标（去掉visibility）"""
        if landmarks_array is None:
            return None
        return landmarks_array[:, :3]  # 只取前3列

    def is_valid_pose(self, landmarks_array: np.ndarray, min_visible_points: int = 20) -> bool:
        """检查姿态是否有效"""
        if landmarks_array is None:
            return False

        # 检查可见性（第4个维度是visibility）
        visible_points = np.sum(landmarks_array[:, 3] > 0.5)
        return visible_points >= min_visible_points

    def draw_landmarks(self, image: np.ndarray, landmarks_array: Optional[np.ndarray] = None,
                       results: Optional[Any] = None) -> np.ndarray:
        """在图像上绘制关键点和骨骼连接"""
        if image is None:
            return image

        annotated_image = image.copy()

        if results and results.pose_landmarks:
            # 使用MediaPipe的绘图功能
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        elif landmarks_array is not None:
            # 手动绘制关键点
            h, w = image.shape[:2]

            # 绘制关键点
            for i, landmark in enumerate(landmarks_array):
                x = int(landmark[0] * w)
                y = int(landmark[1] * h)

                # 根据关键点类型选择颜色
                if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # 面部
                    color = (255, 0, 0)  # 蓝色
                elif i in [11, 12, 23, 24]:  # 躯干
                    color = (0, 255, 0)  # 绿色
                else:  # 四肢
                    color = (0, 0, 255)  # 红色

                cv2.circle(annotated_image, (x, y), 4, color, -1)

                # 可选：显示关键点索引（调试用）
                # cv2.putText(annotated_image, str(i), (x+5, y-5),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

        return annotated_image

    def reset(self) -> None:
        """重置状态"""
        self._frame_count = 0
        self._last_results = None

    def __del__(self) -> None:
        """释放资源"""
        try:
            if hasattr(self, 'pose'):
                self.pose.close()
        except:
            pass