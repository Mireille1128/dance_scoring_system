# config.py
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 路径配置
DATA_DIR = PROJECT_ROOT / "data"
STANDARD_VIDEOS_DIR = DATA_DIR / "standard_videos"
USER_VIDEOS_DIR = DATA_DIR / "user_videos"
OUTPUT_DIR = PROJECT_ROOT / "output"
ASSETS_DIR = PROJECT_ROOT / "assets"

# 创建必要目录
for directory in [STANDARD_VIDEOS_DIR, USER_VIDEOS_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# MediaPipe配置
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'model_complexity': 1,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'smooth_landmarks': True,
    'enable_segmentation': False,
    'smooth_segmentation': True
}
# 视频处理配置
# config.py
VIDEO_CONFIG = {
    "max_frames": 1000,  # 最大处理帧数
    "target_fps": 30,    # 目标帧率
    "min_duration": 3,   # 最小时长（秒）
    "max_duration": 60   # 最大时长（秒）
}

# 评分算法配置
SCORING_CONFIG = {
    "similarity_threshold": 0.6,  # 相似度阈值
    "weight_pose": 0.4,           # 姿态权重
    "weight_rhythm": 0.3,         # 节奏权重
    "weight_smoothness": 0.3       # 流畅度权重
}

# Streamlit界面配置
UI_CONFIG = {
    "page_title": "💃 智能舞蹈评分系统",
    "page_icon": "💃",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}