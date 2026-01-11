# src/utils/file_utils.py
import os
import tempfile
from pathlib import Path
import cv2


def create_directories():
    """创建必要的目录结构"""
    directories = [
        'data/standard_videos',
        'data/user_videos',
        'data/samples',
        'output/results',
        'output/reports',
        'assets/images',
        'assets/css'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("✅ 目录结构创建完成")


def save_uploaded_video(uploaded_file, video_type="user"):
    """保存上传的视频文件"""
    try:
        # 创建临时文件
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        # 验证视频文件
        if not is_valid_video(temp_path):
            os.unlink(temp_path)
            raise ValueError("无效的视频文件")

        return temp_path

    except Exception as e:
        raise ValueError(f"视频保存失败: {str(e)}")


def is_valid_video(video_path):
    """验证视频文件是否有效"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        # 检查是否可以读取帧
        ret, frame = cap.read()
        cap.release()

        return ret and frame is not None
    except:
        return False


def cleanup_temp_files(file_paths):
    """清理临时文件"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass  # 忽略删除错误


def get_video_info(video_path):
    """获取视频文件信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }

    cap.release()
    return info