# src/video_processor.py
import cv2
import numpy as np
import os
# from tqdm import tqdm  # 移除这行
from config import VIDEO_CONFIG


class VideoProcessor:
    """视频处理器 - 处理视频文件并提取姿态序列"""

    def __init__(self, pose_estimator, config=None):
        self.pose_estimator = pose_estimator
        self.config = config or VIDEO_CONFIG

    def process_video(self, video_path, max_frames=None):
        """处理视频文件，提取姿态关键点序列"""
        print(f"📹 处理视频: {video_path}")

        # 验证文件存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cap = None
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")

            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0

            max_frames = max_frames or self.config.get("max_frames", 1000)
            total_frames = min(total_frames, max_frames)

            print(f"  分辨率: {width}x{height}")
            print(f"  帧率: {fps:.1f} FPS")
            print(f"  总帧数: {total_frames}")
            print(f"  时长: {duration:.1f}秒")

            # 处理视频帧
            landmarks_sequence = []
            valid_frames = 0

            # 使用简单的进度显示，不使用tqdm
            print("  处理进度: ", end="", flush=True)
            progress_interval = max(1, total_frames // 10)  # 每10%显示一次

            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # 处理当前帧
                landmarks = self.pose_estimator.process_frame(frame)
                if landmarks is not None:
                    landmarks_sequence.append(landmarks)
                    valid_frames += 1

                # 显示简单进度
                if frame_idx % progress_interval == 0:
                    print("█", end="", flush=True)

            cap.release()
            print()  # 换行

            print(f"✅ 视频处理完成，提取到 {len(landmarks_sequence)} 帧有效姿态数据")

            # 统一返回格式
            return {
                'keypoints': np.array(landmarks_sequence),
                'landmarks_sequence': np.array(landmarks_sequence),
                'frame_indices': np.arange(len(landmarks_sequence)),
                'fps': fps,
                'total_frames': total_frames,
                'processed_frames': valid_frames,
                'detection_rate': valid_frames / total_frames if total_frames > 0 else 0,
                'video_path': video_path,
                'video_name': os.path.basename(video_path),
                'resolution': (width, height),
                'duration': duration,
                'video_info': {
                    'fps': fps,
                    'total_frames': total_frames,
                    'duration': duration,
                    'valid_frames': valid_frames,
                    'detection_rate': valid_frames / total_frames if total_frames > 0 else 0
                }
            }

        except Exception as e:
            if cap is not None:
                cap.release()
            raise e