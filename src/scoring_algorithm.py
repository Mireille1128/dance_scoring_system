# src/scoring_algorithm.py
"""
舞蹈标准对比评分算法
对比学习者视频与标准舞蹈视频，提供改进建议
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import mediapipe as mp
from dataclasses import dataclass
from enum import Enum


class ScoreCategory(Enum):
    """评分维度分类"""
    TIMING = "timing"  # 节奏与时机
    POSITION = "position"  # 位置与方向
    MOVEMENT = "movement"  # 动作幅度
    POSTURE = "posture"  # 姿势标准度
    EXPRESSION = "expression"  # 表情与情感


@dataclass
class FrameAnalysis:
    """单帧分析结果"""
    frame_idx: int
    timing_score: float  # 节奏得分
    position_score: float  # 位置得分
    movement_score: float  # 动作得分
    posture_score: float  # 姿势得分
    landmarks: List  # 关键点数据


class StandardDanceScorer:
    """标准舞蹈对比评分器"""

    def __init__(self, config=None):
        """
        初始化评分器

        Args:
            config (dict, optional): 配置参数
        """
        self.config = config or {}

        # 初始化MediaPipe姿势检测
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 简化模型提高速度
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_drawing = mp.solutions.drawing_utils

        # 标准视频特征（将在加载标准视频时设置）
        self.standard_features = None
        self.standard_frames_data = None

        # 评分权重
        self.weights = {
            'timing': 0.30,  # 节奏时机 30%
            'position': 0.25,  # 位置方向 25%
            'movement': 0.20,  # 动作幅度 20%
            'posture': 0.15,  # 姿势标准 15%
            'expression': 0.10  # 表情表现 10%
        }

    def load_standard_video(self, video_path: str) -> Dict:
        """
        加载标准舞蹈视频并提取特征

        Args:
            video_path (str): 标准视频路径

        Returns:
            dict: 标准视频特征
        """
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise ValueError(f"无法打开标准视频: {video_path}")

            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if frame_count == 0:
                raise ValueError("视频帧数为0")

            # 提取标准视频特征
            standard_frames = []

            frame_idx = 0
            sample_rate = max(1, int(fps / 5))  # 每秒采样5帧，提高速度

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    # 分析当前帧
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.pose.process(rgb_frame)

                    if results.pose_landmarks:
                        landmarks = self._extract_normalized_landmarks(results.pose_landmarks)

                        # 提取帧特征
                        frame_features = self._extract_frame_features(landmarks)
                        standard_frames.append({
                            'frame_idx': frame_idx,
                            'timestamp': frame_idx / fps if fps > 0 else 0,
                            'features': frame_features,
                            'landmarks': landmarks
                        })

                frame_idx += 1
                if frame_idx > 300:  # 只分析前300帧，提高速度
                    break

            cap.release()

            if not standard_frames:
                return {
                    'success': False,
                    'error': '无法从标准视频中提取姿势数据，请确保视频中有清晰的人物舞蹈动作'
                }

            # 计算整体标准特征
            self.standard_frames_data = standard_frames
            self.standard_features = {
                'video_info': {
                    'path': video_path,
                    'frames': frame_count,
                    'fps': fps,
                    'duration': frame_count / fps if fps > 0 else 0,
                    'analyzed_frames': len(standard_frames)
                },
                'avg_pose': self._calculate_average_pose(standard_frames)
            }

            return {
                'success': True,
                'video_info': self.standard_features['video_info'],
                'features': self.standard_features,
                'message': f'标准视频加载成功，分析了{len(standard_frames)}帧'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _extract_normalized_landmarks(self, pose_landmarks):
        """提取归一化的关键点坐标"""
        if pose_landmarks is None:
            return []

        landmarks = []
        for lm in pose_landmarks.landmark:
            # 只提取关键点的x,y坐标和可见度
            landmarks.append((lm.x, lm.y, lm.visibility))

        return landmarks

    def _extract_frame_features(self, landmarks):
        """提取单帧特征 - 简化版本"""
        if not landmarks:
            return {}

        return {
            'body_center': self._calculate_body_center_simple(landmarks),
            'limb_angles': self._calculate_limb_angles_simple(landmarks)
        }

    def _calculate_body_center_simple(self, landmarks):
        """简化版身体中心计算"""
        if len(landmarks) < 25:
            return (0.5, 0.5)  # 默认中心点

        # 使用肩膀和臀部中点作为身体中心
        left_shoulder = landmarks[11] if len(landmarks) > 11 else (0, 0, 0)
        right_shoulder = landmarks[12] if len(landmarks) > 12 else (0, 0, 0)
        left_hip = landmarks[23] if len(landmarks) > 23 else (0, 0, 0)
        right_hip = landmarks[24] if len(landmarks) > 24 else (0, 0, 0)

        center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4
        center_y = (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4

        return (center_x, center_y)

    def _calculate_limb_angles_simple(self, landmarks):
        """简化版肢体角度计算"""
        angles = {}

        # 只计算关键角度
        if len(landmarks) >= 16:
            # 左肘角度 (11-13-15)
            if len(landmarks) > 15:
                angles['left_elbow'] = self._calculate_angle_simple(
                    landmarks[11], landmarks[13], landmarks[15]
                )

            # 右肘角度 (12-14-16)
            if len(landmarks) > 16:
                angles['right_elbow'] = self._calculate_angle_simple(
                    landmarks[12], landmarks[14], landmarks[16]
                )

        return angles

    def _calculate_angle_simple(self, p1, p2, p3):
        """简化版角度计算"""
        try:
            # 转换为numpy数组
            a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            b = np.array([p3[0] - p2[0], p3[1] - p2[1]])

            # 计算夹角
            cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            cosine_angle = np.clip(cosine_angle, -1, 1)
            angle = np.degrees(np.arccos(cosine_angle))

            return angle
        except:
            return 90.0  # 默认角度

    def _calculate_average_pose(self, frames_data):
        """计算平均姿势"""
        if not frames_data:
            return {}

        # 收集所有关键点
        all_landmarks = []
        for frame in frames_data:
            if 'landmarks' in frame and frame['landmarks']:
                all_landmarks.append(frame['landmarks'])

        if not all_landmarks:
            return {}

        # 计算每个关键点的平均值
        avg_landmarks = []
        num_frames = len(all_landmarks)
        num_points = len(all_landmarks[0])

        for i in range(num_points):
            sum_x, sum_y, sum_visibility = 0, 0, 0
            for frame_idx in range(num_frames):
                if i < len(all_landmarks[frame_idx]):
                    landmark = all_landmarks[frame_idx][i]
                    sum_x += landmark[0]
                    sum_y += landmark[1]
                    sum_visibility += landmark[2] if len(landmark) > 2 else 1

            avg_landmarks.append((
                sum_x / num_frames,
                sum_y / num_frames,
                sum_visibility / num_frames
            ))

        return avg_landmarks

    def evaluate_student_video(self, student_video_path: str) -> Dict:
        """
        评估学生视频相对于标准视频的表现

        Args:
            student_video_path (str): 学生视频路径

        Returns:
            dict: 评估结果，包含分数和改进建议
        """
        if self.standard_features is None:
            return {
                'success': False,
                'error': '请先加载标准视频'
            }

        try:
            # 分析学生视频
            student_result = self._analyze_student_video(student_video_path)

            if not student_result['success']:
                return student_result

            # 与标准视频对比
            comparison = self._compare_with_standard(
                student_result['frames_data'],
                student_result['video_info']
            )

            # 计算综合分数
            final_score = self._calculate_final_score(comparison)

            # 生成改进建议
            suggestions = self._generate_suggestions(comparison)

            # 找出关键问题点
            key_issues = self._identify_key_issues(comparison)

            return {
                'success': True,
                'student_info': student_result['video_info'],
                'overall_score': final_score['overall'],
                'category_scores': final_score['categories'],
                'detailed_comparison': comparison,
                'improvement_suggestions': suggestions,
                'key_issues': key_issues,
                'score_breakdown': self._create_score_breakdown(final_score),
                'grade': self._get_grade(final_score['overall'])
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'评估失败: {str(e)}'
            }

    def _analyze_student_video(self, video_path: str) -> Dict:
        """分析学生视频"""
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                return {'success': False, 'error': f'无法打开视频: {video_path}'}

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if frame_count == 0:
                return {'success': False, 'error': '视频帧数为0'}

            # 调整采样率以匹配标准视频
            standard_fps = self.standard_features['video_info']['fps']
            if standard_fps <= 0:
                standard_fps = 30  # 默认值

            frame_ratio = fps / standard_fps if standard_fps > 0 else 1

            student_frames = []
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 根据帧率比例采样
                if int(frame_idx * frame_ratio) % 1 == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.pose.process(rgb_frame)

                    if results.pose_landmarks:
                        landmarks = self._extract_normalized_landmarks(results.pose_landmarks)

                        student_frames.append({
                            'frame_idx': frame_idx,
                            'timestamp': frame_idx / fps if fps > 0 else 0,
                            'landmarks': landmarks,
                            'original_frame': frame_idx
                        })

                frame_idx += 1
                if frame_idx > 300:  # 只分析前300帧
                    break

            cap.release()

            if not student_frames:
                return {'success': False, 'error': '无法从学生视频中提取姿势数据，请确保视频中有清晰的人物舞蹈动作'}

            return {
                'success': True,
                'video_info': {
                    'path': video_path,
                    'frames': frame_count,
                    'fps': fps,
                    'duration': frame_count / fps if fps > 0 else 0,
                    'analyzed_frames': len(student_frames)
                },
                'frames_data': student_frames
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _compare_with_standard(self, student_frames: List, student_info: Dict) -> Dict:
        """对比学生视频与标准视频"""
        comparison = {
            'timing_errors': [],
            'position_errors': [],
            'movement_errors': [],
            'posture_errors': [],
            'frame_comparisons': [],
            'worst_frames': []
        }

        # 检查是否有标准数据
        if self.standard_frames_data is None or len(self.standard_frames_data) == 0:
            return comparison

        # 对齐时间线
        standard_frames = self.standard_frames_data
        max_comparisons = min(len(student_frames), len(standard_frames))

        if max_comparisons == 0:
            return comparison

        # 逐帧对比
        timing_scores = []
        position_scores = []
        movement_scores = []
        posture_scores = []

        for i in range(max_comparisons):
            student_frame = student_frames[i]
            standard_frame = standard_frames[min(i, len(standard_frames) - 1)]

            # 对比关键点
            student_landmarks = student_frame['landmarks']
            standard_landmarks = standard_frame['landmarks']

            if len(student_landmarks) == 0 or len(standard_landmarks) == 0:
                continue

            # 计算各项误差
            timing_score = self._calculate_timing_score(student_frame, standard_frame)
            position_score = self._calculate_position_score(student_landmarks, standard_landmarks)
            movement_score = self._calculate_movement_score(student_landmarks, standard_landmarks)
            posture_score = self._calculate_posture_score(student_landmarks, standard_landmarks)

            # 记录分数
            timing_scores.append(timing_score)
            position_scores.append(position_score)
            movement_scores.append(movement_score)
            posture_scores.append(posture_score)

            # 记录帧对比
            frame_comparison = {
                'frame_idx': i,
                'timestamp': student_frame['timestamp'],
                'timing_score': timing_score,
                'position_score': position_score,
                'movement_score': movement_score,
                'posture_score': posture_score,
                'overall_score': (timing_score + position_score + movement_score + posture_score) / 4
            }

            comparison['frame_comparisons'].append(frame_comparison)

            # 记录错误帧
            overall = frame_comparison['overall_score']
            if overall < 60:
                comparison['worst_frames'].append({
                    'timestamp': student_frame['timestamp'],
                    'score': overall,
                    'issues': self._identify_frame_issues(frame_comparison)
                })

        # 计算平均误差
        comparison['average_scores'] = {
            'timing': np.mean(timing_scores) if timing_scores else 75,
            'position': np.mean(position_scores) if position_scores else 75,
            'movement': np.mean(movement_scores) if movement_scores else 75,
            'posture': np.mean(posture_scores) if posture_scores else 75
        }

        # 找出最差的5帧
        comparison['worst_frames'] = sorted(
            comparison['worst_frames'],
            key=lambda x: x['score']
        )[:5]

        return comparison

    def _calculate_timing_score(self, student_frame: Dict, standard_frame: Dict) -> float:
        """计算节奏时机得分"""
        try:
            # 基于时间戳差异计算
            time_diff = abs(student_frame['timestamp'] - standard_frame['timestamp'])

            # 假设0.1秒内为优秀，0.3秒内为良好，超过0.5秒为差
            if time_diff < 0.1:
                return 95
            elif time_diff < 0.2:
                return 85
            elif time_diff < 0.3:
                return 75
            elif time_diff < 0.5:
                return 60
            else:
                return 40
        except:
            return 70  # 默认分

    def _calculate_position_score(self, student_landmarks: List, standard_landmarks: List) -> float:
        """计算位置方向得分"""
        if len(student_landmarks) < 10 or len(standard_landmarks) < 10:
            return 50

        # 计算关键点位置差异
        key_points = [0, 11, 12, 23, 24]  # 简化：鼻子、肩膀、臀部

        total_error = 0
        valid_points = 0

        for idx in key_points:
            if idx < len(student_landmarks) and idx < len(standard_landmarks):
                student_point = student_landmarks[idx]
                standard_point = standard_landmarks[idx]

                # 计算欧氏距离（忽略深度z）
                error = np.sqrt(
                    (student_point[0] - standard_point[0]) ** 2 +
                    (student_point[1] - standard_point[1]) ** 2
                )
                total_error += error
                valid_points += 1

        if valid_points == 0:
            return 50

        avg_error = total_error / valid_points

        # 转换为分数：误差越小分数越高
        if avg_error < 0.02:  # 2% 误差
            return 95
        elif avg_error < 0.05:  # 5% 误差
            return 85
        elif avg_error < 0.08:  # 8% 误差
            return 75
        elif avg_error < 0.12:  # 12% 误差
            return 60
        else:
            return max(30, 100 - avg_error * 500)

    def _calculate_movement_score(self, student_landmarks: List, standard_landmarks: List) -> float:
        """计算动作幅度得分"""
        if len(student_landmarks) < 16 or len(standard_landmarks) < 16:
            return 50

        # 计算四肢角度差异
        angles_to_check = [
            ('left_elbow', 11, 13, 15),
            ('right_elbow', 12, 14, 16)
        ]

        total_diff = 0
        valid_angles = 0

        for angle_name, p1_idx, p2_idx, p3_idx in angles_to_check:
            if (p1_idx < len(student_landmarks) and p2_idx < len(student_landmarks) and
                    p3_idx < len(student_landmarks)):
                student_angle = self._calculate_angle_simple(
                    student_landmarks[p1_idx],
                    student_landmarks[p2_idx],
                    student_landmarks[p3_idx]
                )

                standard_angle = self._calculate_angle_simple(
                    standard_landmarks[p1_idx],
                    standard_landmarks[p2_idx],
                    standard_landmarks[p3_idx]
                )

                angle_diff = abs(student_angle - standard_angle)
                total_diff += angle_diff
                valid_angles += 1

        if valid_angles == 0:
            return 50

        avg_angle_diff = total_diff / valid_angles

        # 转换为分数：角度差异越小分数越高
        if avg_angle_diff < 10:  # 10度以内
            return 95
        elif avg_angle_diff < 20:  # 20度以内
            return 85
        elif avg_angle_diff < 30:  # 30度以内
            return 75
        elif avg_angle_diff < 45:  # 45度以内
            return 60
        else:
            return max(30, 100 - avg_angle_diff)

    def _calculate_posture_score(self, student_landmarks: List, standard_landmarks: List) -> float:
        """计算姿势标准度得分"""
        # 简化版姿势评分
        if len(student_landmarks) < 24:
            return 70

        # 1. 脊柱垂直度
        spine_score = self._check_spine_alignment_simple(student_landmarks)

        # 2. 肩膀水平度
        shoulder_score = self._check_shoulder_level_simple(student_landmarks)

        # 综合姿势分数
        posture_score = (spine_score * 0.6 + shoulder_score * 0.4)

        return posture_score

    def _check_spine_alignment_simple(self, landmarks: List) -> float:
        """简化版脊柱垂直度检查"""
        if len(landmarks) < 24:
            return 70

        # 使用肩膀(11,12)和臀部(23,24)中点计算脊柱
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]

        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2

        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        hip_center_y = (left_hip[1] + right_hip[1]) / 2

        # 计算垂直偏差
        vertical_deviation = abs(shoulder_center_x - hip_center_x)

        if vertical_deviation < 0.02:
            return 95
        elif vertical_deviation < 0.05:
            return 85
        elif vertical_deviation < 0.08:
            return 75
        else:
            return 60

    def _check_shoulder_level_simple(self, landmarks: List) -> float:
        """简化版肩膀水平度检查"""
        if len(landmarks) < 13:
            return 70

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])

        if shoulder_diff < 0.01:
            return 95
        elif shoulder_diff < 0.03:
            return 85
        elif shoulder_diff < 0.05:
            return 75
        else:
            return 60

    def _calculate_final_score(self, comparison: Dict) -> Dict:
        """计算最终分数"""
        avg_scores = comparison['average_scores']

        # 应用权重计算总分
        total_score = 0
        category_scores = {}

        for category, weight in self.weights.items():
            if category in avg_scores:
                score = avg_scores[category]
                weighted = score * weight
                total_score += weighted
                category_scores[category] = {
                    'raw_score': score,
                    'weight': weight,
                    'weighted_score': weighted
                }

        return {
            'overall': round(total_score, 2),
            'categories': category_scores
        }

    def _generate_suggestions(self, comparison: Dict) -> List[str]:
        """生成改进建议"""
        suggestions = []
        avg_scores = comparison['average_scores']

        # 根据各项分数生成建议
        if avg_scores.get('timing', 100) < 70:
            suggestions.append("💃 **节奏感训练**：建议使用节拍器练习，加强音乐节奏感，注意动作与音乐的同步")

        if avg_scores.get('position', 100) < 70:
            suggestions.append("📍 **位置准确性**：注意身体各部位的标准位置，多对照镜子练习，确保动作到位")

        if avg_scores.get('movement', 100) < 70:
            suggestions.append("🎯 **动作幅度**：动作要更舒展，达到标准幅度要求，注意动作的完整性")

        if avg_scores.get('posture', 100) < 70:
            suggestions.append("🧘 **姿势纠正**：保持脊柱挺直，注意肩膀和骨盆的水平，加强核心力量训练")

        # 添加一般建议
        if len(suggestions) == 0:
            suggestions.append("🎉 **表现优秀**！继续保持练习，注意细节的完美呈现")
        else:
            suggestions.append("📝 **练习建议**：每天针对性练习20-30分钟，重点改进上述问题，录制视频自我检查")

        return suggestions

    def _identify_key_issues(self, comparison: Dict) -> List[Dict]:
        """识别关键问题点"""
        issues = []
        avg_scores = comparison['average_scores']

        # 找出分数最低的3个项目
        sorted_categories = sorted(
            avg_scores.items(),
            key=lambda x: x[1]
        )[:3]

        category_names = {
            'timing': '节奏时机',
            'position': '位置方向',
            'movement': '动作幅度',
            'posture': '姿势标准'
        }

        for category, score in sorted_categories:
            if score < 80:
                issue = {
                    'category': category_names.get(category, category),
                    'score': round(score, 1),
                    'severity': '严重' if score < 60 else '中等' if score < 70 else '轻微',
                    'description': self._get_issue_description(category, score)
                }
                issues.append(issue)

        return issues

    def _get_issue_description(self, category: str, score: float) -> str:
        """获取问题描述"""
        descriptions = {
            'timing': {
                'high': '节奏感很好，与音乐完美同步',
                'medium': '节奏基本准确，偶尔有延迟',
                'low': '节奏感需要加强，经常抢拍或拖拍'
            },
            'position': {
                'high': '位置非常准确，与标准完全一致',
                'medium': '位置基本正确，有轻微偏差',
                'low': '位置偏差较大，需要对照标准纠正'
            },
            'movement': {
                'high': '动作幅度恰到好处，非常标准',
                'medium': '动作幅度基本到位，可更舒展',
                'low': '动作幅度不足或过度，需要调整'
            },
            'posture': {
                'high': '姿势非常标准，身体线条优美',
                'medium': '姿势基本正确，可更挺拔',
                'low': '姿势需要纠正，注意身体对齐'
            }
        }

        if category not in descriptions:
            return '需要改进'

        if score >= 85:
            level = 'high'
        elif score >= 70:
            level = 'medium'
        else:
            level = 'low'

        return descriptions[category][level]

    def _identify_frame_issues(self, frame_comparison: Dict) -> List[str]:
        """识别单帧问题"""
        issues = []

        if frame_comparison['timing_score'] < 60:
            issues.append('节奏不准')
        if frame_comparison['position_score'] < 60:
            issues.append('位置偏差')
        if frame_comparison['movement_score'] < 60:
            issues.append('动作变形')
        if frame_comparison['posture_score'] < 60:
            issues.append('姿势不正')

        return issues if issues else ['表现良好']

    def _create_score_breakdown(self, final_score: Dict) -> Dict:
        """创建分数分解说明"""
        breakdown = {}

        for category, scores in final_score['categories'].items():
            category_name = {
                'timing': '节奏与时机',
                'position': '位置与方向',
                'movement': '动作幅度',
                'posture': '姿势标准度',
                'expression': '表情表现'
            }.get(category, category)

            breakdown[category_name] = {
                '得分': f"{scores['raw_score']:.1f}",
                '权重': f"{scores['weight'] * 100:.0f}%",
                '加权分': f"{scores['weighted_score']:.2f}",
                '评价': self._get_category_evaluation(category, scores['raw_score'])
            }

        return breakdown

    def _get_category_evaluation(self, category: str, score: float) -> str:
        """获取维度评价"""
        if score >= 90:
            return '优秀'
        elif score >= 80:
            return '良好'
        elif score >= 70:
            return '合格'
        elif score >= 60:
            return '需改进'
        else:
            return '需重点训练'

    def _get_grade(self, score: float) -> str:
        """获取等级"""
        if score >= 90:
            return "A+ (卓越)"
        elif score >= 85:
            return "A (优秀)"
        elif score >= 80:
            return "A- (很好)"
        elif score >= 75:
            return "B+ (良好)"
        elif score >= 70:
            return "B (中等)"
        elif score >= 60:
            return "C (合格)"
        else:
            return "D (需改进)"