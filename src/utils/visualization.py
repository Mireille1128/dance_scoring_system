# src/utils/visualization.py
import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


class ResultVisualizer:
    """结果可视化类"""

    def __init__(self):
        self.colors = {
            'standard': '#1f77b4',
            'user': '#ff7f0e',
            'difference': '#d62728'
        }

    def create_similarity_chart(self, similarity_scores):
        """创建相似度雷达图"""
        categories = ['整体相似度', '姿态准确度', '节奏同步', '动作流畅度']
        scores = [
            similarity_scores.get('overall', 0),
            similarity_scores.get('pose', 0),
            similarity_scores.get('rhythm', 0),
            similarity_scores.get('smoothness', 0)
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],  # 闭合图形
            theta=categories + [categories[0]],
            fill='toself',
            name='用户表现',
            line=dict(color=self.colors['user'], width=2)
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title='舞蹈动作相似度分析'
        )

        return fig

    def create_movement_comparison(self, std_trajectory, user_trajectory):
        """创建动作轨迹对比图"""
        fig = go.Figure()

        # 标准动作轨迹
        fig.add_trace(go.Scatter(
            x=list(range(len(std_trajectory))),
            y=std_trajectory,
            mode='lines',
            name='标准动作',
            line=dict(color=self.colors['standard'], width=3)
        ))

        # 用户动作轨迹
        fig.add_trace(go.Scatter(
            x=list(range(len(user_trajectory))),
            y=user_trajectory,
            mode='lines',
            name='您的动作',
            line=dict(color=self.colors['user'], width=3)
        ))

        fig.update_layout(
            title='动作轨迹对比',
            xaxis_title='时间帧',
            yaxis_title='动作幅度',
            hovermode='x unified'
        )

        return fig

    def create_joint_analysis_chart(self, joint_scores):
        """创建关节分析柱状图"""
        joints = list(joint_scores.keys())
        scores = list(joint_scores.values())

        fig = go.Figure(data=[
            go.Bar(
                x=joints,
                y=scores,
                marker_color=[
                    'green' if score >= 80 else
                    'orange' if score >= 60 else 'red'
                    for score in scores
                ]
            )
        ])

        fig.update_layout(
            title='各关节动作准确度',
            xaxis_title='关节部位',
            yaxis_title='准确度 (%)',
            yaxis=dict(range=[0, 100])
        )

        return fig

    def draw_pose_comparison(self, std_frame, user_frame, std_landmarks, user_landmarks):
        """绘制姿态对比图像"""
        # 调整图像大小一致
        h1, w1 = std_frame.shape[:2]
        h2, w2 = user_frame.shape[:2]
        target_height = max(h1, h2)
        target_width = max(w1, w2)

        # 调整大小
        std_frame_resized = cv2.resize(std_frame, (target_width, target_height))
        user_frame_resized = cv2.resize(user_frame, (target_width, target_height))

        # 绘制关键点
        std_annotated = self._draw_landmarks_on_image(std_frame_resized, std_landmarks, (0, 255, 0))
        user_annotated = self._draw_landmarks_on_image(user_frame_resized, user_landmarks, (255, 0, 0))

        # 水平拼接
        comparison = np.hstack([std_annotated, user_annotated])

        # 添加标签
        cv2.putText(comparison, "标准动作", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "您的动作", (target_width + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return comparison

    def _draw_landmarks_on_image(self, image, landmarks, color):
        """在图像上绘制关键点"""
        annotated = image.copy()
        h, w = image.shape[:2]

        for landmark in landmarks:
            x = int(landmark[0] * w)
            y = int(landmark[1] * h)
            cv2.circle(annotated, (x, y), 5, color, -1)

        return annotated