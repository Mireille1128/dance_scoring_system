# 💃 基于MediaPipe的舞蹈动作匹配评分系统

## 项目简介
本项目是一个基于MediaPipe和Streamlit的智能舞蹈动作评分系统，能够自动分析用户舞蹈视频与标准舞蹈视频的相似度，并提供详细的评分报告和改进建议。

## 功能特性
- 🎯 **姿态估计**：使用MediaPipe提取33个人体关键点
- 📊 **动作对比**：采用DTW算法进行时间序列匹配
- 🎨 **可视化分析**：实时显示姿态检测结果和对比分析
- 📈 **详细报告**：生成全面的评分报告和改进建议
- 🌐 **Web界面**：基于Streamlit的现代化Web应用

## 快速开始

### 环境要求
- Python 3.8+
- 支持CUDA的GPU（可选，推荐）

### 安装步骤
1. 克隆项目
bash
git clone <项目地址>
cd dance_scoring_system
2. 安装依赖
bash
pip install -r requirements.txt
3. 运行应用
bash
streamlit run streamlit_app.py
4. 打开浏览器访问 `http://localhost:8501`

## 项目结构
（此处省略，与上面结构一致）

## 技术栈
- **前端**：Streamlit
- **姿态估计**：MediaPipe Pose
- **视频处理**：OpenCV
- **算法核心**：NumPy, SciPy, DTW算法
- **数据可视化**：Plotly, Matplotlib

## 使用说明
1. 上传标准舞蹈视频和用户舞蹈视频
2. 点击"开始分析"按钮
3. 查看实时处理进度
4. 浏览详细的评分报告和可视化分析

## 毕业设计相关信息
- **选题**：基于MediaPipe的舞蹈动作匹配评分系统设计与实现
- **技术**：Python, MediaPipe, Streamlit, OpenCV
- **周期**：3个月
- **难度**：中等