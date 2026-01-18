# streamlit_app.py
"""
舞蹈标准对比学习系统
用户上传自己的舞蹈视频与标准视频对比，获得评分和改进建议
"""

import streamlit as st
import sys
import os
import tempfile
from pathlib import Path
import cv2
import time
import json
import pandas as pd

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="舞蹈标准对比学习系统",
    page_icon="💃",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Mireille1128/dance-standard-comparison.wiki.git',
        'Report a bug': "https://github.com/Mireille1128/dance-standard-comparison/issues/1#issue-3800758238",
        'About': """
        # 舞蹈标准对比学习系统 v1.0

        ## 系统功能：
        - 上传标准舞蹈教学视频
        - 上传个人练习视频
        - 智能对比分析动作标准度
        - 提供详细的改进建议
        - 识别关键问题点

        ## 评分维度：
        - 节奏与时机 (30%)
        - 位置与方向 (25%)
        - 动作幅度 (20%)
        - 姿势标准度 (15%)
        - 表情表现 (10%)
        """
    }
)

# ==================== 路径设置 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")

if src_dir not in sys.path:
    sys.path.append(src_dir)

# ==================== 导入本地模块 ====================
try:
    from src.scoring_algorithm import StandardDanceScorer

    st.sidebar.success("✅ 标准对比评分算法加载成功")
except ImportError as e:
    st.sidebar.error(f"⚠️ 无法导入评分模块: {e}")
    st.error("请确保 scoring_algorithm.py 文件存在且包含 StandardDanceScorer 类")
    st.stop()


# ==================== 初始化 ====================
@st.cache_resource
def init_scorer():
    """初始化评分器"""
    return StandardDanceScorer()


scorer = init_scorer()

# ==================== 会话状态 ====================
if 'standard_loaded' not in st.session_state:
    st.session_state.standard_loaded = False
if 'standard_video_path' not in st.session_state:
    st.session_state.standard_video_path = None
if 'evaluation_result' not in st.session_state:
    st.session_state.evaluation_result = None


# ==================== 辅助函数 ====================
def save_uploaded_file(uploaded_file, temp_dir):
    """保存上传的文件"""
    try:
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            dir=temp_dir,
            suffix=Path(uploaded_file.name).suffix
        )
        temp_file.write(uploaded_file.read())
        temp_file.close()
        return temp_file.name
    except Exception as e:
        st.error(f"文件保存失败: {str(e)}")
        return None


def extract_video_thumbnail(video_path):
    """提取视频缩略图"""
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (300, 200))
                cap.release()
                return frame_resized
        cap.release()
    except:
        pass
    return None


def get_video_info(video_path):
    """获取视频信息"""
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

            return {
                "分辨率": f"{width}×{height}",
                "帧率": f"{fps:.1f} FPS",
                "时长": f"{duration:.1f}秒",
                "总帧数": frame_count
            }
    except:
        pass
    return {}


# ==================== 主界面 ====================
st.title("🏆 舞蹈标准对比学习系统")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0;'>
<h4 style='color: #1f77b4; margin-top: 0;'>系统介绍</h4>
<p>本系统通过对比您的舞蹈视频与标准教学视频，智能分析动作标准度，提供个性化改进建议，帮助您更快提升舞蹈水平。</p>
<p><strong>使用步骤：</strong> 1. 上传标准视频 → 2. 上传个人视频 → 3. 开始对比分析</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# 创建临时目录
temp_dir = tempfile.mkdtemp()

# ==================== 步骤1：标准视频上传 ====================
st.header("📚 步骤1：上传标准舞蹈教学视频")

col_standard1, col_standard2 = st.columns([2, 1])

with col_standard1:
    standard_file = st.file_uploader(
        "选择标准舞蹈教学视频",
        type=['mp4', 'avi', 'mov', 'mkv'],
        key="standard_video",
        help="请选择专业的舞蹈教学视频作为标准参考"
    )

    if standard_file:
        with st.spinner("正在处理标准视频..."):
            standard_path = save_uploaded_file(standard_file, temp_dir)
            if standard_path:
                # 加载标准视频
                result = scorer.load_standard_video(standard_path)

                if result['success']:
                    st.session_state.standard_loaded = True
                    st.session_state.standard_video_path = standard_path
                    st.success("✅ 标准视频加载成功！")

                    # 显示视频信息
                    video_info = result['video_info']
                    st.info(f"""
                    **视频信息：**
                    - 时长: {video_info['duration']:.1f}秒
                    - 帧率: {video_info['fps']:.1f} FPS
                    - 分析帧数: {video_info['analyzed_frames']}帧
                    """)
                else:
                    st.error(f"❌ 标准视频加载失败: {result.get('error', '未知错误')}")

with col_standard2:
    if st.session_state.standard_loaded:
        # 显示标准视频缩略图
        thumbnail = extract_video_thumbnail(st.session_state.standard_video_path)
        if thumbnail is not None:
            st.image(thumbnail, caption="标准视频预览")
        else:
            st.info("无法生成视频预览")

        st.metric("状态", "✅ 已加载", "标准视频就绪")

# ==================== 步骤2：个人视频上传 ====================
st.markdown("---")
st.header("🎬 步骤2：上传个人舞蹈练习视频")

if not st.session_state.standard_loaded:
    st.warning("⚠️ 请先上传标准视频")
    st.stop()

col_personal1, col_personal2 = st.columns([2, 1])

with col_personal1:
    personal_file = st.file_uploader(
        "选择您的舞蹈练习视频",
        type=['mp4', 'avi', 'mov', 'mkv'],
        key="personal_video",
        help="请上传您要对比分析的舞蹈练习视频"
    )

    personal_video_path = None

    if personal_file:
        with st.spinner("正在处理个人视频..."):
            personal_path = save_uploaded_file(personal_file, temp_dir)
            if personal_path:
                personal_video_path = personal_path

                # 显示视频信息
                info = get_video_info(personal_path)
                if info:
                    st.info(f"""
                    **您的视频信息：**
                    - 分辨率: {info['分辨率']}
                    - 时长: {info['时长']}
                    - 帧率: {info['帧率']}
                    """)

with col_personal2:
    if personal_video_path:
        # 显示个人视频缩略图
        thumbnail = extract_video_thumbnail(personal_video_path)
        if thumbnail is not None:
            st.image(thumbnail, caption="您的视频预览")
        else:
            st.info("无法生成视频预览")

# ==================== 步骤3：开始对比分析 ====================
if personal_video_path and st.session_state.standard_loaded:
    st.markdown("---")
    st.header("🔍 步骤3：开始对比分析")

    col_analyze1, col_analyze2, col_analyze3 = st.columns([1, 2, 1])

    with col_analyze2:
        analyze_button = st.button(
            "🎯 开始智能对比分析",
            type="primary",
            use_container_width=True
        )

    if analyze_button:
        with st.spinner("正在分析您的舞蹈动作，请稍候..."):
            # 显示进度
            progress_bar = st.progress(0)

            # 模拟分析过程
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            # 执行对比分析
            evaluation_result = scorer.evaluate_student_video(personal_video_path)

            progress_bar.empty()

            if evaluation_result['success']:
                st.session_state.evaluation_result = evaluation_result
                st.success("✅ 分析完成！")
                st.balloons()
            else:
                st.error(f"❌ 分析失败: {evaluation_result.get('error', '未知错误')}")

# ==================== 步骤4：显示分析结果 ====================
if st.session_state.evaluation_result:
    result = st.session_state.evaluation_result

    st.markdown("---")
    st.header("📊 分析结果报告")

    # 总体分数卡片
    col_overall1, col_overall2, col_overall3 = st.columns(3)

    with col_overall1:
        st.metric(
            label="🏆 综合得分",
            value=f"{result['overall_score']:.1f}",
            delta=result['grade']
        )

    with col_overall2:
        st.metric(
            label="📈 表现等级",
            value=result['grade'].split(' ')[0],
            delta="详细分析见下方"
        )

    with col_overall3:
        duration = result['student_info']['duration']
        st.metric(
            label="⏱️ 视频时长",
            value=f"{duration:.1f}秒",
            delta="已分析"
        )

    # 分数分解雷达图
    st.subheader("📈 各维度评分分析")

    # 创建分数数据
    category_scores = result['category_scores']

    categories = ['节奏与时机', '位置与方向', '动作幅度', '姿势标准度']
    scores = []
    weights = []

    for cat_name, cat_data in result['score_breakdown'].items():
        if cat_name in categories:
            scores.append(float(cat_data['得分']))
            weights.append(float(cat_data['权重'].replace('%', '')))

    # 显示评分表
    score_data = []
    for cat_name, cat_data in result['score_breakdown'].items():
        score_data.append({
            '评分维度': cat_name,
            '得分': cat_data['得分'],
            '权重': cat_data['权重'],
            '加权分': cat_data['加权分'],
            '评价': cat_data['评价']
        })

    st.dataframe(pd.DataFrame(score_data), use_container_width=True)

    # 关键问题点
    st.subheader("🔴 关键问题识别")

    if result['key_issues']:
        col_issues1, col_issues2 = st.columns(2)

        for i, issue in enumerate(result['key_issues']):
            with col_issues1 if i % 2 == 0 else col_issues2:
                severity_color = {
                    '严重': 'red',
                    '中等': 'orange',
                    '轻微': 'green'
                }.get(issue['severity'], 'gray')

                st.markdown(f"""
                <div style='border-left: 4px solid {severity_color}; padding: 10px; margin: 10px 0; background-color: #f9f9f9;'>
                <h4 style='margin-top: 0;'>{issue['category']} <span style='color: {severity_color};'>({issue['severity']})</span></h4>
                <p><strong>得分:</strong> {issue['score']}/100</p>
                <p><strong>问题:</strong> {issue['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("🎉 未发现明显问题点，表现优秀！")

    # 改进建议
    st.subheader("💡 个性化改进建议")

    suggestions = result['improvement_suggestions']

    for i, suggestion in enumerate(suggestions):
        st.markdown(f"""
        <div style='background-color: #e8f4fd; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #1f77b4;'>
        <h5 style='margin-top: 0; color: #1f77b4;'>建议 {i + 1}</h5>
        <p style='margin-bottom: 0;'>{suggestion}</p>
        </div>
        """, unsafe_allow_html=True)

    # 练习计划建议
    st.subheader("📅 推荐练习计划")

    practice_plan = {
        "周一": "节奏感训练 - 使用节拍器练习基础步伐",
        "周二": "位置准确性练习 - 对照镜子修正动作位置",
        "周三": "动作幅度练习 - 重点练习伸展和收缩",
        "周四": "姿势纠正训练 - 加强核心力量练习",
        "周五": "完整舞蹈练习 - 结合所有改进点",
        "周六": "复习巩固 - 重复练习薄弱环节",
        "周日": "休息恢复 - 观看标准视频学习"
    }

    practice_df = pd.DataFrame(list(practice_plan.items()), columns=['日期', '练习内容'])
    st.dataframe(practice_df, use_container_width=True)

    # 最差帧分析
    comparison = result['detailed_comparison']
    if comparison.get('worst_frames'):
        st.subheader("⏱️ 问题时间点分析")

        worst_frames = comparison['worst_frames']

        for frame in worst_frames[:3]:  # 显示最差的3帧
            col_time, col_score, col_issues = st.columns([1, 1, 2])

            with col_time:
                st.metric("时间点", f"{frame['timestamp']:.1f}秒")

            with col_score:
                st.metric("得分", f"{frame['score']:.1f}")

            with col_issues:
                issues_text = "、".join(frame['issues'])
                st.info(f"主要问题: {issues_text}")

    # 导出报告
    st.markdown("---")
    st.subheader("💾 导出分析报告")

    if st.button("📥 生成完整分析报告"):
        # 准备报告数据
        report_data = {
            'analysis_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'student_name': '舞蹈学员',
            'standard_video': st.session_state.standard_video_path,
            'student_video': personal_video_path,
            'overall_score': result['overall_score'],
            'grade': result['grade'],
            'category_scores': result['category_scores'],
            'key_issues': result['key_issues'],
            'improvement_suggestions': result['improvement_suggestions'],
            'practice_plan': practice_plan
        }

        # 生成JSON报告
        json_report = json.dumps(report_data, indent=2, ensure_ascii=False)

        # 下载按钮
        st.download_button(
            label="下载JSON格式报告",
            data=json_report,
            file_name=f"舞蹈分析报告_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        # 生成文本总结
        text_report = f"""舞蹈学习分析报告
================================
分析时间: {time.strftime("%Y-%m-%d %H:%M:%S")}
综合得分: {result['overall_score']:.1f} ({result['grade']})

各维度评分:
"""
        for cat_name, cat_data in result['score_breakdown'].items():
            text_report += f"- {cat_name}: {cat_data['得分']}分 ({cat_data['评价']})\n"

        text_report += "\n改进建议:\n"
        for i, suggestion in enumerate(result['improvement_suggestions'], 1):
            text_report += f"{i}. {suggestion}\n"

        st.download_button(
            label="下载文本格式报告",
            data=text_report,
            file_name=f"舞蹈分析报告_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("⚙️ 系统设置")

    st.subheader("📊 评分权重设置")

    # 权重调整滑块
    timing_weight = st.slider("节奏与时机", 0.1, 0.5, 0.30, 0.05)
    position_weight = st.slider("位置与方向", 0.1, 0.4, 0.25, 0.05)
    movement_weight = st.slider("动作幅度", 0.1, 0.3, 0.20, 0.05)
    posture_weight = st.slider("姿势标准度", 0.1, 0.3, 0.15, 0.05)
    expression_weight = st.slider("表情表现", 0.05, 0.2, 0.10, 0.05)

    # 更新权重
    total_weight = timing_weight + position_weight + movement_weight + posture_weight + expression_weight
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"权重总和应为100%，当前为{total_weight * 100:.0f}%")
    else:
        scorer.weights = {
            'timing': timing_weight,
            'position': position_weight,
            'movement': movement_weight,
            'posture': posture_weight,
            'expression': expression_weight
        }

    st.markdown("---")

    st.subheader("📚 舞蹈类型预设")

    dance_type = st.selectbox(
        "选择舞蹈类型",
        ["街舞/Hip-hop", "爵士舞", "芭蕾舞", "现代舞", "民族舞", "自定义"],
        help="不同舞蹈类型有不同的评分标准"
    )

    if dance_type != "自定义":
        preset_weights = {
            "街舞/Hip-hop": {'timing': 0.35, 'position': 0.20, 'movement': 0.25, 'posture': 0.10, 'expression': 0.10},
            "爵士舞": {'timing': 0.30, 'position': 0.25, 'movement': 0.20, 'posture': 0.15, 'expression': 0.10},
            "芭蕾舞": {'timing': 0.25, 'position': 0.30, 'movement': 0.20, 'posture': 0.20, 'expression': 0.05},
            "现代舞": {'timing': 0.20, 'position': 0.25, 'movement': 0.25, 'posture': 0.20, 'expression': 0.10},
            "民族舞": {'timing': 0.30, 'position': 0.25, 'movement': 0.20, 'posture': 0.15, 'expression': 0.10}
        }

        if dance_type in preset_weights:
            scorer.weights = preset_weights[dance_type]
            st.success(f"已应用{dance_type}权重预设")

    st.markdown("---")

    st.subheader("ℹ️ 使用说明")
    st.markdown("""
    1. **上传标准视频**：专业的舞蹈教学视频
    2. **上传个人视频**：您的练习视频
    3. **开始分析**：系统自动对比并评分
    4. **查看报告**：获取详细分析和改进建议

    **支持格式**：MP4, AVI, MOV, MKV
    **建议时长**：30秒 - 5分钟
    """)

# ==================== 页脚 ====================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>舞蹈标准对比学习系统 v3.0</strong></p>
    <p>基于姿势识别的智能舞蹈分析 | 助力舞蹈学习进步</p>
    <p>💡 提示：保持视频光线充足，背景简洁，动作完整可见</p>
    </div>
    """,
    unsafe_allow_html=True
)