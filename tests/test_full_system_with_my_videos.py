# test_full_system_with_my_videos.py
import os
import sys
import cv2
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.pose_estimator import PoseEstimator
from src.video_processor import VideoProcessor
from src.scoring_algorithm import DanceScorer


def test_with_my_videos(std_video_path, user_video_path):
    """用你的视频进行完整系统测试"""
    print("=" * 70)
    print("🎬 个人舞蹈视频完整系统测试")
    print("=" * 70)

    print(f"📹 标准视频: {os.path.basename(std_video_path)}")
    print(f"👤 个人视频: {os.path.basename(user_video_path)}")

    # 1. 检查视频文件
    print("\n1️⃣ 检查视频文件...")
    for video_path in [std_video_path, user_video_path]:
        if not os.path.exists(video_path):
            print(f"❌ 文件不存在: {video_path}")
            return False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开: {video_path}")
            return False
        cap.release()

    print("✅ 视频文件检查通过")

    # 2. 初始化处理器
    print("\n2️⃣ 初始化处理模块...")
    try:
        pose_estimator = PoseEstimator()
        video_processor = VideoProcessor(pose_estimator)
        scorer = DanceScorer()
        print("✅ 模块初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False

    # 3. 处理标准视频
    print("\n3️⃣ 处理标准视频...")
    try:
        std_result = video_processor.process_video(std_video_path, max_frames=200)
        if std_result is None or 'keypoints' not in std_result:
            print("❌ 标准视频处理失败")
            return False

        std_keypoints = std_result['keypoints'][:, :, :3]  # 只取xyz
        print(f"✅ 标准视频处理成功")
        print(f"   提取帧数: {len(std_keypoints)}")
        print(f"   数据形状: {std_keypoints.shape}")
    except Exception as e:
        print(f"❌ 标准视频处理出错: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 处理个人视频
    print("\n4️⃣ 处理个人视频...")
    try:
        user_result = video_processor.process_video(user_video_path, max_frames=200)
        if user_result is None or 'keypoints' not in user_result:
            print("❌ 个人视频处理失败")
            return False

        user_keypoints = user_result['keypoints'][:, :, :3]  # 只取xyz
        print(f"✅ 个人视频处理成功")
        print(f"   提取帧数: {len(user_keypoints)}")
        print(f"   数据形状: {user_keypoints.shape}")
    except Exception as e:
        print(f"❌ 个人视频处理出错: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. 进行评分
    print("\n5️⃣ 进行动作评分...")
    try:
        results = scorer.compare_poses(std_keypoints, user_keypoints)

        print("✅ 评分完成！")
        print("\n📊 评分结果：")
        print("-" * 40)

        if 'overall_score' in results:
            print(f"🎯 总体评分: {results['overall_score']:.1f}/100")

        if 'pose_similarity' in results:
            print(f"📈 动作相似度: {results['pose_similarity']:.1f}%")

        if 'rhythm_similarity' in results:
            print(f"🎵 节奏准确度: {results['rhythm_similarity']:.1f}%")

        if 'body_part_scores' in results:
            print("\n🦵 身体部位评分：")
            for part, score in results['body_part_scores'].items():
                print(f"  {part}: {score:.1f}分")

        return True

    except Exception as e:
        print(f"❌ 评分过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_video_pair():
    """自动查找标准和个人视频对"""
    print("🔍 自动查找视频对...")

    # 查找所有视频
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

    std_videos = []
    user_videos = []

    # 搜索标准视频
    std_locations = ["data/standard_videos", "data/samples"]
    for location in std_locations:
        if os.path.exists(location):
            for file in os.listdir(location):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    full_path = os.path.join(location, file)
                    std_videos.append(full_path)

    # 搜索个人视频
    user_locations = ["data/user_videos", "data/samples"]
    for location in user_locations:
        if os.path.exists(location):
            for file in os.listdir(location):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    full_path = os.path.join(location, file)
                    user_videos.append(full_path)

    return std_videos, user_videos


def main():
    """主函数"""
    print("🚀 个人舞蹈视频完整系统测试")
    print("=" * 60)

    # 查找视频
    std_videos, user_videos = find_video_pair()

    if not std_videos:
        print("❌ 未找到标准示范视频")
        print("💡 请将标准舞蹈视频放入：data/standard_videos/")
        return

    if not user_videos:
        print("❌ 未找到个人舞蹈视频")
        print("💡 请将你的舞蹈视频放入：data/user_videos/")
        return

    print(f"📁 找到 {len(std_videos)} 个标准视频")
    print(f"📁 找到 {len(user_videos)} 个个人视频")

    # 让用户选择
    print("\n选择标准视频：")
    for i, video in enumerate(std_videos, 1):
        print(f"  {i}. {os.path.basename(video)}")

    std_choice = int(input("请输入编号: ")) - 1
    if not (0 <= std_choice < len(std_videos)):
        print("❌ 无效选择")
        return

    print("\n选择个人视频：")
    for i, video in enumerate(user_videos, 1):
        print(f"  {i}. {os.path.basename(video)}")

    user_choice = int(input("请输入编号: ")) - 1
    if not (0 <= user_choice < len(user_videos)):
        print("❌ 无效选择")
        return

    # 进行测试
    std_video = std_videos[std_choice]
    user_video = user_videos[user_choice]

    print(f"\n🎬 测试配对：")
    print(f"  标准视频: {os.path.basename(std_video)}")
    print(f"  个人视频: {os.path.basename(user_video)}")

    success = test_with_my_videos(std_video, user_video)

    if success:
        print("\n" + "=" * 70)
        print("🎉 完整系统测试成功！")
        print("💡 下一步：运行网页应用进行完整体验")
        print("   streamlit run streamlit_app.py")
        print("=" * 70)
    else:
        print("\n❌ 测试失败，请检查错误信息")


if __name__ == "__main__":
    main()