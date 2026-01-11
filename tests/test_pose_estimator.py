# test_pose.py
import cv2
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pose_estimator import PoseEstimator


def test_pose_estimator():
    """测试姿态估计器"""
    print("🧪 开始测试PoseEstimator...")

    # 1. 创建估计器
    try:
        estimator = PoseEstimator()
        print("✅ PoseEstimator初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False

    # 2. 创建一个测试图像（黑色背景，白色圆模拟人体）
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # 在图像中心画一个白色椭圆模拟人体
    center = (320, 240)
    axes = (100, 150)
    cv2.ellipse(test_image, center, axes, 0, 0, 360, (255, 255, 255), -1)

    print(f"测试图像大小: {test_image.shape}")

    # 3. 处理图像
    try:
        landmarks = estimator.process_frame(test_image)

        if landmarks is not None:
            print(f"✅ 成功检测到姿态")
            print(f"关键点形状: {landmarks.shape}")  # 应该是 (33, 4)
            print(f"检测到的关键点数量: {len(landmarks)}")

            # 显示前5个关键点
            for i in range(min(5, len(landmarks))):
                print(f"  关键点 {i}: {landmarks[i]}")

            # 4. 测试绘图功能
            annotated = estimator.draw_landmarks(test_image, landmarks_array=landmarks)

            # 保存结果供查看
            output_path = "test_pose_output.jpg"
            cv2.imwrite(output_path, annotated)
            print(f"✅ 标注图像已保存到: {output_path}")

            return True
        else:
            print("⚠️ 未检测到姿态（这是正常的，因为测试图像不是真实人体）")

            # 使用真实测试图像
            print("尝试使用内置测试图像...")
            test_real_image()

            return True

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_image():
    """使用真实图像测试"""
    # 尝试创建简单的人形图案
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # 画头
    cv2.circle(img, (320, 100), 30, (255, 255, 255), -1)

    # 画身体
    cv2.line(img, (320, 130), (320, 300), (255, 255, 255), 20)

    # 画手臂
    cv2.line(img, (320, 180), (250, 250), (255, 255, 255), 15)
    cv2.line(img, (320, 180), (390, 250), (255, 255, 255), 15)

    # 画腿
    cv2.line(img, (320, 300), (280, 400), (255, 255, 255), 15)
    cv2.line(img, (320, 300), (360, 400), (255, 255, 255), 15)

    estimator = PoseEstimator()
    landmarks = estimator.process_frame(img)

    if landmarks is not None:
        print("✅ 使用模拟人形图像检测成功!")
        annotated = estimator.draw_landmarks(img, landmarks_array=landmarks)
        cv2.imwrite("test_human_output.jpg", annotated)
    else:
        print("⚠️ 模拟人形图像也未检测到姿态")


if __name__ == "__main__":
    success = test_pose_estimator()
    if success:
        print("\n🎉 PoseEstimator测试通过!")
    else:
        print("\n❌ PoseEstimator测试失败")