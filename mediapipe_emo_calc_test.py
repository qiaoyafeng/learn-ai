import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
import matplotlib

matplotlib.use("TkAgg")

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 情感映射配置
EMOTION_MAPPING = {
    "攻击性": ["mouthPucker", "jawForward", "browDownLeft", "browDownRight"],
    "压力": ["eyeSquintLeft", "eyeSquintRight", "mouthPressLeft", "mouthPressRight"],
    "不安": ["mouthStretchLeft", "mouthStretchRight", "browInnerUp"],
    "焦虑": ["cheekSquintLeft", "cheekSquintRight", "jawOpen"],
    "怀疑": ["mouthDimpleLeft", "mouthDimpleRight", "noseSneerLeft"],
    "平衡": ["mouthSmileLeft", "mouthSmileRight", "eyeBlinkLeft"],
    "能量": ["eyeWideLeft", "eyeWideRight", "jawOpen"],
    "活力": ["mouthSmileLeft", "mouthSmileRight", "cheekPuff"],
    "自控能力": ["mouthClose", "lipsToward", "browInnerUp"],
    "抑制": ["mouthFrownLeft", "mouthFrownRight", "browDownLeft"],
    "神经质": ["eyeBlinkLeft", "eyeBlinkRight", "mouthShrugLower"],
    "抑郁": ["mouthFrownLeft", "mouthFrownRight", "browInnerUp"],
    "幸福": ["mouthSmileLeft", "mouthSmileRight", "cheekSquintLeft"]
}

REVERSE_EMOTIONS = ["自控能力"]  # 需要反向计算的维度


def extract_blendshapes_from_video(video_path, output_csv='blendshapes.csv'):
    """
    从视频中提取面部Blendshapes数据并保存为CSV文件
    使用MediaPipe FaceLandmarker的VIDEO模式
    """
    # 初始化MediaPipe解决方案
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection

    # 创建FaceLandmarker选项
    base_options = mp.tasks.BaseOptions(model_asset_path='face_landmarker.task')
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return None

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息: {frame_count}帧, {fps:.1f} FPS, {width}x{height}")

    # 准备存储结果
    all_blendshapes = []
    frame_numbers = []
    timestamps = []
    blendshapes_names = None

    # 创建进度条
    pbar = tqdm(total=frame_count, desc="处理视频帧")

    # 使用FaceLandmarker处理视频
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # 转换图像为MediaPipe图像格式
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # 计算时间戳（微秒）
            timestamp_ms = int(1000 * frame_idx / fps)

            # 检测面部地标
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # 初始化当前帧的blendshapes数据
            frame_blendshapes = {}

            if detection_result.face_blendshapes:
                # 只处理检测到的第一张脸
                face_blendshapes = detection_result.face_blendshapes[0]

                # 获取blendshapes名称（如果尚未获取）
                if blendshapes_names is None:
                    blendshapes_names = [c.category_name for c in face_blendshapes]

                # 提取blendshapes值
                for blend in face_blendshapes:
                    frame_blendshapes[blend.category_name] = blend.score

            # 保存结果
            if frame_blendshapes:
                all_blendshapes.append(frame_blendshapes)
            else:
                # 如果没有检测到面部，填充零值
                if blendshapes_names:
                    all_blendshapes.append({name: 0.0 for name in blendshapes_names})

            frame_numbers.append(frame_idx)
            timestamps.append(frame_idx / fps)

            frame_idx += 1
            pbar.update(1)

    pbar.close()
    cap.release()

    # 创建DataFrame
    if blendshapes_names:
        blendshapes_df = pd.DataFrame(all_blendshapes, columns=blendshapes_names)
        blendshapes_df.insert(0, 'frame', frame_numbers)
        blendshapes_df.insert(1, 'timestamp', timestamps)

        # 保存到CSV
        blendshapes_df.to_csv(output_csv, index=False)
        print(f"Blendshapes数据已保存到: {output_csv}")
        return blendshapes_df
    else:
        print("未检测到任何面部数据")
        return None


def calculate_emotion_scores(blendshapes_data):
    """
    计算每帧的情感维度得分
    :param blendshapes_data: DataFrame格式的Blendshapes数据
    :return: 情感维度DataFrame (frame x emotions)
    """
    emotion_scores = pd.DataFrame()

    for emotion, features in EMOTION_MAPPING.items():
        # 提取相关特征列
        valid_features = [f for f in features if f in blendshapes_data.columns]

        if not valid_features:
            print(f"警告: {emotion} 无有效特征")
            continue

        # 计算特征均值
        mean_values = blendshapes_data[valid_features].mean(axis=1)

        # 反向特征处理
        if emotion in REVERSE_EMOTIONS:
            mean_values = 1 - mean_values

        # 归一化到0-100
        emotion_scores[emotion] = mean_values * 100

    return emotion_scores


def generate_visualizations(emotion_scores):
    """
    生成三种可视化图表
    :param emotion_scores: 情感维度DataFrame
    """
    # 1. 雷达图 (均值)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)

    stats = emotion_scores.mean()
    categories = list(stats.index)
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    stats = np.concatenate((stats.values, [stats.values[0]]))
    angles += angles[:1]

    ax.plot(angles, stats, 'o-', linewidth=2, color='blue')
    ax.fill(angles, stats, alpha=0.25, color='skyblue')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=7)
    plt.ylim(0, 100)
    plt.title("情感维度雷达图", size=20, pad=20)
    plt.savefig('emotion_radar.png', dpi=300, bbox_inches='tight')

    # 2. 箱线图 (分布)
    plt.figure(figsize=(14, 8))
    boxplot = emotion_scores.plot(kind='box', vert=True, patch_artist=True)
    plt.title("情感维度分布箱线图")
    plt.ylabel("得分 (0-100)")
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 设置箱线图颜色
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink',
              'lightseagreen', 'lightsteelblue', 'lightskyblue', 'lightcyan',
              'lightyellow', 'lightgray', 'lightgoldenrodyellow', 'lightcoral']

    for patch, color in zip(boxplot.findobj(match=PathPatch), colors):
        patch.set_facecolor(color)

    plt.tight_layout()
    plt.savefig('emotion_boxplot.png', dpi=300)

    # 3. 时间变化曲线
    plt.figure(figsize=(15, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(emotion_scores.columns)))

    for i, emotion in enumerate(emotion_scores.columns, 1):
        plt.subplot(4, 4, i)
        plt.plot(emotion_scores[emotion], lw=1.5, color=colors[i - 1])
        plt.title(emotion)
        plt.ylim(0, 100)
        plt.grid(alpha=0.3)
        if i > 12:
            plt.xlabel("帧数")

    plt.suptitle("情感维度时间变化曲线", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('emotion_timeseries.png', dpi=300)


# 主处理流程
if __name__ == "__main__":
    # 设置视频路径
    video_path = "wenzeng.mp4"
    output_csv = "blendshapes.csv"

    # 从视频中提取Blendshapes数据
    blendshapes_data = extract_blendshapes_from_video(video_path, output_csv)

    if blendshapes_data is not None:
        # 计算情感得分
        emotion_scores = calculate_emotion_scores(blendshapes_data.drop(columns=['frame', 'timestamp']))

        # 添加时间信息
        emotion_scores['frame'] = blendshapes_data['frame']
        emotion_scores['timestamp'] = blendshapes_data['timestamp']

        # 生成可视化图表
        generate_visualizations(emotion_scores.drop(columns=['frame', 'timestamp']))

        # 输出统计报告
        report = emotion_scores.drop(columns=['frame', 'timestamp']).describe().T
        report['健康范围'] = ['0-30', '0-40', '0-30', '0-35', '0-25',
                          '60-100', '50-100', '60-100', '70-100', '0-30', '0-25', '0-20', '60-100']

        # 添加参考值
        report['参考解释'] = [
            '低值表示平静，高值表示攻击倾向',
            '低值表示放松，高值表示压力大',
            '低值表示安心，高值表示不安',
            '低值表示镇定，高值表示焦虑',
            '低值表示信任，高值表示怀疑',
            '低值表示失衡，高值表示平衡',
            '低值表示疲惫，高值表示精力充沛',
            '低值表示萎靡，高值表示充满活力',
            '低值表示失控，高值表示自控能力强',
            '低值表示表达自由，高值表示情感抑制',
            '低值表示情绪稳定，高值表示神经质',
            '低值表示情绪正常，高值表示抑郁倾向',
            '低值表示不快乐，高值表示幸福'
        ]

        print(report[['mean', 'std', 'min', '50%', 'max', '健康范围', '参考解释']])

        # 保存情感得分结果
        emotion_scores.to_csv('emotion_scores.csv', index=False)
        print("情感分析结果已保存到 emotion_scores.csv")

        # 保存统计报告
        report.to_csv('emotion_report.csv')
        print("情感统计报告已保存到 emotion_report.csv")