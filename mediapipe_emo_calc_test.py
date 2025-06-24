import os
import uuid

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib

matplotlib.use("TkAgg")

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 情感维度映射配置 (特征: 权重)
EMOTION_MAPPING = {
    '攻击性': {
        'browDownLeft': 0.5, 'browDownRight': 0.5,
        'eyeWideLeft': 0.5, 'eyeWideRight': 0.5
    },
    '压力': {
        'browInnerUp': 0.3, 'eyeSquintLeft': 0.2, 'eyeSquintRight': 0.2,
        'mouthFrownLeft': 0.15, 'mouthFrownRight': 0.15
    },
    '不安': {
        'eyeLookDownLeft': 0.2, 'eyeLookDownRight': 0.2, 'mouthDimpleLeft': 0.15,
        'mouthDimpleRight': 0.15, 'browOuterUpLeft': 0.15, 'browOuterUpRight': 0.15
    },
    '怀疑': {
        'eyeLookInLeft': 0.25, 'eyeLookInRight': 0.25, 'mouthUpperUpLeft': 0.15,
        'mouthUpperUpRight': 0.15, 'noseSneerLeft': 0.1, 'noseSneerRight': 0.1
    },
    '平衡': {
        '_neutral': 0.5, 'mouthSmileLeft': 0.15, 'mouthSmileRight': 0.15,
        'browInnerUp': 0.2
    },
    '自信': {
        'eyeLookUpLeft': 0.2, 'eyeLookUpRight': 0.2, 'mouthSmileLeft': 0.2,
        'mouthSmileRight': 0.2, 'jawForward': 0.2
    },
    '能量': {
        'eyeWideLeft': 0.2, 'eyeWideRight': 0.2, 'mouthSmileLeft': 0.2,
        'mouthSmileRight': 0.2, 'browInnerUp': 0.2
    },
    '自我调节': {
        'mouthPucker': 0.3, 'mouthFunnel': 0.3, 'eyeSquintLeft': 0.2, 'eyeSquintRight': 0.2
    },
    '抑制': {
        'mouthClose': 0.5, 'jawForward': 0.5
    },
    '神经质': {
        'eyeBlinkLeft': 0.2, 'eyeBlinkRight': 0.2, 'browDownLeft': 0.2,
        'browDownRight': 0.2, 'mouthFrownLeft': 0.1, 'mouthFrownRight': 0.1
    },
    '抑郁': {
        'browDownLeft': 0.25, 'browDownRight': 0.25, 'eyeLookDownLeft': 0.1,
        'eyeLookDownRight': 0.1, 'mouthFrownLeft': 0.15, 'mouthFrownRight': 0.15
    },
    '幸福': {
        'mouthSmileLeft': 0.3, 'mouthSmileRight': 0.3, 'cheekSquintLeft': 0.1,
        'cheekSquintRight': 0.1, 'eyeSquintLeft': 0.1, 'eyeSquintRight': 0.1
    }
}

# 视频处理函数
def process_video(video_path, output_dir="./output"):
    """
        从视频中提取面部Blendshapes数据并保存为CSV文件
        使用MediaPipe FaceLandmarker的VIDEO模式
        """

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
        output_csv = f'{output_dir}/blendshapes.csv'
        blendshapes_df.to_csv(output_csv, index=False)
        print(f"Blendshapes数据已保存到: {output_csv}")
        return blendshapes_df
    else:
        print("未检测到任何面部数据")
        return None


# 情感维度计算
def calculate_emotions(blendshapes_df):
    emotions_df = pd.DataFrame()

    for emotion, features in EMOTION_MAPPING.items():
        total_weight = sum(features.values())
        weighted_scores = []

        for idx, row in blendshapes_df.iterrows():
            score_sum = 0
            valid_features = 0

            for feature, weight in features.items():
                if feature in row and not pd.isna(row[feature]):
                    score_sum += row[feature] * weight
                    valid_features += weight

            # 标准化处理
            if valid_features > 0:
                normalized_score = min(100, max(0, (score_sum / valid_features) * 100))
            else:
                normalized_score = np.nan

            weighted_scores.append(normalized_score)

        emotions_df[emotion] = weighted_scores

    # 添加时间戳
    emotions_df['timestamp'] = blendshapes_df['timestamp']
    emotions_df['frame'] = blendshapes_df['frame']
    return emotions_df.dropna()


# 可视化函数（使用中位数）
def visualize_emotions(emotions_df, output_dir="./output"):
    # 计算参考范围（基于整个视频的情感分数）
    reference_ranges = {}
    for emotion in EMOTION_MAPPING.keys():
        data = emotions_df[emotion].dropna()
        median = data.median()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        reference_ranges[emotion] = {
            'low': max(0, q1 - 1.5 * iqr),
            'normal_low': q1,
            'normal_high': q3,
            'high': min(100, q3 + 1.5 * iqr)
        }

    # 1. 雷达图（带参考范围）
    plt.figure(figsize=(12, 10))
    median_emotions = emotions_df.drop(columns=["timestamp", "frame"]).median(numeric_only=True)
    categories = list(EMOTION_MAPPING.keys())
    N = len(categories)

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合多边形

    # 创建极坐标轴
    ax = plt.subplot(111, polar=True)

    # 绘制参考范围区域
    low_range = [reference_ranges[cat]['low'] for cat in categories] + [reference_ranges[categories[0]]['low']]
    normal_low_range = [reference_ranges[cat]['normal_low'] for cat in categories] + [
        reference_ranges[categories[0]]['normal_low']]
    normal_high_range = [reference_ranges[cat]['normal_high'] for cat in categories] + [
        reference_ranges[categories[0]]['normal_high']]
    high_range = [reference_ranges[cat]['high'] for cat in categories] + [reference_ranges[categories[0]]['high']]

    # 填充低范围区域 (0 - 低边界)
    ax.fill_between(angles, 0, low_range, color='lightcoral', alpha=0.3, label='低范围')

    # 填充正常范围区域 (正常低边界 - 正常高边界)
    ax.fill_between(angles, normal_low_range, normal_high_range, color='lightgreen', alpha=0.4, label='中间范围')

    # 填充高范围区域 (高边界 - 100)
    ax.fill_between(angles, high_range, [100] * len(angles), color='gold', alpha=0.3, label='高范围')

    # 绘制中位情感分数
    values = median_emotions.values.tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, markersize=8, color='dodgerblue', label='情感分数')
    ax.fill(angles, values, color='dodgerblue', alpha=0.2)

    # 设置坐标轴
    plt.xticks(angles[:-1], categories, fontsize=10)
    plt.yticks([0, 20, 40, 60, 80, 100], ["0", "20", "40", "60", "80", "100"], fontsize=8)
    plt.ylim(0, 100)

    # 添加标题和图例
    plt.title('情感维度雷达图（含参考范围）', fontsize=14, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    # 添加参考范围说明
    # plt.figtext(0.5, 0.01,
    #             f"参考范围说明（基于视频数据计算）:\n"
    #             f"低范围: 0-{np.median([ref['low'] for ref in reference_ranges.values()]):.1f} "
    #             f"正常范围: {np.median([ref['normal_low'] for ref in reference_ranges.values()]):.1f}-"
    #             f"{np.median([ref['normal_high'] for ref in reference_ranges.values()]):.1f} "
    #             f"高范围: {np.median([ref['high'] for ref in reference_ranges.values()]):.1f}-100",
    #             ha="center", fontsize=9, bbox={"facecolor": "white", "alpha": 0.7, "pad": 5})

    plt.savefig(f'{output_dir}/emotion_radar_with_ranges.png', bbox_inches='tight')
    plt.close()

    # 2. 箱线图（使用中位数）
    plt.figure(figsize=(14, 8))
    # 箱线图自动显示中位数，无需额外设置
    emotions_df.drop(columns=['timestamp', 'frame']).boxplot(showmeans=False,
                                                             medianprops={'color': 'blue', 'linewidth': 2})
    plt.title('情感维度分布箱线图', fontsize=14)
    plt.ylabel('分值 (0-100)', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(alpha=0.3)

    # 添加中位数说明
    plt.figtext(0.5, 0.01,
                "箱线图说明：箱体表示四分位距(IQR)，中间线表示中位数，须线表示1.5倍IQR范围",
                ha="center", fontsize=9, bbox={"facecolor": "white", "alpha": 0.7, "pad": 5})

    plt.savefig(f'{output_dir}/emotion_boxplot.png', bbox_inches='tight')
    plt.close()

    # 3. 时间变化曲线（使用滚动中位数）
    plt.figure(figsize=(14, 10))
    time_seconds = emotions_df['timestamp'] / 1000

    # 计算滚动中位数（窗口大小为15帧）
    rolling_median = emotions_df.copy()
    for emotion in EMOTION_MAPPING.keys():
        rolling_median[emotion] = rolling_median[emotion].rolling(window=15, min_periods=1).median()

    for i, emotion in enumerate(EMOTION_MAPPING.keys()):
        plt.subplot(4, 3, i + 1)

        # 原始值（浅色）
        plt.plot(time_seconds, emotions_df[emotion], color='lightsteelblue', alpha=0.4, label='原始值')

        # 滚动中位数（深色）
        plt.plot(time_seconds, rolling_median[emotion], color='steelblue', linewidth=2, label='滚动中位数')

        # 添加参考范围线
        plt.axhline(y=reference_ranges[emotion]['normal_low'], color='green', linestyle='--', alpha=0.5)
        plt.axhline(y=reference_ranges[emotion]['normal_high'], color='green', linestyle='--', alpha=0.5)
        plt.axhspan(reference_ranges[emotion]['normal_low'], reference_ranges[emotion]['normal_high'],
                    facecolor='lightgreen', alpha=0.2)

        # 添加中位数线
        plt.axhline(y=reference_ranges[emotion]['normal_low'] + (
                    reference_ranges[emotion]['normal_high'] - reference_ranges[emotion]['normal_low']) / 2,
                    color='red', linestyle='-', alpha=0.3, linewidth=1)

        plt.title(emotion, fontsize=12)
        plt.xlabel('时间 (秒)', fontsize=9)
        plt.ylabel('强度', fontsize=9)
        plt.ylim(0, 100)
        plt.grid(alpha=0.2)

        # 只在第一张图添加图例
        if i == 0:
            plt.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/emotion_timeline.png')
    plt.close()

    # 4. 变化斜率图（基于中位数）
    plt.figure(figsize=(14, 8))
    slopes_df = pd.DataFrame()

    # 使用滚动中位数计算斜率
    for emotion in EMOTION_MAPPING.keys():
        # 计算一阶导数（变化斜率）
        smoothed_values = rolling_median[emotion].values
        slopes = np.gradient(smoothed_values, emotions_df['timestamp'].values)
        slopes_df[emotion] = slopes

        # 可视化变化率
        plt.subplot(3, 4, list(EMOTION_MAPPING.keys()).index(emotion) + 1)
        plt.plot(time_seconds, slopes, color='purple')
        plt.axhline(y=0, color='r', linestyle='-')

        # 添加斜率阈值线
        plt.axhline(y=0.2, color='orange', linestyle=':', alpha=0.5)
        plt.axhline(y=-0.2, color='orange', linestyle=':', alpha=0.5)

        plt.title(f'{emotion}变化率', fontsize=10)
        plt.xlabel('时间 (秒)', fontsize=8)
        plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/emotion_slopes.png')
    plt.close()
    return slopes_df, reference_ranges


# 主处理流程
if __name__ == "__main__":
    batch_no = f"{uuid.uuid4().hex}"
    output_dir = f"./output/{batch_no}"
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    # 处理视频并提取BlendShapes
    video_path = 'zhangmi.mp4'
    # video_path = 'portrait.mp4'
    video_path = 'yunfei.mp4'
    video_path = 'qiao.mp4'
    blendshapes_df = process_video(video_path, output_dir=output_dir)

    # 计算情感维度
    emotions_df = calculate_emotions(blendshapes_df)
    emotions_df.to_csv(f'{output_dir}/emotion_scores.csv', index=False)

    # 可视化结果（返回参考范围）
    slopes_df, reference_ranges = visualize_emotions(emotions_df, output_dir=output_dir)

    # 保存参考范围到文件（基于中位数）
    with open(f'{output_dir}/reference_ranges.txt', 'w') as f:
        f.write("情感维度参考范围 (基于视频数据分析，使用中位数和四分位数):\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'情感维度':<20}{'中位数':<10}{'低范围':<15}{'中间范围':<20}{'高范围':<15}\n")
        f.write("-" * 70 + "\n")

        for emotion, ranges in reference_ranges.items():
            median_val = emotions_df[emotion].median()
            low = f"{float(ranges['low']):.2f}"
            normal_low = f"{float(ranges['normal_low']):.2f}"
            normal_high = f"{float(ranges['normal_high']):.2f}"
            normal_low_high = normal_low + "-" + normal_high
            high = f"{float(ranges['high']):.2f}"

            f.write(f"{emotion:<20}{median_val:<10.2f}{low:<15}")
            f.write(f"{normal_low_high :<20}")
            f.write(f"{high :<15}\n")
    print("处理完成！结果文件已保存。")