import os
import subprocess
import uuid

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

openface_bin = r"D:\Programs\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"


def sigmoid(x):
    """Sigmoid函数增强非线性特征"""
    return 1 / (1 + np.exp(-x))


def calculate_emotions(au_data):
    """
    基于17个AU特征值计算12维情感指标
    输入: au_data - 包含17个AU值的字典
    输出: 包含12个情感维度的字典
    """
    # ========== 基础预处理 ==========
    # 确保所有AU值为非负数
    au_values = {k: max(0, v) for k, v in au_data.items()}

    # ========== 情感维度计算 ==========
    emotions = {}

    # 攻击性 (Aggressiveness)
    emotions['敌意'] = sigmoid(
        0.8 * au_values['AU04_r'] +  # 眉毛下压
        0.6 * au_values['AU23_r'] +  # 嘴唇紧缩
        0.5 * au_values['AU05_r']  # 上眼睑上抬
    ) / 2.5

    # 压力 (Stress)
    emotions['压力'] = min(1.0,
                         0.7 * au_values['AU07_r'] +  # 眼睑收紧
                         0.5 * au_values['AU45_r'] +  # 眨眼频率
                         0.4 * au_values['AU04_r']  # 眉毛下压
                         ) / 2.2

    # 不安 (Restlessness)
    emotions['焦虑'] = sigmoid(
        0.6 * au_values['AU02_r'] +  # 眉毛上抬
        0.5 * au_values['AU20_r']  # 嘴唇拉伸
    ) / 1.8

    # 怀疑 (Suspicion)
    emotions['稳定性'] = (
                             0.7 * au_values['AU04_r'] +  # 眉毛下压
                             0.5 * au_values['AU14_r']  # 酒窝
                     ) / 2.0

    # 平衡 (Balance)
    emotions['协调性'] = sigmoid(
        0.6 * au_values['AU12_r'] +  # 嘴角上提（正向）
        0.5 * au_values['AU25_r'] -  # 嘴唇分开（正向）
        0.4 * au_values['AU04_r']  # 眉毛下压（负向）
    ) / 1.8

    # 自信 (Confidence)
    emotions['自信'] = min(1.0,
                         0.7 * au_values['AU12_r'] +  # 嘴角上提
                         0.6 * au_values['AU06_r'] +  # 脸颊上提
                         0.4 * au_values['AU25_r']  # 嘴唇分开
                         ) / 2.2

    # 能量 (Energy)
    emotions['活跃度'] = (
                             0.7 * au_values['AU05_r'] +  # 上眼睑上抬
                             0.6 * au_values['AU26_r']  # 下颌下降
                     ) / 1.8

    # 自我调节 (Self-regulation)
    emotions['自我调节'] = 1.0 - min(1.0,
                                 0.6 * au_values['AU10_r'] +  # 上唇上抬
                                 0.4 * au_values['AU17_r']  # 下巴抬起
                                 )

    # 抑制 (Inhibition)
    emotions['自控力'] = sigmoid(
        0.8 * au_values['AU14_r'] +  # 酒窝
        0.6 * au_values['AU17_r']  # 下巴抬起
    ) / 2.0

    # 神经质 (Neuroticism)
    emotions['易激性'] = min(1.0,
                          0.5 * au_values['AU01_r'] +  # 眉毛内角上抬
                          0.6 * au_values['AU04_r'] +  # 眉毛下压
                          0.4 * au_values['AU15_r']  # 嘴角下降
                          ) / 2.0

    # 抑郁 (Depression)
    emotions['低迷值'] = (
                             0.8 * au_values['AU15_r'] +  # 嘴角下降
                             0.7 * au_values['AU17_r']  # 下巴抬起
                     ) / 2.0

    # 幸福 (Happiness)
    emotions['幸福'] = sigmoid(
        1.5 * au_values['AU06_r'] +  # 脸颊上提
        1.8 * au_values['AU12_r'] -  # 嘴角上提
        0.8 * au_values['AU15_r']  # 抑制因子(嘴角下降)
    ) / 3.5

    # ========== 后处理 ==========
    # 确保所有值在[0,1]区间
    for k, v in emotions.items():
        emotions[k] = max(0.0, min(1.0, v))

    return emotions


def process_video(video_path, output_dir="./output"):
    """
    处理视频文件：提取AU特征并计算情感维度
    :param video_path: 输入视频文件路径
    :param output_dir: 输出目录
    :return: 视频级情感分析结果DataFrame
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 步骤1: 使用OpenFace提取面部特征
    print(f"正在使用OpenFace处理视频: {video_path}")
    openface_path = openface_bin
    csv_file = os.path.join(
        output_dir, os.path.basename(video_path).replace(".mp4", ".csv")
    )

    # 执行OpenFace命令
    cmd = [openface_path, "-f", video_path, "-out_dir", output_dir, "-aus"]
    subprocess.run(cmd, check=True)

    # 检查输出文件
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"OpenFace输出文件未找到: {csv_file}")

    # 步骤2: 加载并预处理AU数据
    print("加载并预处理AU数据...")
    au_df = pd.read_csv(csv_file)
    au_df.columns = [col.strip() for col in au_df.columns]

    # 检查必需的AU列
    required_aus = [
        "AU01_r",
        "AU02_r",
        "AU04_r",
        "AU05_r",
        "AU06_r",
        "AU07_r",
        "AU09_r",
        "AU10_r",
        "AU12_r",
        "AU14_r",
        "AU15_r",
        "AU17_r",
        "AU20_r",
        "AU23_r",
        "AU25_r",
        "AU26_r",
        "AU45_r",
    ]

    for au in required_aus:
        if au not in au_df.columns:
            raise ValueError(f"缺少必需的AU列: {au}")

    # 填充缺失值
    au_df[required_aus] = au_df[required_aus].fillna(method="ffill").fillna(0)

    # 步骤3: 计算每帧的情感维度
    print("计算帧级情感维度...")
    frame_emotions = []

    for _, row in au_df.iterrows():
        au_data = {au: row[au] for au in required_aus}
        emotions = calculate_emotions(au_data)
        frame_emotions.append(emotions)

    emotions_df = pd.DataFrame(frame_emotions)

    # 添加时间戳
    if "timestamp" in au_df.columns:
        emotions_df["timestamp"] = au_df["timestamp"]
    else:
        frame_rate = 30  # 默认帧率
        emotions_df["timestamp"] = emotions_df.index / frame_rate

    # 步骤4: 时序平滑处理
    print("应用时序平滑...")
    emotion_cols = emotions_df.columns.drop("timestamp")

    # 使用Savitzky-Golay滤波器进行平滑
    window_size = min(15, len(emotions_df) // 4)  # 动态窗口大小
    if window_size % 2 == 0:  # 确保窗口大小为奇数
        window_size = max(3, window_size - 1)

    for col in emotion_cols:
        # 双重平滑：先高斯后SG
        smoothed = gaussian_filter1d(emotions_df[col], sigma=1)
        emotions_df[f"{col}_smoothed"] = savgol_filter(
            smoothed, window_length=window_size, polyorder=2
        )

    # 步骤5: 计算视频级情感指标
    print("计算视频级情感指标...")
    video_summary = {}

    for col in emotion_cols:
        # 基本统计量
        video_summary[f"{col}_mean"] = emotions_df[f"{col}_smoothed"].mean()
        video_summary[f"{col}_std"] = emotions_df[f"{col}_smoothed"].std()
        video_summary[f"{col}_max"] = emotions_df[f"{col}_smoothed"].max()
        video_summary[f"{col}_min"] = emotions_df[f"{col}_smoothed"].min()

        # 峰值检测 - 情感爆发点
        peaks = emotions_df[f"{col}_smoothed"][
            (
                emotions_df[f"{col}_smoothed"]
                > emotions_df[f"{col}_smoothed"].mean()
                + 1.5 * emotions_df[f"{col}_smoothed"].std()
            )
        ]
        video_summary[f"{col}_peak_count"] = len(peaks)
        if len(peaks) > 0:
            video_summary[f"{col}_peak_mean"] = peaks.mean()

    # 情感平衡分析
    positive_emotions = ["自信", "活跃度", "幸福", "协调性"]
    negative_emotions = ["敌意", "压力", "焦虑", "低迷值", "易激性"]

    # video_summary["positive_mean"] = emotions_df[positive_emotions].mean(axis=1).mean()
    # video_summary["negative_mean"] = emotions_df[negative_emotions].mean(axis=1).mean()
    # video_summary["emotional_balance"] = video_summary["positive_mean"] / (
    #     video_summary["positive_mean"] + video_summary["negative_mean"] + 1e-6
    # )

    # 步骤6: 保存结果和可视化
    print("生成结果可视化...")
    # 保存原始数据
    emotions_df.to_csv(os.path.join(output_dir, "emotion_analysis.csv"), index=False)

    # 情感曲线图
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(emotion_cols, 1):
        plt.subplot(4, 4, i)
        plt.plot(emotions_df["timestamp"], emotions_df[f"{col}_smoothed"])
        plt.title(col)
        plt.xlabel("时间 (秒)")
        plt.ylabel("强度")
        plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "emotion_timeline.png"))

    # 情感分布图
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=emotions_df[emotion_cols])
    plt.title("情感维度分布")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, "emotion_distribution.png"))

    # 情感相关性热力图
    plt.figure(figsize=(12, 10))
    corr_matrix = emotions_df[emotion_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("情感维度相关性")
    plt.savefig(os.path.join(output_dir, "emotion_correlation.png"))

    # 情感雷达图（视频平均）
    summary_df = pd.DataFrame([video_summary])
    radar_cols = [c for c in summary_df.columns if "_mean" in c and "_peak" not in c]
    radar_data = summary_df[radar_cols].values.flatten()
    radar_labels = [c.replace("_mean", "") for c in radar_cols]

    angles = np.linspace(0, 2 * np.pi, len(radar_data), endpoint=False).tolist()
    radar_data = np.concatenate((radar_data, [radar_data[0]]))
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, radar_data, linewidth=2, linestyle="solid")
    ax.fill(angles, radar_data, alpha=0.25)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], radar_labels)
    plt.title("情感维度雷达图 (视频平均)", y=1.1)
    plt.savefig(os.path.join(output_dir, "emotion_radar.png"))

    # 保存摘要结果
    summary_df.to_csv(
        os.path.join(output_dir, "video_emotion_summary.csv"), index=False
    )

    print(f"分析完成! 结果保存至: {output_dir}")
    return summary_df


if __name__ == "__main__":
    # 使用示例
    video_path = "wenzeng.mp4"
    # video_path = "yunfei.mp4"
    # video_path = "lvjiao.mp4"
    # video_path = "zhangmi.mp4"
    # video_path = "qiaoyafeng.mp4"
    batch_no = f"{uuid.uuid4().hex}"
    result = process_video(video_path, output_dir=f"./output/{batch_no}")

    print(f"\n视频情感分析:")
    print(result.T)
