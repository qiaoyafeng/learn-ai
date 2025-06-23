from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm


# 视频处理函数
def process_video(video_path, output_csv='blendshapes.csv'):
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
        blendshapes_df.to_csv(output_csv, index=False)
        print(f"Blendshapes数据已保存到: {output_csv}")
        return blendshapes_df
    else:
        print("未检测到任何面部数据")
        return None


if __name__ == "__main__":
    video_directory = "./"
    blendshapes_directory = "./blendshapes"
    videos = Path(video_directory).rglob('*.mp4')
    for video in videos:
        index = 0
        print(f"{index}-------{video}")
        output_csv = Path(f"{blendshapes_directory}/{video.stem}.csv")
        # 处理视频并提取BlendShapes
        process_video(video, output_csv)
        index += 1

