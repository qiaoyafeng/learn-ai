from deepface import DeepFace
import cv2
import numpy as np


backends = [
    'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
    'retinaface', 'mediapipe', 'yolov8', 'yolov11s',
    'yolov11n', 'yolov11m', 'yunet', 'centerface',
]
detector = backends[6]
print(f"detector: {detector}")
align = True

actions = ("emotion", "age", "gender", "race")

video_path = "wenzeng.mp4"


def analyze_video_test(video_path=video_path):
    frame_rate = 30  # 每5帧分析一次
    frame_idx = 0
    cap = cv2.VideoCapture(video_path)
    detector = backends[6]
    print(f"detector: {detector}")
    align = True

    actions = ("emotion", "age", "gender", "race")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_rate == 0:
            try:
                result = DeepFace.analyze(frame, actions=actions, detector_backend=detector, align=align,
                                          enforce_detection=False)
                print(f"result: {result}")
                print(f"Frame {frame_idx} 情绪: {result[0]['emotion']}")
            except Exception as e:
                print(f"跳过帧 {frame_idx}，原因：{e}")
        frame_idx += 1

    cap.release()


def compute_emotion_indices_from_au(au_dict):
    # 输入为一帧 AU 特征的字典，例如 {"AU01_r": 0.5, "AU04_r": 0.2, ...}
    def v(name):
        return au_dict.get(name, 0.0)

    return {
        "攻击性": 0.5 * v("AU04_r") + 0.3 * v("AU07_r") + 0.2 * v("AU23_r"),
        "压力": 0.4 * v("AU01_r") + 0.4 * v("AU05_r") + 0.2 * v("AU24_r"),
        "不安": 0.3 * v("AU02_r") + 0.3 * v("AU05_r") + 0.4 * v("AU14_r"),
        "怀疑": 0.5 * v("AU14_r") + 0.3 * v("AU04_r") + 0.2 * v("AU45_r"),
        "平衡": 0.6 * v("AU06_r") + 0.4 * v("AU12_r"),
        "自信": 0.5 * v("AU12_r") + 0.3 * v("AU10_r") + 0.2 * (1 - v("AU15_r")),
        "能量": 0.4 * v("AU05_r") + 0.3 * v("AU26_r") + 0.3 * v("AU06_r"),
        "自我调节": 0.5 * v("AU12_r") + 0.3 * (1 - v("AU01_r")) + 0.2 * (1 - v("AU04_r")),
        "抑制": 0.4 * v("AU15_r") + 0.3 * v("AU20_r") + 0.3 * v("AU23_r"),
        "神经质": 0.3 * v("AU01_r") + 0.3 * v("AU04_r") + 0.4 * v("AU07_r"),
        "抑郁": 0.4 * v("AU01_r") + 0.3 * v("AU15_r") + 0.3 * v("AU04_r"),
        "幸福": 0.6 * v("AU12_r") + 0.4 * v("AU06_r")
    }

# 自定义十二维情绪组合权重
def compute_emotion_indices(basic_emotions):
    """
    basic_emotions: dict, keys=[angry, disgust, fear, happy, sad, surprise, neutral]
    returns: dict, 12情绪维度分数
    """
    A = basic_emotions

    indices = {
        "攻击性": A["angry"] + 0.5 * A["disgust"],
        "压力": A["fear"] + 0.5 * A["sad"] + 0.5 * A["angry"],
        "不安": A["fear"] + A["surprise"] + 0.5 * A["sad"],
        "怀疑": A["disgust"] + 0.5 * A["neutral"] + 0.5 * A["angry"],
        "平衡": A["neutral"] + 0.5 * A["happy"] - 0.5 * A["sad"],
        "自信": A["happy"] + 0.5 * A["neutral"] - 0.5 * A["fear"],
        "能量": A["surprise"] + A["angry"] + 0.5 * A["happy"],
        "自我调节": A["neutral"] + 0.5 * A["sad"] + 0.5 * A["happy"],
        "抑制": A["neutral"] + A["sad"] - A["happy"],
        "神经质": A["fear"] + A["angry"] + A["sad"],
        "抑郁": A["sad"] + 0.5 * A["disgust"] - 0.5 * A["happy"],
        "幸福": A["happy"] - A["sad"] - A["disgust"]
    }

    return indices


# 单图像处理
def analyze_image(image_path):
    result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
    emotion_scores = result[0]['emotion']
    emotion_scores = {k.lower(): v for k, v in emotion_scores.items()}
    indices = compute_emotion_indices(emotion_scores)
    return indices


# 视频处理（每秒N帧取平均）
def analyze_video(video_path, frame_rate=5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps / frame_rate)

    all_emotions = []

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % frames_to_skip == 0:
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion_scores = result[0]['emotion']
                emotion_scores = {k.lower(): v for k, v in emotion_scores.items()}
                all_emotions.append(emotion_scores)
            except Exception as e:
                print("Skipping frame due to error:", e)
        i += 1
    cap.release()

    # 平均各帧情绪
    if not all_emotions:
        return None
    avg_emotions = {k: np.mean([e[k] for e in all_emotions]) for k in all_emotions[0]}
    indices = compute_emotion_indices(avg_emotions)
    return indices

if __name__ == "__main__":

    image_path = "image.png"
    image_results = analyze_image(image_path)
    print("图片十二维情绪指数：")
    for k, v in image_results.items():
        print(f"{k}: {abs(v)}")
    print(f"{'*' * 50}")
    video_path = "wenzeng.mp4"
    video_results = analyze_video(video_path)
    print("视频十二维情绪指数：")
    for k, v in video_results.items():
        print(f"{k}: {abs(v)}")



