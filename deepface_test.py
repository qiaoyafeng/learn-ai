from deepface import DeepFace
import cv2

video_path = "wenzeng.mp4"
cap = cv2.VideoCapture(video_path)

frame_rate = 30  # 每5帧分析一次
frame_idx = 0

backends = [
    'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
    'retinaface', 'mediapipe', 'yolov8', 'yolov11s',
    'yolov11n', 'yolov11m', 'yunet', 'centerface',
]
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
            result = DeepFace.analyze(frame, actions=actions,  detector_backend=detector, align=align,enforce_detection=False)
            print(f"result: {result}")
            print(f"Frame {frame_idx} 情绪: {result[0]['emotion']}")
        except Exception as e:
            print(f"跳过帧 {frame_idx}，原因：{e}")
    frame_idx += 1

cap.release()



composite_emotions ={}