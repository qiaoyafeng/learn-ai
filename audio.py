import wave

import librosa
import soundfile

wavFile = "yao.wav"
f = wave.open(wavFile)
# 音频头 参数
params = f.getparams()
Channels = f.getnchannels()
SampleRate = f.getframerate()
bit_type = f.getsampwidth() * 8
frames = f.getnframes()
# Duration 也就是音频时长 = 采样点数/采样率
Duration = wav_time = frames / float(SampleRate)  # 单位为s

print("音频头参数：", params)
print("通道数(Channels)：", Channels)
print("采样率(SampleRate)：", SampleRate)
print("比特(Precision)：", bit_type)
print("采样点数(frames)：", frames)
print("帧数或者时间(Duration)：", Duration)


# #####################转换采样率###############



filename = r"hxq_mix_8k.wav"  # 源文件
newFilename = r"hxq_mix_16k.wav"  # 新采样率保存的文件

y, sr = librosa.load(filename, sr=8000)  # 读取8k的音频文件
y_16 = librosa.resample(y, orig_sr=sr, target_sr=16000)  # 采样率转化

# 在0.8.0以后的版本，librosa都会将这个函数删除
# librosa.output.write_wav(newFilename, y_16, 16000)
# 推荐用下面的函数进行文件保存
soundfile.write(newFilename, y_16, 16000)  # 重新采样的音频文件保存
