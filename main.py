import librosa

audio, sample_rate = librosa.load(
    "test.wav",
    sr=None
)

print("サンプルレート:", sample_rate)
print("音声の長さ:", len(audio) / sample_rate, "秒")