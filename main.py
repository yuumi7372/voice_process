from flask import Flask, render_template, jsonify
import pyaudio
import wave
import speech_recognition as sr
import threading
import os
import librosa
import numpy as np


app = Flask(__name__)


# =========================
# 録音設定
# =========================

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

OUTPUT_FILE = "recorded.wav"


# =========================
# 録音状態
# =========================

recording = False
recording_thread = None
frames = []


# =========================
# 録音処理
# =========================

def record_audio():
    global recording
    global frames

    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    frames = []

    print("録音開始！")

    while recording:
        try:
            data = stream.read(
                CHUNK,
                exception_on_overflow=False
            )
            frames.append(data)

        except Exception as e:
            print("録音エラー:", e)
            break

    print("録音終了！")

    stream.stop_stream()
    stream.close()

    # WAVとして保存
    with wave.open(OUTPUT_FILE, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(
            audio.get_sample_size(FORMAT)
        )
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    audio.terminate()


# =========================
# 音名に変換
# =========================

def frequency_to_note(frequency):
    if frequency is None or frequency <= 0:
        return "-"

    # A4 = 440Hz
    note_names = [
        "C", "C#", "D", "D#",
        "E", "F", "F#", "G",
        "G#", "A", "A#", "B"
    ]

    midi = round(
        69 + 12 * np.log2(frequency / 440)
    )

    note = note_names[midi % 12]
    octave = midi // 12 - 1

    return f"{note}{octave}"


# =========================
# 基本周波数（F0）解析
# =========================

def analyze_pitch():

    print("基本周波数を解析中...")

    try:
        # WAV読み込み
        y, sr_rate = librosa.load(
            OUTPUT_FILE,
            sr=RATE,
            mono=True
        )

        # YINによる基本周波数推定
        f0 = librosa.yin(
            y,
            fmin=60,
            fmax=1000,
            sr=sr_rate,
            frame_length=2048,
            hop_length=512
        )

        # フレームごとの時間
        times = librosa.times_like(
            f0,
            sr=sr_rate,
            hop_length=512
        )

        return f0, times

    except Exception as e:

        print("基本周波数解析エラー:", e)

        return np.array([]), np.array([])


# =========================
# 文字ごとの基本周波数
# =========================

def analyze_text_pitch(text):

    f0, times = analyze_pitch()

    if len(f0) == 0:
        return []

    # 音声の長さ
    audio_duration = times[-1]

    # 文字ごとの時間幅
    character_duration = (
        audio_duration / len(text)
    )

    results = []

    for i, character in enumerate(text):

        start_time = (
            i * character_duration
        )

        end_time = (
            (i + 1) * character_duration
        )

        # この文字に対応すると考えるF0を取得
        mask = (
            (times >= start_time) &
            (times < end_time) &
            ~np.isnan(f0)
        )

        character_f0 = f0[mask]

        if len(character_f0) > 0:

            # 中央値を代表値にする
            frequency = float(
                np.median(character_f0)
            )

            note = frequency_to_note(
                frequency
            )

        else:

            frequency = None
            note = "-"

        results.append({
            "character": character,
            "frequency": frequency,
            "note": note
        })

    return results


# =========================
# 音声認識
# =========================

def recognize_audio():

    recognizer = sr.Recognizer()

    print("音声認識中...")

    try:

        with sr.AudioFile(OUTPUT_FILE) as source:

            audio = recognizer.record(source)

        text = recognizer.recognize_google(
            audio,
            language="ja-JP"
        )

        print("認識結果：")
        print(text)

        return text

    except sr.UnknownValueError:

        print("音声を認識できませんでした")

        return ""

    except sr.RequestError as e:

        print("音声認識サービスに接続できませんでした")
        print(e)

        return ""

    except Exception as e:

        print("エラー:", e)

        return ""


# =========================
# Web画面
# =========================

@app.route("/")
def index():

    return render_template("index.html")


# =========================
# START
# =========================

@app.route("/start", methods=["POST"])
def start_recording():

    global recording
    global recording_thread

    if recording:

        return jsonify({
            "status": "already_recording"
        })

    recording = True

    recording_thread = threading.Thread(
        target=record_audio
    )

    recording_thread.start()

    return jsonify({
        "status": "recording"
    })


# =========================
# STOP
# =========================

@app.route("/stop", methods=["POST"])
def stop_recording():

    global recording
    global recording_thread

    if not recording:

        return jsonify({
            "status": "not_recording"
        })

    recording = False

    # 録音終了を待つ
    if recording_thread is not None:

        recording_thread.join()

    # 音声認識
    text = recognize_audio()

    if not text:

        return jsonify({
            "status": "stopped",
            "text": "",
            "pitch_data": []
        })

    # 基本周波数解析
    pitch_data = analyze_text_pitch(text)

    return jsonify({
        "status": "stopped",
        "text": text,
        "pitch_data": pitch_data
    })


# =========================
# RESET
# =========================

@app.route("/reset", methods=["POST"])
def reset():

    global recording
    global frames

    recording = False
    frames = []

    if os.path.exists(OUTPUT_FILE):

        os.remove(OUTPUT_FILE)

    return jsonify({
        "status": "reset"
    })


# =========================
# 起動
# =========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )