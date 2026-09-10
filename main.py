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
# 録音
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

    sample_width = audio.get_sample_size(FORMAT)

    audio.terminate()

    # WAV保存
    with wave.open(OUTPUT_FILE, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))


# =========================
# 音名
# =========================

def frequency_to_note(frequency):

    if frequency is None or frequency <= 0:
        return "-"

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
# 音声読み込み
# =========================

def load_audio():

    y, sr_rate = librosa.load(
        OUTPUT_FILE,
        sr=RATE,
        mono=True
    )

    return y, sr_rate


# =========================
# 基本周波数 F0
# =========================

def analyze_pitch():

    try:

        y, sr_rate = load_audio()

        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=60,
            fmax=1000,
            sr=sr_rate,
            frame_length=2048,
            hop_length=512
        )

        times = librosa.times_like(
            f0,
            sr=sr_rate,
            hop_length=512
        )

        print("F0:")
        print(f0)
        print("最低:", np.nanmin(f0))
        print("最高:", np.nanmax(f0))

        return f0, times

    except Exception as e:

        print("F0解析エラー:", e)

        return (
            np.array([]),
            np.array([])
        )


# =========================
# F0の全体情報
# =========================

def analyze_f0():

    f0, times = analyze_pitch()

    if len(f0) == 0:
        return {
            "current": None,
            "note": "-",
            "min": None,
            "max": None,
            "average": None,
            "times": [],
            "values": []
        }

    valid = f0[
        ~np.isnan(f0) &
        (f0 > 0)
    ]

    if len(valid) == 0:

        return {
            "current": None,
            "note": "-",
            "min": None,
            "max": None,
            "average": None,
            "times": [],
            "values": []
        }

    return {
        "current": float(valid[-1]),
        "note": frequency_to_note(
            float(valid[-1])
        ),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "average": float(np.mean(valid)),
        "times": times.tolist(),
        "values": [
            None if np.isnan(value)
            else float(value)
            for value in f0
        ]
    }


# =========================
# FFT スペクトル
# =========================

def analyze_spectrum():

    try:

        y, sr_rate = load_audio()

        # 音量が大きすぎる場合に備えて正規化
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        # ハニング窓
        window = np.hanning(len(y))

        signal = y * window

        # FFT
        fft_result = np.fft.rfft(signal)

        magnitude = np.abs(fft_result)

        frequencies = np.fft.rfftfreq(
            len(signal),
            1 / sr_rate
        )

        # dB化
        magnitude_db = librosa.amplitude_to_db(
            magnitude,
            ref=np.max
        )

        # 0〜5000Hz程度を表示対象にする
        max_display_frequency = 5000

        mask = frequencies <= max_display_frequency

        return {
            "frequencies":
                frequencies[mask].tolist(),

            "magnitudes":
                magnitude_db[mask].tolist()
        }

    except Exception as e:

        print("スペクトル解析エラー:", e)

        return {
            "frequencies": [],
            "magnitudes": []
        }


# =========================
# 倍音解析
# =========================

def analyze_harmonics():

    try:

        y, sr_rate = load_audio()

        f0, _ = analyze_pitch()

        valid_f0 = f0[
            ~np.isnan(f0) &
            (f0 > 0)
        ]

        if len(valid_f0) == 0:
            return []

        # 基本周波数は中央値
        fundamental = float(
            np.median(valid_f0)
        )

        # FFT
        window = np.hanning(len(y))
        spectrum = np.abs(
            np.fft.rfft(y * window)
        )

        frequencies = np.fft.rfftfreq(
            len(y),
            1 / sr_rate
        )

        harmonics = []

        # 1〜12倍音
        for harmonic_number in range(1, 13):

            target_frequency = (
                fundamental *
                harmonic_number
            )

            if target_frequency >= sr_rate / 2:
                break

            # 対象周波数の±5Hzを見る
            frequency_range = 5

            mask = (
                np.abs(
                    frequencies -
                    target_frequency
                ) <= frequency_range
            )

            if np.any(mask):

                strength = float(
                    np.max(spectrum[mask])
                )

            else:

                strength = 0.0

            harmonics.append({
                "number": harmonic_number,
                "frequency":
                    target_frequency,
                "note":
                    frequency_to_note(
                        target_frequency
                    ),
                "strength": strength
            })

        # 強さを0〜100に正規化
        if harmonics:

            max_strength = max(
                h["strength"]
                for h in harmonics
            )

            if max_strength > 0:

                for h in harmonics:
                    h["strength"] = (
                        h["strength"] /
                        max_strength *
                        100
                    )

        return harmonics

    except Exception as e:

        print("倍音解析エラー:", e)

        return []


# =========================
# 倍音豊かさ
# =========================

def calculate_harmonic_richness(
    harmonics
):

    if not harmonics:
        return 0

    strengths = [
        h["strength"]
        for h in harmonics
    ]

    total = sum(strengths)

    if total <= 0:
        return 0

    # 4倍音以上を高次倍音として扱う
    high_harmonics = sum(
        h["strength"]
        for h in harmonics
        if h["number"] >= 4
    )

    richness = (
        high_harmonics /
        total *
        100
    )

    return float(
        min(100, richness)
    )


# =========================
# 明るさ
# =========================

def calculate_brightness(
    harmonics
):

    if not harmonics:
        return 0

    strengths = [
        h["strength"]
        for h in harmonics
    ]

    total = sum(strengths)

    if total <= 0:
        return 0

    # 3倍音以上の割合
    high_harmonics = sum(
        h["strength"]
        for h in harmonics
        if h["number"] >= 3
    )

    brightness = (
        high_harmonics /
        total *
        100
    )

    return float(
        min(100, brightness)
    )


# =========================
# 声質の傾向
# =========================

def get_voice_type(
    richness
):

    if richness < 25:
        return "倍音が少ない"

    elif richness < 45:
        return "バランス型"

    elif richness < 65:
        return "倍音豊か"

    else:
        return "高次倍音が強い"


# =========================
# 文字ごとのF0
# =========================

def analyze_text_pitch(text):

    f0, times = analyze_pitch()

    if len(f0) == 0 or not text:
        return []

    valid_f0 = f0[
        ~np.isnan(f0) &
        (f0 > 0)
    ]

    if len(valid_f0) == 0:
        return []

    audio_duration = len(
        librosa.load(
            OUTPUT_FILE,
            sr=RATE,
            mono=True
        )[0]
    ) / RATE

    character_duration = (
        audio_duration /
        len(text)
    )

    results = []

    for i, character in enumerate(text):

        start_time = (
            i *
            character_duration
        )

        end_time = (
            (i + 1) *
            character_duration
        )

        mask = (
            (times >= start_time) &
            (times < end_time) &
            ~np.isnan(f0) &
            (f0 > 0)
        )

        character_f0 = f0[mask]

        if len(character_f0) > 0:

            frequency = float(
                np.median(
                    character_f0
                )
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
            "note": note,
            "start": start_time,
            "end": end_time
        })

    return results


# =========================
# 音声認識
# =========================

def recognize_audio():

    recognizer = sr.Recognizer()

    print("音声認識中...")

    try:

        with sr.AudioFile(
            OUTPUT_FILE
        ) as source:

            audio = recognizer.record(
                source
            )

        text = recognizer.recognize_google(
            audio,
            language="ja-JP"
        )

        print("認識結果：")
        print(text)

        return text

    except sr.UnknownValueError:

        print(
            "音声を認識できませんでした"
        )

        return ""

    except sr.RequestError as e:

        print(
            "音声認識サービスに接続できませんでした"
        )

        print(e)

        return ""

    except Exception as e:

        print("音声認識エラー:", e)

        return ""


# =========================
# Web画面
# =========================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# =========================
# START
# =========================

@app.route(
    "/start",
    methods=["POST"]
)
def start_recording():

    global recording
    global recording_thread

    if recording:

        return jsonify({
            "status":
                "already_recording"
        })

    recording = True

    recording_thread = (
        threading.Thread(
            target=record_audio
        )
    )

    recording_thread.start()

    return jsonify({
        "status": "recording"
    })


# =========================
# STOP
# =========================

@app.route(
    "/stop",
    methods=["POST"]
)
def stop_recording():

    global recording
    global recording_thread

    if not recording:

        return jsonify({
            "status":
                "not_recording"
        })

    recording = False

    if recording_thread is not None:

        recording_thread.join()

    # =====================
    # 音声認識
    # =====================

    text = recognize_audio()

    # =====================
    # 音響解析
    # =====================

    f0_data = analyze_f0()

    spectrum_data = analyze_spectrum()

    harmonics = analyze_harmonics()

    richness = (
        calculate_harmonic_richness(
            harmonics
        )
    )

    brightness = (
        calculate_brightness(
            harmonics
        )
    )

    voice_type = get_voice_type(
        richness
    )

    # =====================
    # 文字ごとのF0
    # =====================

    pitch_data = []

    if text:

        pitch_data = (
            analyze_text_pitch(
                text
            )
        )

    return jsonify({

        "status": "stopped",

        "text": text,

        "pitch_data":
            pitch_data,

        "f0":
            f0_data,

        "spectrum":
            spectrum_data,

        "harmonics":
            harmonics,

        "harmonic_richness":
            richness,

        "brightness":
            brightness,

        "voice_type":
            voice_type

    })


# =========================
# RESET
# =========================

@app.route(
    "/reset",
    methods=["POST"]
)
def reset():

    global recording
    global frames

    recording = False
    frames = []

    if os.path.exists(
        OUTPUT_FILE
    ):

        os.remove(
            OUTPUT_FILE
        )

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