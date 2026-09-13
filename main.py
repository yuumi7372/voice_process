from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pyaudio
import wave
import threading
import os
import librosa
import numpy as np
import whisperx


app = Flask(__name__)
CORS(app)


# =========================
# 録音設定
# =========================

#FORMAT = pyaudio.paInt16
#CHANNELS = 1
RATE = 44100
#CHUNK = 1024

OUTPUT_FILE = "voice_test.wav"


# =========================
# WhisperX設定
# =========================

# GPUが使えるなら cuda
# CPUなら cpu
DEVICE = "cpu"
ALIGN_DEVICE = "xpu"

# CPUの場合は int8
# GPUなら float16 が基本
COMPUTE_TYPE = "int8"

WHISPER_MODEL = "small"


# =========================
# 録音状態
# =========================

recording = False
recording_thread = None
frames = []


# =========================
# WhisperXモデル
# =========================

whisper_model = None
align_model = None
align_metadata = None


def load_whisper_models():

    global whisper_model
    global align_model
    global align_metadata

    if whisper_model is None:

        print("WhisperXモデルを読み込み中...")

        whisper_model = whisperx.load_model(
            WHISPER_MODEL,
            DEVICE,
            compute_type=COMPUTE_TYPE
        )

        print("WhisperXモデル読み込み完了")


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

    sample_width = audio.get_sample_size(
        FORMAT
    )

    audio.terminate()

    # WAV保存

    with wave.open(
        OUTPUT_FILE,
        "wb"
    ) as wf:

        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(
            b"".join(frames)
        )


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
        69 + 12 * np.log2(
            frequency / 440
        )
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

        "current":
            float(valid[-1]),

        "note":
            frequency_to_note(
                float(valid[-1])
            ),

        "min":
            float(np.min(valid)),

        "max":
            float(np.max(valid)),

        "average":
            float(np.mean(valid)),

        "times":
            times.tolist(),

        "values": [

            None
            if np.isnan(value)
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

        if np.max(np.abs(y)) > 0:

            y = y / np.max(
                np.abs(y)
            )

        window = np.hanning(
            len(y)
        )

        signal = y * window

        fft_result = np.fft.rfft(
            signal
        )

        magnitude = np.abs(
            fft_result
        )

        frequencies = np.fft.rfftfreq(
            len(signal),
            1 / sr_rate
        )

        magnitude_db = librosa.amplitude_to_db(
            magnitude,
            ref=np.max
        )

        max_display_frequency = 5000

        mask = (
            frequencies <=
            max_display_frequency
        )

        return {

            "frequencies":
                frequencies[mask].tolist(),

            "magnitudes":
                magnitude_db[mask].tolist()

        }

    except Exception as e:

        print(
            "スペクトル解析エラー:",
            e
        )

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

        fundamental = float(
            np.median(valid_f0)
        )

        window = np.hanning(
            len(y)
        )

        spectrum = np.abs(
            np.fft.rfft(
                y * window
            )
        )

        frequencies = np.fft.rfftfreq(
            len(y),
            1 / sr_rate
        )

        harmonics = []

        for harmonic_number in range(
            1,
            13
        ):

            target_frequency = (
                fundamental *
                harmonic_number
            )

            if (
                target_frequency >=
                sr_rate / 2
            ):

                break

            frequency_range = 5

            mask = (
                np.abs(
                    frequencies -
                    target_frequency
                ) <=
                frequency_range
            )

            if np.any(mask):

                strength = float(
                    np.max(
                        spectrum[mask]
                    )
                )

            else:

                strength = 0.0

            harmonics.append({

                "number":
                    harmonic_number,

                "frequency":
                    target_frequency,

                "note":
                    frequency_to_note(
                        target_frequency
                    ),

                "strength":
                    strength

            })

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

        print(
            "倍音解析エラー:",
            e
        )

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
# WhisperXによる音声認識
# =========================

def recognize_audio():

    try:

        load_whisper_models()

        print("音声認識中...")

        audio = whisperx.load_audio(
            OUTPUT_FILE
        )

        AUDIO_SR = 16000

        print("audio samples:", len(audio))
        print("audio duration:", len(audio) / AUDIO_SR)

        result = whisper_model.transcribe(
            audio,
            language="ja",
            batch_size=4
        )

        print("認識結果:")

        for segment in result["segments"]:

            print(
                segment["start"],
                "〜",
                segment["end"],
                segment["text"]
            )

        return result

    except Exception as e:

        print(
            "WhisperX音声認識エラー:",
            e
        )

        return {
            "segments": [],
            "language": "ja"
        }


# =========================
# 文字ごとの時間情報＋音量
# =========================

def analyze_text_alignment(
    result
):

    global align_model
    global align_metadata

    if not result.get("segments"):

        return []

    try:

        # 音声読み込み
        audio = whisperx.load_audio(
            OUTPUT_FILE
        )

        AUDIO_SR = 16000

        # =========================
        # 日本語用アライメントモデル
        # =========================

        if align_model is None:

            print(
                "日本語アライメントモデルを読み込み中..."
            )

            align_model, align_metadata = whisperx.load_align_model(
                language_code="ja",
                device=ALIGN_DEVICE
            )

        # =========================
        # 文字単位でアライメント
        # =========================

        aligned = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            ALIGN_DEVICE,
            return_char_alignments=True
        )

        results = []

        # =========================
        # 文字ごとの情報を取得
        # =========================

        print("ALIGN RESULT:")
        print(aligned["segments"])

        for segment in aligned["segments"]:

            chars = segment.get(
                "chars",
                []
            )

            for char_data in chars:

                character = char_data.get(
                    "char",
                    ""
                )

                start = char_data.get(
                    "start"
                )

                end = char_data.get(
                    "end"
                )

                if (
                    start is None or
                    end is None
                ):

                    continue

                start = float(start)
                end = float(end)

                # =========================
                # その文字に対応する音声を切り出す
                # =========================

                start_sample = int(start * AUDIO_SR)
                end_sample = int(end * AUDIO_SR)

                # 音声配列の範囲内に収める
                start_sample = max(0, min(start_sample, len(audio)))
                end_sample = max(0, min(end_sample, len(audio)))

                segment_audio = audio[start_sample:end_sample]

                if len(segment_audio) == 0:
                    print(
                        character,
                        "samples: 0",
                        "time:", start, "〜", end
                    )
                    continue

                print(
                    character,
                    "samples:", len(segment_audio),
                    "min:", np.min(segment_audio),
                    "max:", np.max(segment_audio),
                    "max_abs:", np.max(np.abs(segment_audio))
                )
                # =========================
                # 音量（RMS）を計算
                # =========================

                if len(segment_audio) > 0:

                    rms = np.sqrt(
                        np.mean(
                            segment_audio ** 2
                        )
                    )

                    # dBFSに変換
                    if rms > 0:

                        volume = 20 * np.log10(
                            rms
                        )

                    else:

                        volume = -100.0

                else:

                    volume = -100.0

                results.append({

                    "character":
                        character,

                    "start":
                        start,

                    "end":
                        end,

                    "volume":
                        float(volume)

                })

        # =========================
        # 結果表示
        # =========================

        print("文字ごとの時間＋音量:")

        for item in results:

            print(
                item["character"],
                item["start"],
                "〜",
                item["end"],
                "音量:",
                item["volume"],
                "dBFS"
            )

        return results

    except Exception as e:

        print(
            "アライメントエラー:",
            e
        )

        return []

# =========================
# 文字 × F0
# =========================

def analyze_text_pitch(
    alignment_data
):

    f0, times = analyze_pitch()

    if (
        len(f0) == 0 or
        not alignment_data
    ):

        return []

    results = []

    for item in alignment_data:

        start_time = item["start"]
        end_time = item["end"]

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

            "character":
                item["character"],

            "frequency":
                frequency,

            "note":
                note,

            "start":
                start_time,

            "end":
                end_time

        })

    return results


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
        "status":
            "recording"
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

    transcription = recognize_audio()

    text = "".join(
        segment["text"]
        for segment
        in transcription.get(
            "segments",
            []
        )
    )

    # =====================
    # 文字アライメント
    # =====================

    alignment_data = (
        analyze_text_alignment(
            transcription
        )
    )

    # =====================
    # 音響解析
    # =====================

    f0_data = analyze_f0()

    spectrum_data = (
        analyze_spectrum()
    )

    harmonics = (
        analyze_harmonics()
    )

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
    # 文字 × F0
    # =====================

    pitch_data = (
        analyze_text_pitch(
            alignment_data
        )
    )

    return jsonify({

        "status":
            "stopped",

        "text":
            text,

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
        "status":
            "reset"
    })

#フロントからWAVを受診して解析するエンドポイント
@app.route("/upload", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "音声ファイルがありません"}), 400

    audio_file = request.files["audio"]
    audio_file.save(OUTPUT_FILE)
    
    #音声認識
    transcription = recognize_audio()
    text = "".join(
        segment["text"]
        for segment
        in transcription.get(
            "segments",
            []
        )
    )
    #文字アライメント
    alignment_data = analyze_text_alignment(transcription)
    #音響解析
    f0_data = analyze_f0()
    spectrum_data = analyze_spectrum()
    harmonics = analyze_harmonics()
    richness = calculate_harmonic_richness(harmonics)
    brightness = calculate_brightness(harmonics)
    voice_type = get_voice_type(richness)
    #文字
    pitch_data = analyze_text_pitch(alignment_data)
    return jsonify({
        "status": "success",
        "text": text,
        "pitch_data": pitch_data,
        "f0": f0_data,
        "spectrum": spectrum_data,
        "harmonics": harmonics,
        "harmonic_richness": richness,
        "brightness": brightness,
        "voice_type": voice_type
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