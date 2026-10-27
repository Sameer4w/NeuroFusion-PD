
import os
import io
import zipfile
import numpy as np
import pandas as pd
import librosa
import joblib
import tensorflow as tf
import gradio as gr
import parselmouth
from parselmouth.praat import call

# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))

VOICE_MODEL_PATH = os.path.join(
    BASE, "models", "voice", "voice_model_compatible_v2.keras"
)
VOICE_SCALER_PATH = os.path.join(
    BASE, "models", "voice", "voice_scaler_combined.pkl"
)
HANDWRITING_ARTIFACT_PATH = os.path.join(
    BASE, "models", "handwriting", "handwriting_final_fusion_artifact.joblib"
)
GAIT_MODEL_PATH = os.path.join(
    BASE, "models", "gait", "final_random_forest_model.pkl"
)

# ============================================================
# LOAD MODELS
# ============================================================

voice_model = tf.keras.models.load_model(
    VOICE_MODEL_PATH, compile=False
)
voice_scaler = joblib.load(VOICE_SCALER_PATH)

hw_artifact = joblib.load(HANDWRITING_ARTIFACT_PATH)
hw_model = hw_artifact["model"]
hw_scaler = hw_artifact["scaler"]
hw_features = hw_artifact["features"]
hw_winsor_limits = hw_artifact["winsor_limits"]

gait_model = joblib.load(GAIT_MODEL_PATH)

# ============================================================
# VOICE
# ============================================================

def extract_voice_features(audio, sr=16000):
    features = []

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    rms = librosa.feature.rms(y=audio)
    features.append(np.mean(rms))

    zcr = librosa.feature.zero_crossing_rate(audio)
    features.append(np.mean(zcr))

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.append(np.mean(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features.append(np.mean(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features.append(np.mean(rolloff))

    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features.extend(np.mean(contrast, axis=1))

    harmonic_audio = librosa.effects.harmonic(audio)
    tonnetz = librosa.feature.tonnetz(
        y=harmonic_audio, sr=sr
    )
    features.extend(np.mean(tonnetz, axis=1))

    try:
        f0 = librosa.yin(
            audio, fmin=75, fmax=500, sr=sr
        )
        valid_f0 = f0[np.isfinite(f0)]
        mean_f0 = (
            np.mean(valid_f0)
            if len(valid_f0) > 0
            else 0.0
        )
    except Exception:
        mean_f0 = 0.0

    features.append(mean_f0)

    try:
        sound = parselmouth.Sound(
            audio, sampling_frequency=sr
        )

        point_process = call(
            sound,
            "To PointProcess (periodic, cc)",
            75,
            500
        )

        jitter = call(
            point_process,
            "Get jitter (local)",
            0, 0, 0.0001, 0.02, 1.3
        )

        shimmer = call(
            [sound, point_process],
            "Get shimmer (local)",
            0, 0, 0.0001, 0.02, 1.3, 1.6
        )

        harmonicity = call(
            sound,
            "To Harmonicity (cc)",
            0.01, 75, 0.1, 1.0
        )

        hnr = call(
            harmonicity,
            "Get mean",
            0, 0
        )

        if not np.isfinite(jitter):
            jitter = 0.0
        if not np.isfinite(shimmer):
            shimmer = 0.0
        if not np.isfinite(hnr):
            hnr = 0.0

    except Exception:
        jitter = 0.0
        shimmer = 0.0
        hnr = 0.0

    features.append(jitter)
    features.append(shimmer)
    features.append(hnr)

    return np.array(features, dtype=np.float32)


def predict_voice_file(file_path):

    audio, original_sr = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    if original_sr != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=original_sr,
            target_sr=16000
        )

    segment_length = 5 * 16000

    segments = [
        audio[i:i + segment_length]
        for i in range(
            0,
            len(audio) - segment_length + 1,
            segment_length
        )
    ]

    if not segments:
        raise ValueError(
            "Audio is shorter than 5 seconds."
        )

    probabilities = []

    for segment in segments:

        features = extract_voice_features(
            segment,
            sr=16000
        )

        X = features.reshape(1, -1)

        X_scaled = voice_scaler.transform(X)

        probability = float(
            voice_model.predict(
                X_scaled,
                verbose=0
            )[0][0]
        )

        probabilities.append(probability)

    recording_probability = float(
        np.mean(probabilities)
    )

    return {
        "probability": recording_probability,
        "prediction": int(
            recording_probability >= 0.50
        ),
        "num_segments": len(probabilities),
        "segment_probabilities": probabilities
    }


# ============================================================
# HANDWRITING
# ============================================================

def extract_handwriting_features(file_path):

    df = pd.read_csv(
        file_path,
        sep=";",
        header=None,
        names=[
            "x",
            "y",
            "pressure",
            "azimuth",
            "altitude",
            "timestamp",
            "pen"
        ]
    )

    numeric_cols = [
        "x",
        "y",
        "pressure",
        "azimuth",
        "altitude",
        "timestamp"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=numeric_cols
    )

    x = df["x"].values
    y = df["y"].values
    timestamp = df["timestamp"].values

    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(timestamp)

    valid_dt = (
        np.isfinite(dt)
        & (dt > 0)
        & (dt <= 10)
    )

    movement = np.sqrt(
        dx[valid_dt] ** 2
        + dy[valid_dt] ** 2
    )

    speed = movement / dt[valid_dt]

    direction = np.arctan2(
        dy[valid_dt],
        dx[valid_dt]
    )

    direction_change_mean = (
        np.mean(
            np.abs(
                np.diff(
                    np.unwrap(direction)
                )
            )
        )
        if len(direction) > 1
        else 0.0
    )

    features = {
        "speed_mean": np.mean(speed),
        "zero_movement_ratio": np.mean(
            movement == 0
        ),
        "total_path_length": np.sum(
            movement
        ),
        "speed_max": np.max(speed),
        "movement_mean": np.mean(
            movement
        ),

        # IMPORTANT:
        # The trained artifact's channel2_unique
        # corresponds to azimuth.nunique().
        "channel2_unique": df[
            "azimuth"
        ].nunique(),

        "direction_change_mean":
            direction_change_mean,

        "speed_std": np.std(speed)
    }

    return features


def predict_handwriting_file(file_path):

    feature_dict = (
        extract_handwriting_features(
            file_path
        )
    )

    X = np.array(
        [
            feature_dict[name]
            for name in hw_features
        ],
        dtype=float
    ).reshape(1, -1)

    # Apply the exact saved winsorization limits.
    for i, feature_name in enumerate(
        hw_features
    ):

        limits = hw_winsor_limits[
            feature_name
        ]

        X[0, i] = np.clip(
            X[0, i],
            limits[0],
            limits[1]
        )

    X_scaled = hw_scaler.transform(X)

    probability = float(
        hw_model.predict_proba(
            X_scaled
        )[0, 1]
    )

    return {
        "probability": probability,
        "prediction": int(
            probability >= 0.50
        ),
        "features": feature_dict
    }


# ============================================================
# GAIT
# ============================================================

SENSOR_CHANNELS = [
    "acx",
    "acy",
    "acz",
    "gyrx",
    "gyry",
    "gyrz",
    "roll",
    "pitch",
    "yaw"
]


def extract_gait_features(window):

    features = {}

    for channel in SENSOR_CHANNELS:

        x = window[channel].to_numpy(
            dtype=float
        )

        features[
            f"{channel}_mean"
        ] = np.mean(x)

        features[
            f"{channel}_std"
        ] = np.std(x)

        features[
            f"{channel}_min"
        ] = np.min(x)

        features[
            f"{channel}_max"
        ] = np.max(x)

        features[
            f"{channel}_median"
        ] = np.median(x)

        features[
            f"{channel}_range"
        ] = (
            np.max(x) - np.min(x)
        )

        features[
            f"{channel}_rms"
        ] = np.sqrt(
            np.mean(x ** 2)
        )

    return np.array(
        [
            features[name]
            for name in gait_model.feature_names_in_
        ],
        dtype=np.float64
    )


def predict_gait_file(file_path):

    with zipfile.ZipFile(
        file_path,
        "r"
    ) as z:

        acc = pd.read_csv(
            io.BytesIO(
                z.read("Accelerometer.csv")
            )
        )

        gyr = pd.read_csv(
            io.BytesIO(
                z.read("Gyroscope.csv")
            )
        )

        ori = pd.read_csv(
            io.BytesIO(
                z.read("Orientation.csv")
            )
        )

    acc_data = acc[
        [
            "seconds_elapsed",
            "x",
            "y",
            "z"
        ]
    ].rename(
        columns={
            "x": "acx",
            "y": "acy",
            "z": "acz"
        }
    )

    gyr_data = gyr[
        [
            "seconds_elapsed",
            "x",
            "y",
            "z"
        ]
    ].rename(
        columns={
            "x": "gyrx",
            "y": "gyry",
            "z": "gyrz"
        }
    )

    ori_data = ori[
        [
            "seconds_elapsed",
            "roll",
            "pitch",
            "yaw"
        ]
    ]

    aligned = pd.merge_asof(
        acc_data.sort_values(
            "seconds_elapsed"
        ),
        gyr_data.sort_values(
            "seconds_elapsed"
        ),
        on="seconds_elapsed",
        direction="nearest"
    )

    aligned = pd.merge_asof(
        aligned.sort_values(
            "seconds_elapsed"
        ),
        ori_data.sort_values(
            "seconds_elapsed"
        ),
        on="seconds_elapsed",
        direction="nearest"
    )

    aligned = aligned.dropna(
        subset=SENSOR_CHANNELS
    )

    window_size = 5.0

    start_time = aligned[
        "seconds_elapsed"
    ].min()

    end_time = aligned[
        "seconds_elapsed"
    ].max()

    probabilities = []

    current = start_time

    while current + window_size <= end_time:

        window = aligned[
            (aligned["seconds_elapsed"] >= current)
            & (
                aligned["seconds_elapsed"]
                < current + window_size
            )
        ]

        if len(window) > 0:

            X = extract_gait_features(
                window
            ).reshape(1, -1)

            probability = float(
                gait_model.predict_proba(
                    X
                )[0, 1]
            )

            probabilities.append(
                probability
            )

        current += window_size

    if not probabilities:
        raise ValueError(
            "Gait recording is shorter than 5 seconds."
        )

    recording_probability = float(
        np.mean(probabilities)
    )

    return {
        "probability": recording_probability,
        "prediction": int(
            recording_probability >= 0.50
        ),
        "num_windows": len(probabilities),
        "window_probabilities": probabilities
    }


# ============================================================
# FUSION
# ============================================================

FUSION_WEIGHTS = {
    "voice": 0.9164,
    "handwriting": 0.8644,
    "gait": 0.9953108396
}


def fuse_probabilities(
    modalities,
    method="average"
):

    if not modalities:
        raise ValueError(
            "At least one modality is required."
        )

    probabilities = list(
        modalities.values()
    )

    if method == "average":

        final = np.mean(
            probabilities
        )

    elif method == "max":

        final = np.max(
            probabilities
        )

    elif method == "reliability_weighted":

        available_weights = {
            m: FUSION_WEIGHTS[m]
            for m in modalities
        }

        weight_sum = sum(
            available_weights.values()
        )

        final = sum(
            available_weights[m]
            * modalities[m]
            for m in modalities
        ) / weight_sum

    else:
        raise ValueError(
            "Unknown fusion method."
        )

    return float(final)


def generate_recommendations(
    voice_probability=None,
    handwriting_probability=None,
    gait_probability=None,
    final_probability=None
):
    recommendations = []

    if voice_probability is not None and voice_probability >= 0.50:
        recommendations.append(
            "Voice screening signal is elevated. "
            "If you have persistent changes in voice volume, clarity, "
            "or speech, consider discussing them with a qualified "
            "healthcare professional or speech-language specialist."
        )

    if handwriting_probability is not None and handwriting_probability >= 0.50:
        recommendations.append(
            "Handwriting screening signal is elevated. "
            "If you have noticed changes in handwriting or fine-motor "
            "coordination, consider discussing them with a qualified "
            "healthcare professional or occupational therapist."
        )

    if gait_probability is not None and gait_probability >= 0.50:
        recommendations.append(
            "Gait screening signal is elevated. "
            "If you experience walking, balance, or fall-related concerns, "
            "consider discussing them with a qualified healthcare professional "
            "or considering a clinical mobility or balance assessment."
        )

    if final_probability is not None:
        if final_probability >= 0.50:
            recommendations.append(
                "The combined screening signal is elevated. "
                "Consider discussing the result and any symptoms with "
                "a qualified healthcare professional."
            )
        else:
            recommendations.append(
                "The combined screening signal did not cross the "
                "screening threshold. This result does not rule out "
                "Parkinson's disease."
            )

    recommendations.append(
        "Maintain regular physical activity appropriate for your abilities "
        "and health status. Any exercise program should be individualized."
    )

    recommendations.append(
        "This system is a screening/research prototype and does not "
        "diagnose Parkinson's disease or replace professional medical evaluation."
    )

    return recommendations


def analyze_patient(
    selected_modalities,
    fusion_method,
    voice_file,
    handwriting_file,
    gait_file
):

    if not selected_modalities:
        raise ValueError(
            "Please select at least one modality."
        )

    method_map = {
        "Average": "average",
        "Maximum Probability": "max",
        "Reliability Weighted":
            "reliability_weighted"
    }

    method = method_map[
        fusion_method
    ]

    results = {}

    if "Voice" in selected_modalities:

        if voice_file is None:
            raise ValueError(
                "Voice is selected, but no Voice recording was uploaded."
            )

        results["voice"] = (
            predict_voice_file(
                voice_file
            )["probability"]
        )

    if "Handwriting" in selected_modalities:

        if handwriting_file is None:
            raise ValueError(
                "Handwriting is selected, but no handwriting TXT file was uploaded."
            )

        results["handwriting"] = (
            predict_handwriting_file(
                handwriting_file
            )["probability"]
        )

    if "Gait" in selected_modalities:

        if gait_file is None:
            raise ValueError(
                "Gait is selected, but no Gait ZIP file was uploaded."
            )

        results["gait"] = (
            predict_gait_file(
                gait_file
            )["probability"]
        )

    final_probability = fuse_probabilities(
        results,
        method
    )

    configuration = " + ".join(
        selected_modalities
    )

    recommendations = generate_recommendations(
        voice_probability=results.get("voice"),
        handwriting_probability=results.get("handwriting"),
        gait_probability=results.get("gait"),
        final_probability=final_probability
    )

    return {
        "configuration": configuration,
        "modalities_used": list(
            results.keys()
        ),
        "fusion_method": method,
        "voice_probability":
            results.get("voice"),
        "handwriting_probability":
            results.get("handwriting"),
        "gait_probability":
            results.get("gait"),
        "final_probability":
            final_probability,
        "prediction":
            int(final_probability >= 0.50),
        "recommendations":
            recommendations
    }


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="NeuroFusion-PD"
) as demo:

    gr.Markdown(
        "# 🧠 NeuroFusion-PD"
    )

    gr.Markdown(
        "Multi-modal Parkinson's disease screening "
        "using Voice, Handwriting, and Gait."
    )

    modality = gr.CheckboxGroup(
        choices=[
            "Voice",
            "Handwriting",
            "Gait"
        ],
        label="Select available modalities",
        info="Select one or more modalities."
    )

    fusion_method = gr.Radio(
        choices=[
            "Average",
            "Maximum Probability",
            "Reliability Weighted"
        ],
        value="Average",
        label="Fusion Method"
    )

    voice_input = gr.Audio(
        label="Voice Recording",
        type="filepath",
        format="wav"
    )

    handwriting_input = gr.File(
        label="Handwriting Trajectory (.txt)",
        file_types=[".txt"],
        type="filepath"
    )

    gait_input = gr.File(
        label="Gait Sensor Recording (.zip)",
        file_types=[".zip"],
        type="filepath"
    )

    predict_button = gr.Button(
        "🔍 Analyze for Parkinson's",
        variant="primary"
    )

    result_output = gr.JSON(
        label="Prediction Results"
    )

    predict_button.click(
        fn=analyze_patient,
        inputs=[
            modality,
            fusion_method,
            voice_input,
            handwriting_input,
            gait_input
        ],
        outputs=result_output
    )


if __name__ == "__main__":
    demo.launch(
        share=True,
        debug=True
    )



