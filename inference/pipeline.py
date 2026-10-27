"""NeuroFusion-PD inference pipeline.

Deployment-safe extraction of the existing V4 inference logic. Model
artifacts, feature extraction, thresholds and fusion mathematics are preserved.
"""
from __future__ import annotations

import zipfile
from functools import lru_cache
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

BASE = Path(__file__).resolve().parents[1]
VOICE_MODEL_PATH = BASE / "models" / "voice" / "voice_model_compatible_v2.keras"
VOICE_SCALER_PATH = BASE / "models" / "voice" / "voice_scaler_combined.pkl"
HANDWRITING_ARTIFACT_PATH = BASE / "models" / "handwriting" / "handwriting_final_fusion_artifact.joblib"
GAIT_MODEL_PATH = BASE / "models" / "gait" / "final_random_forest_model.pkl"
FUSION_WEIGHTS = {"voice": 0.9164, "handwriting": 0.8644, "gait": 0.9953108396}
THRESHOLD = 0.50
SENSOR_CHANNELS = ["acx", "acy", "acz", "gyrx", "gyry", "gyrz", "roll", "pitch", "yaw"]
GAIT_REQUIRED_FILES = ("Accelerometer.csv", "Gyroscope.csv", "Orientation.csv")

@lru_cache(maxsize=1)
def load_voice_artifacts():
    import tensorflow as tf
    voice_model = tf.keras.models.load_model(str(VOICE_MODEL_PATH), compile=False)
    voice_scaler = joblib.load(VOICE_SCALER_PATH)
    return voice_model, voice_scaler

@lru_cache(maxsize=1)
def load_classical_artifacts():
    hw_artifact = joblib.load(HANDWRITING_ARTIFACT_PATH)
    gait_model = joblib.load(GAIT_MODEL_PATH)
    return (hw_artifact["model"], hw_artifact["scaler"], hw_artifact["features"],
            hw_artifact["winsor_limits"], gait_model)

@lru_cache(maxsize=1)
def load_models():
    voice_model, voice_scaler = load_voice_artifacts()
    hw_model, hw_scaler, hw_features, hw_winsor_limits, gait_model = load_classical_artifacts()
    return voice_model, voice_scaler, hw_model, hw_scaler, hw_features, hw_winsor_limits, gait_model

def extract_voice_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    features = []
    features.extend(np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20), axis=1))
    features.extend(np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1))
    features.append(np.mean(librosa.feature.rms(y=audio)))
    features.append(np.mean(librosa.feature.zero_crossing_rate(audio)))
    features.append(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
    features.extend(np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr), axis=1))
    harmonic_audio = librosa.effects.harmonic(audio)
    features.extend(np.mean(librosa.feature.tonnetz(y=harmonic_audio, sr=sr), axis=1))
    try:
        f0 = librosa.yin(audio, fmin=75, fmax=500, sr=sr)
        valid_f0 = f0[np.isfinite(f0)]
        mean_f0 = np.mean(valid_f0) if len(valid_f0) else 0.0
    except Exception:
        mean_f0 = 0.0
    features.append(mean_f0)
    try:
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        jitter = jitter if np.isfinite(jitter) else 0.0
        shimmer = shimmer if np.isfinite(shimmer) else 0.0
        hnr = hnr if np.isfinite(hnr) else 0.0
    except Exception:
        jitter = shimmer = hnr = 0.0
    features.extend([jitter, shimmer, hnr])
    result = np.asarray(features, dtype=np.float32)
    if result.shape != (54,) or not np.all(np.isfinite(result)):
        raise ValueError("Voice preprocessing produced an invalid feature vector.")
    return result

def predict_voice_file(file_path: str) -> dict:
    try:
        audio, original_sr = librosa.load(file_path, sr=None, mono=True)
    except Exception as exc:
        raise ValueError("Invalid voice file. Please upload a readable supported audio recording.") from exc
    if audio is None or len(audio) == 0 or original_sr is None:
        raise ValueError("The uploaded voice recording is empty or unreadable.")
    if not np.all(np.isfinite(audio)):
        raise ValueError("The uploaded voice recording contains invalid audio data.")
    if original_sr != 16000:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=16000)
    segment_length = 5 * 16000
    segments = [audio[i:i + segment_length] for i in range(0, len(audio) - segment_length + 1, segment_length)]
    if not segments:
        raise ValueError("Voice recording must be at least 5 seconds long.")
    voice_model, voice_scaler = load_voice_artifacts()
    probabilities = []
    for segment in segments:
        features = extract_voice_features(segment, sr=16000)
        X_scaled = voice_scaler.transform(features.reshape(1, -1))
        probability = float(voice_model.predict(X_scaled, verbose=0)[0][0])
        if not 0 <= probability <= 1:
            raise ValueError("Voice model returned an invalid probability.")
        probabilities.append(probability)
    recording_probability = float(np.mean(probabilities))
    return {"probability": recording_probability, "prediction": int(recording_probability >= THRESHOLD), "num_segments": len(probabilities)}

def extract_handwriting_features(file_path: str) -> dict:
    try:
        df = pd.read_csv(file_path, sep=";", header=None,
                         names=["x", "y", "pressure", "azimuth", "altitude", "timestamp", "pen"])
    except Exception as exc:
        raise ValueError("Invalid handwriting file. Please upload the required semicolon-separated .txt trajectory file.") from exc
    numeric_cols = ["x", "y", "pressure", "azimuth", "altitude", "timestamp"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)
    if len(df) < 3:
        raise ValueError("The handwriting trajectory does not contain enough valid points.")
    x, y, timestamp = df["x"].values, df["y"].values, df["timestamp"].values
    dx, dy, dt = np.diff(x), np.diff(y), np.diff(timestamp)
    valid_dt = np.isfinite(dt) & (dt > 0) & (dt <= 10)
    if not np.any(valid_dt):
        raise ValueError("The handwriting trajectory has no valid increasing timestamps.")
    movement = np.sqrt(dx[valid_dt] ** 2 + dy[valid_dt] ** 2)
    speed = movement / dt[valid_dt]
    direction = np.arctan2(dy[valid_dt], dx[valid_dt])
    direction_change_mean = float(np.mean(np.abs(np.diff(np.unwrap(direction))))) if len(direction) > 1 else 0.0
    features = {
        "speed_mean": float(np.mean(speed)), "zero_movement_ratio": float(np.mean(movement == 0)),
        "total_path_length": float(np.sum(movement)), "speed_max": float(np.max(speed)),
        "movement_mean": float(np.mean(movement)), "channel2_unique": float(df["azimuth"].nunique()),
        "direction_change_mean": direction_change_mean, "speed_std": float(np.std(speed)),
    }
    if not np.all(np.isfinite(list(features.values()))):
        raise ValueError("The handwriting trajectory contains invalid numeric values.")
    return features

def predict_handwriting_file(file_path: str) -> dict:
    feature_dict = extract_handwriting_features(file_path)
    hw_model, hw_scaler, hw_features, hw_winsor_limits, _ = load_classical_artifacts()
    X = np.array([feature_dict[name] for name in hw_features], dtype=float).reshape(1, -1)
    for i, feature_name in enumerate(hw_features):
        limits = hw_winsor_limits[feature_name]
        X[0, i] = np.clip(X[0, i], limits[0], limits[1])
    X_scaled = hw_scaler.transform(X)
    probability = float(hw_model.predict_proba(X_scaled)[0, 1])
    return {"probability": probability, "prediction": int(probability >= THRESHOLD)}

def _read_zip_csv(z: zipfile.ZipFile, filename: str) -> pd.DataFrame:
    try:
        with z.open(filename) as f:
            return pd.read_csv(f)
    except KeyError as exc:
        raise ValueError(f"Gait ZIP is missing {filename}.") from exc
    except Exception as exc:
        raise ValueError(f"Gait ZIP contains an unreadable {filename}.") from exc

def predict_gait_file(file_path: str) -> dict:
    if not zipfile.is_zipfile(file_path):
        raise ValueError("Invalid gait file. Please upload a ZIP exported in the required Sensor Logger format.")
    try:
        with zipfile.ZipFile(file_path, "r") as z:
            names = set(z.namelist())
            missing = [name for name in GAIT_REQUIRED_FILES if name not in names]
            if missing:
                raise ValueError("Gait ZIP is missing: " + ", ".join(missing) + ".")
            acc, gyr, ori = (_read_zip_csv(z, name) for name in GAIT_REQUIRED_FILES)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The gait ZIP could not be read. Please export a fresh Sensor Logger recording.") from exc
    for name, df, required in [
        ("Accelerometer.csv", acc, {"seconds_elapsed", "x", "y", "z"}),
        ("Gyroscope.csv", gyr, {"seconds_elapsed", "x", "y", "z"}),
        ("Orientation.csv", ori, {"seconds_elapsed", "roll", "pitch", "yaw"}),
    ]:
        if not required.issubset(df.columns):
            raise ValueError(f"{name} is missing required columns: {', '.join(sorted(required - set(df.columns)))}.")
        if len(df) < 2:
            raise ValueError(f"{name} does not contain enough sensor samples.")
    acc_data = acc[["seconds_elapsed", "x", "y", "z"]].rename(columns={"x": "acx", "y": "acy", "z": "acz"})
    gyr_data = gyr[["seconds_elapsed", "x", "y", "z"]].rename(columns={"x": "gyrx", "y": "gyry", "z": "gyrz"})
    ori_data = ori[["seconds_elapsed", "roll", "pitch", "yaw"]]
    for df in (acc_data, gyr_data, ori_data):
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)
        if df.empty:
            raise ValueError("Gait sensor data contains no valid numeric samples.")
    aligned = pd.merge_asof(acc_data.sort_values("seconds_elapsed"), gyr_data.sort_values("seconds_elapsed"), on="seconds_elapsed", direction="nearest")
    aligned = pd.merge_asof(aligned.sort_values("seconds_elapsed"), ori_data.sort_values("seconds_elapsed"), on="seconds_elapsed", direction="nearest")
    aligned = aligned.dropna(subset=SENSOR_CHANNELS)
    if aligned.empty:
        raise ValueError("The three gait sensors could not be aligned. Check their timestamps.")
    window_size = 5.0
    start_time, end_time = float(aligned["seconds_elapsed"].min()), float(aligned["seconds_elapsed"].max())
    if end_time - start_time < window_size:
        raise ValueError("Gait recording must contain at least 5 seconds of aligned sensor data.")
    _, _, _, _, gait_model = load_classical_artifacts()
    probabilities, current = [], start_time
    while current + window_size <= end_time:
        window = aligned[(aligned["seconds_elapsed"] >= current) & (aligned["seconds_elapsed"] < current + window_size)]
        if len(window) > 0:
            features = {}
            for channel in SENSOR_CHANNELS:
                x = window[channel].to_numpy(dtype=float)
                features[f"{channel}_mean"] = np.mean(x); features[f"{channel}_std"] = np.std(x)
                features[f"{channel}_min"] = np.min(x); features[f"{channel}_max"] = np.max(x)
                features[f"{channel}_median"] = np.median(x); features[f"{channel}_range"] = np.max(x) - np.min(x)
                features[f"{channel}_rms"] = np.sqrt(np.mean(x ** 2))
            X = np.array([features[name] for name in gait_model.feature_names_in_], dtype=np.float64).reshape(1, -1)
            probability = float(gait_model.predict_proba(X)[0, 1])
            if not 0 <= probability <= 1:
                raise ValueError("Gait model returned an invalid probability.")
            probabilities.append(probability)
        current += window_size
    if not probabilities:
        raise ValueError("No valid 5-second gait windows were found.")
    recording_probability = float(np.mean(probabilities))
    return {"probability": recording_probability, "prediction": int(recording_probability >= THRESHOLD), "num_windows": len(probabilities)}

def fuse_probabilities(modalities: dict[str, float], method: str = "average") -> float:
    if not modalities:
        raise ValueError("At least one modality is required.")
    probabilities = list(modalities.values())
    if method == "average":
        final = np.mean(probabilities)
    elif method == "max":
        final = np.max(probabilities)
    elif method == "reliability_weighted":
        available_weights = {m: FUSION_WEIGHTS[m] for m in modalities}
        weight_sum = sum(available_weights.values())
        final = sum(available_weights[m] * modalities[m] for m in modalities) / weight_sum
    else:
        raise ValueError("Unknown fusion method.")
    return float(final)

def generate_recommendations(final_probability: float) -> list[str]:
    if final_probability >= THRESHOLD:
        return [
            "A Parkinson's screening signal was detected. Please discuss this result and any symptoms with a qualified healthcare professional.",
            "Clinical evaluation should consider symptoms, medical history and professional assessment rather than this screening score alone.",
            "This is an AI-based screening/research prototype and does not diagnose Parkinson's disease.",
        ]
    return [
        "No significant Parkinson's screening signal was detected by this analysis.",
        "A negative screening result does not rule out Parkinson's disease. Please consult a healthcare professional if you have symptoms or concerns.",
        "This is an AI-based screening/research prototype and does not diagnose Parkinson's disease.",
    ]

def analyze_patient(selected_modalities, fusion_method, voice_file=None, handwriting_file=None, gait_file=None) -> dict:
    if not selected_modalities:
        raise ValueError("Please select at least one modality.")
    method_map = {"Average": "average", "Maximum Probability": "max", "Reliability Weighted": "reliability_weighted"}
    if fusion_method not in method_map:
        raise ValueError("Please select a valid fusion method.")
    results = {}
    if "Voice" in selected_modalities:
        if not voice_file: raise ValueError("Voice is selected. Please upload a voice recording.")
        results["voice"] = predict_voice_file(voice_file)["probability"]
    if "Handwriting" in selected_modalities:
        if not handwriting_file: raise ValueError("Handwriting is selected. Please upload a handwriting trajectory TXT file.")
        results["handwriting"] = predict_handwriting_file(handwriting_file)["probability"]
    if "Gait" in selected_modalities:
        if not gait_file: raise ValueError("Gait is selected. Please upload the required gait ZIP file.")
        results["gait"] = predict_gait_file(gait_file)["probability"]
    final_probability = fuse_probabilities(results, method_map[fusion_method])
    return {"final_probability": final_probability, "prediction": int(final_probability >= THRESHOLD), "recommendations": generate_recommendations(final_probability)}
