# NeuroFusion-PD V4 Inspection Report

## A. Existing architecture

The supplied ZIP contains 36 archive entries. The important runtime artifacts are:

- `neurofusion_app.py` — the current all-in-one Gradio application, including model loading, all three preprocessing pipelines, fusion, recommendations and UI.
- `neurofusion_final_ui.py` — a second minimal Gradio UI that duplicates the interface wiring but does not contain the inference implementation itself.
- `models/voice/voice_model_compatible_v2.keras` — saved Keras voice model.
- `models/voice/voice_scaler_combined.pkl` — saved voice feature scaler.
- `models/handwriting/handwriting_final_fusion_artifact.joblib` — saved handwriting model, scaler, feature order and winsorization limits.
- `models/gait/final_random_forest_model.pkl` — saved gait Random Forest pipeline.
- `models/gait/feature_information.pkl`, `final_model_selection.pkl`, `final_project_summary.pkl`, `final_test_results.pkl` — gait research metadata/artifacts.
- `fusion/neurofusion_engine.py` — reusable fusion engine.
- `fusion/fusion_functions.py` — reusable fusion helpers.
- `metadata/*.json` and `fusion_21_case_evaluation.csv` — validation/research metadata.
- `datasets/gait/X_gait_sample.csv` — one already-extracted 63-feature gait sample; it is not a raw Sensor Logger recording.
- `datasets/handwriting/v2_fixed_features_all_files.csv` — processed handwriting feature table.

There was no `requirements.txt`, authentication implementation, `.env.example`, Dockerfile, or permanent deployment configuration in the supplied package.

## B. Existing ML pipeline

### Voice

`audio -> librosa load -> resample to 16 kHz -> non-overlapping 5-second segments -> 54 acoustic features -> saved StandardScaler -> saved Keras model -> segment probabilities -> arithmetic mean`

The 54 features are: 20 MFCC means, 12 chroma means, RMS, zero-crossing rate, spectral centroid, bandwidth, rolloff, 7 spectral-contrast means, 6 tonnetz means, mean F0, jitter, shimmer and HNR.

The saved `.keras` file declares an input shape of 54 features. The model file metadata reports Keras 3.15.1.

### Handwriting

`semicolon-separated TXT -> seven named fields -> numeric cleaning -> trajectory differences -> speed/movement/direction features -> 8 saved feature order -> saved winsorization limits -> saved scaler -> saved classifier -> probability`

The trained features are:

1. `speed_mean`
2. `zero_movement_ratio`
3. `total_path_length`
4. `speed_max`
5. `movement_mean`
6. `channel2_unique` (specifically `azimuth.nunique()` in the current code)
7. `direction_change_mean`
8. `speed_std`

The saved artifact maps class 1 to Parkinson and class 0 to Control.

### Gait

`ZIP -> Accelerometer.csv + Gyroscope.csv + Orientation.csv -> select/rename sensor columns -> timestamp nearest-neighbor alignment -> 5-second windows -> 63 statistical features -> saved Random Forest pipeline -> window probabilities -> mean`

The model consumes 63 features: 7 statistics each for 9 channels (`acx`, `acy`, `acz`, `gyrx`, `gyry`, `gyrz`, `roll`, `pitch`, `yaw`).

The current code requires the three CSV files at the ZIP root and does not support a direct raw CSV.

## C. Existing fusion

The application uses a 0.50 threshold.

- Average = mean of the probabilities that are actually available.
- Maximum Probability = maximum of the available probabilities.
- Reliability Weighted = weighted mean using the current stored weights:
  - Voice `0.9164`
  - Handwriting `0.8644`
  - Gait `0.9953108396`

Missing modalities are excluded. They are not converted to zero.

The metadata explicitly says the three modality datasets are not participant-paired and that paired multimodal performance is not established. The reliability weights are provisional.

## D. Existing Gradio UI

The supplied screenshot matches the current UI design:

- Checkbox selection at the top.
- Fusion method selection.
- All three upload components are rendered immediately, even when only one modality is selected.
- Analyze button.
- Raw JSON output.
- No authentication.
- No gait collection guidance.
- No user-friendly result card.
- Minimal styling.

The V4 launcher also uses `demo.launch(share=True, debug=True)`, which creates the temporary `gradio.live` workflow and requires the Colab runtime to remain alive.

## E. Model files

| Modality | Artifact | Type | Expected features |
|---|---|---|---:|
| Voice | `voice_model_compatible_v2.keras` | Keras Sequential | 54 |
| Voice | `voice_scaler_combined.pkl` | scaler | 54 |
| Handwriting | `handwriting_final_fusion_artifact.joblib` | model + scaler + metadata | 8 |
| Gait | `final_random_forest_model.pkl` | sklearn Pipeline / Random Forest | 63 |

The gait model is a 500-tree Random Forest pipeline.

## F. Dependencies

The original application imports:

- NumPy
- pandas
- librosa
- joblib
- TensorFlow/Keras
- Gradio
- praat-parselmouth
- scikit-learn

The supplied package did not include a requirements file. The upgraded project adds one and pins scikit-learn to 1.6.1 because the serialized sklearn artifacts were trained with 1.6.1. The deployment TensorFlow target is 2.21.0, which supports Python 3.13 according to the current TensorFlow/PyPI release information.

## G. Deployment problems

1. The Colab launcher uploads a ZIP into `/content` and extracts it on every session.
2. The public URL is generated through `share=True` and is temporary.
3. The app has no authentication.
4. The app has no environment/secrets management.
5. There is no reproducible requirements file.
6. The raw Gradio JSON is exposed to users.
7. Upload controls are not conditional.
8. Invalid files can produce low-level exceptions instead of clear validation messages.
9. The current gait ZIP structure is undocumented in the UI.
10. `fusion/fusion_functions.py` uses NumPy without importing it; this is fixed as a small non-ML bugfix in V5.
11. The current application loads all model artifacts at module import. V5 separates the artifact loaders and caches them per process, while preserving model behavior.
12. The package contains no raw gait ZIP suitable for end-to-end validation; only an already-extracted 63-feature gait sample is included.

## Local verification performed

- All original Python files compile successfully with `py_compile`.
- Voice `.keras` archive was inspected without altering it; it declares a 54-feature input.
- Handwriting artifact loads successfully and exposes the saved 8-feature order and winsorization limits.
- Gait Random Forest artifact loads successfully and exposes 63 feature names.
- Handwriting feature extraction was tested with a valid synthetic trajectory and invalid TXT input; malformed input is rejected.
- Gait feature/model execution was tested with a synthetic 3-file, 5+ second ZIP; the saved Random Forest produced a probability, confirming the adapter and feature ordering execute.
- Invalid gait ZIP and missing-column cases were rejected with validation errors.
- The full V4 application could not be launched in this sandbox because TensorFlow was not installed and this environment has no outbound package-install network access. This is an environment limitation, not a claim that the supplied Keras model is broken.

## H. Recommended architecture

### Application

Use **Streamlit** as the user-facing application framework. It provides the interactive controls needed for dynamic modality selection, polished CSS, progress/status components and a straightforward local development command.

### Authentication

Use **Supabase Auth** for email/password registration, login and logout. The application only uses the public anon key and never needs a service-role key. Supabase Auth handles password security.

### Deployment

Use a **Docker-capable ML host**, with Hugging Face Spaces Docker as the recommended target for this project. The application is an ML inference workload and needs more memory headroom than the current Streamlit Community Cloud limits may comfortably provide for TensorFlow. Hugging Face Spaces provides a stable `hf.space` URL; sleeping affects availability time, not the identity of the URL.

### ML boundary

Keep the trained artifacts untouched. Move only the inference code into `inference/pipeline.py`, with the same feature extraction, thresholds and fusion mathematics. Model loading is cached per process and uploaded files are temporary.
