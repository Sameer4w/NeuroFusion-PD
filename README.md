---
title: NeuroFusion-PD
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
---

# 🧠 NeuroFusion-PD

NeuroFusion-PD is a multimodal Parkinson's disease **screening/research prototype** using the existing Voice, Handwriting and Gait models with decision-level fusion.

> **Medical notice:** This application does not diagnose Parkinson's disease. A screening result is not a medical diagnosis and should not replace professional evaluation.

## What changed in V5

- Replaced the temporary Colab + `gradio.live` launch flow with a persistent web application entry point.
- Preserved the existing trained model artifacts and core feature extraction/fusion mathematics.
- Added Supabase email/password authentication.
- Added dynamic modality selection: upload controls appear only for selected modalities.
- Added user-friendly gait collection instructions.
- Added input validation and safe temporary-file handling.
- Replaced raw JSON output with a clean screening result card.
- Added responsive CSS, cards, shadows, gradients, loading states, toast messages and friendly errors.
- Added deployment configuration for a Docker-based ML deployment.

## Existing ML pipeline preserved

### Voice

The original V4 pipeline remains:

`audio → 54 acoustic features → saved scaler → saved Keras model → probability`

The saved model is `models/voice/voice_model_compatible_v2.keras` and the scaler is `models/voice/voice_scaler_combined.pkl`.

The model expects 54 features and the existing 5-second segmentation + averaging behavior is preserved.

### Handwriting

The original V4 pipeline remains:

`semicolon-separated TXT trajectory → 8 features → saved winsorization limits → saved scaler → saved classifier → probability`

Expected columns are:

`x; y; pressure; azimuth; altitude; timestamp; pen`

The eight trained features and their order are taken from the saved artifact rather than recreated manually.

### Gait

The original V4 pipeline remains:

`Sensor Logger ZIP → Accelerometer + Gyroscope + Orientation → timestamp alignment → 5-second windows → 63 features → saved Random Forest pipeline → probability`

The current input ZIP must contain these files at its root:

- `Accelerometer.csv`
- `Gyroscope.csv`
- `Orientation.csv`

The required columns are validated before inference. The current production UI intentionally keeps ZIP support because the existing trained gait pipeline expects these three files. A direct CSV is not silently converted into a different representation.

## Fusion

The existing decision-level fusion mathematics is preserved:

- **Average:** arithmetic mean of selected modality probabilities.
- **Maximum Probability:** maximum selected modality probability.
- **Reliability Weighted:** weighted average using the existing research weights:
  - Voice: `0.9164`
  - Handwriting: `0.8644`
  - Gait: `0.9953108396`

Missing modalities are excluded; they are never treated as zero.

Threshold: `0.50`.

## Gait data collection

Use Sensor Logger on Android:

https://play.google.com/store/apps/details?id=com.kelvin.sensorapp

Collect Accelerometer, Gyroscope and Orientation data according to your study protocol. Export the recordings and place the three required CSV files at the root of a ZIP. Upload that ZIP in the Gait section.

## Authentication

Supabase Auth is used for email/password authentication. Passwords are not stored by this application. Hosted Supabase projects can require email confirmation by default; configure the Supabase Auth email confirmation and redirect settings for your deployment URL.

Required secrets:

```text
SUPABASE_URL=
SUPABASE_ANON_KEY=
```

Never commit real credentials. The anon key is intended for client-facing Supabase use, but Row Level Security and Auth configuration should still be used correctly. Never put a Supabase service-role key in this application.

## Local setup

Recommended Python version: 3.13 for the current TensorFlow wheel.

```bash
git clone <your-repository>
cd NeuroFusion-PD

python -m venv .venv

# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Set the environment variables from `.env.example` in your local shell or use a local secrets file that is excluded by `.gitignore`.

Then:

```bash
streamlit run app.py
```

## Deployment recommendation

The application is designed for a Docker-capable ML host. Hugging Face Spaces Docker is a good fit because the project is an ML inference application, the model files are small enough for repository deployment, and the Space provides a stable `hf.space` application URL. Free/paid hardware and sleep behavior depend on the current Spaces plan; the URL itself remains the same when the Space sleeps and wakes.

A Docker Space should expose port `7860`.

## Project structure

```text
NeuroFusion-PD/
├── app.py
├── inference/
│   ├── __init__.py
│   └── pipeline.py
├── fusion/
│   ├── neurofusion_engine.py
│   └── fusion_functions.py
├── models/
│   ├── voice/
│   ├── handwriting/
│   └── gait/
├── datasets/
├── metadata/
├── .streamlit/config.toml
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

The original `neurofusion_app.py` is retained as a V4 reference/backup while `app.py` is the deployment entry point.

## Privacy

Uploaded modality files are written to temporary server files only for inference and are deleted after processing. The application does not intentionally persist voice, handwriting or gait files or place them in a database.

Do not enable persistent storage for raw health data unless the research protocol, consent, retention policy and applicable requirements explicitly call for it.

## Testing checklist

- [ ] Register with valid email/password
- [ ] Reject invalid email
- [ ] Reject weak password
- [ ] Reject password mismatch
- [ ] Reject duplicate registration
- [ ] Log in with valid credentials
- [ ] Reject wrong password
- [ ] Log out
- [ ] No modality selected
- [ ] Voice only
- [ ] Handwriting only
- [ ] Gait only
- [ ] Voice + Handwriting
- [ ] Voice + Gait
- [ ] Handwriting + Gait
- [ ] All three modalities
- [ ] Average fusion
- [ ] Maximum Probability fusion
- [ ] Reliability Weighted fusion
- [ ] Invalid voice
- [ ] Invalid handwriting
- [ ] Invalid gait ZIP
- [ ] Missing gait file
- [ ] Missing gait columns
- [ ] Empty uploads
- [ ] No raw JSON in normal UI
- [ ] Temporary uploads deleted after inference

## Research limitations

The supplied project metadata states that the three modality datasets are not participant-paired and therefore do not establish patient-level paired multimodal performance. Reliability weights are provisional. This deployment preserves those research constraints rather than presenting the fusion result as a clinical diagnostic model.
