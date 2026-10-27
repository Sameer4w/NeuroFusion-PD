# NeuroFusion-PD V5 Deployment Plan

## Recommended target

Hugging Face Spaces using the Docker SDK is the recommended production target for this particular ML application. It supports arbitrary Docker applications and provides a stable Space URL. The application uses CPU TensorFlow plus a small Keras model and a ~68 MB Random Forest artifact, so a CPU container is a reasonable starting point.

Current Hugging Face plan/hardware rules can change; verify the Space's available CPU hardware and billing/subscription requirements when creating it.

## 1. Create Supabase project

1. Create a Supabase project.
2. Open Authentication → Providers.
3. Enable Email/Password.
4. Decide whether email confirmation is required. Hosted Supabase projects enable confirmation by default.
5. Add the final application URL to the allowed redirect/site URL settings if confirmation links are used.
6. Copy the project URL and public anon key.

Do not use the Supabase service-role key in this application.

## 2. Local secrets

Use environment variables:

```text
SUPABASE_URL=...
SUPABASE_ANON_KEY=...
```

Never commit `.env` or secrets files.

## 3. Local run

```bash
python -m venv .venv

# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## 4. Docker test

```bash
docker build -t neurofusion-pd .
docker run --rm -p 7860:7860 \
  -e SUPABASE_URL="..." \
  -e SUPABASE_ANON_KEY="..." \
  neurofusion-pd
```

Then open `http://localhost:7860`.

## 5. Hugging Face Space

Create a new Space with Docker SDK and push this repository. The Dockerfile exposes port 7860 and starts Streamlit on that port.

Set these as Space secrets/variables:

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`

Do not put them into Git.

The resulting application will use the Space's stable URL rather than a temporary `gradio.live` link.

## 6. GitHub

Recommended repository layout:

```text
NeuroFusion-PD/
├── app.py
├── inference/
├── models/
├── fusion/
├── metadata/
├── datasets/
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

Because the repository contains trained artifacts, keep raw health datasets out of Git unless their licensing and research protocol explicitly permit redistribution.

## 7. Post-deployment checks

Open the final URL in an incognito window and test:

- New registration
- Email confirmation if enabled
- Login
- Logout
- Voice only
- Handwriting only
- Gait only
- Pair combinations
- All three modalities
- All fusion methods
- Invalid files
- Result card
- No raw JSON

## 8. URL behavior

The public Space URL is stable. The compute instance may sleep under free/limited plans, but users continue to use the same URL when the Space wakes again.
