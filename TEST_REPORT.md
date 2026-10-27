# NeuroFusion-PD V5 Test Report

Date: 2026-10-27

## Automated/static checks completed

| Test | Result | Notes |
|---|---|---|
| Original Python syntax | PASS | Original V4 Python files compile successfully. |
| V5 Python syntax | PASS | `app.py` and `inference/pipeline.py` compile successfully. |
| Voice artifact inspection | PASS | `.keras` archive opens; input shape is 54 features. |
| Handwriting artifact load | PASS | Saved model/scaler/features/winsor limits load. |
| Gait artifact load | PASS | Saved sklearn pipeline loads and exposes 63 features. |
| Fusion average | PASS | Uses only selected modalities. |
| Fusion maximum | PASS | Uses only selected modalities. |
| Fusion reliability weighted | PASS | Uses stored research weights and renormalizes over selected modalities. |
| Missing modality not treated as zero | PASS | Fusion receives only available probability keys. |
| Invalid handwriting | PASS | Too-short/malformed trajectory rejected before inference. |
| Valid handwriting feature extraction | PASS | Eight expected features produced. |
| Invalid gait ZIP | PASS | Missing required files rejected. |
| Missing gait columns | PASS | Required sensor schema is checked. |
| Synthetic valid gait ZIP | PASS | 5+ second three-file ZIP reaches saved Random Forest and returns probability. |
| Temporary upload deletion | PASS by code inspection | `finally` removes temporary files after inference. |
| Raw JSON display | PASS by code inspection | V5 UI displays only result, probability and recommendations. |

## Tests that require the final runtime environment

These need to be run after installing the pinned deployment dependencies and configuring Supabase:

1. Full TensorFlow voice inference with a known valid project voice sample.
2. Full end-to-end Voice only.
3. Handwriting only in the deployed Streamlit runtime.
4. Gait only with a real Sensor Logger ZIP.
5. All seven modality combinations.
6. All three fusion methods for each applicable combination.
7. Supabase registration.
8. Supabase duplicate-registration handling.
9. Supabase login and wrong-password handling.
10. Logout/session isolation.
11. Browser microphone recording.
12. Deployed public URL from a fresh/incognito browser.

## Important environment limitation during this inspection

The analysis sandbox does not have TensorFlow or Streamlit installed and has no outbound package-install network access. Therefore the full web server and Keras voice model could not be executed here. This does not indicate a failure of the supplied model. The deployment requirements pin TensorFlow 2.21.0 and scikit-learn 1.6.1 for the target runtime.

## Expected production test matrix

| Test | Expected |
|---|---|
| No modality | Informational validation message; no upload controls |
| Voice only | Voice upload/record control only |
| Handwriting only | TXT control only |
| Gait only | Gait instructions + ZIP control only |
| Voice + Handwriting | Voice + TXT controls only |
| Voice + Gait | Voice + gait controls only |
| Handwriting + Gait | TXT + gait controls only |
| All three | All three controls |
| Missing selected input | Analyze remains disabled |
| Invalid voice | Friendly error; no prediction |
| Invalid handwriting | Friendly error; no prediction |
| Invalid gait | Friendly error; no prediction |
| Average | Arithmetic mean of selected probabilities |
| Maximum Probability | Maximum selected probability |
| Reliability Weighted | Weighted mean using available selected modalities only |
| Positive result | Signal detected + 2-decimal probability + clinical recommendation + disclaimer |
| Negative result | No significant signal + 2-decimal probability + negative-result caveat + disclaimer |
| Register | Account created / confirmation message |
| Duplicate register | Friendly Supabase error |
| Login | Dashboard |
| Wrong password | Friendly error |
| Logout | Login screen |
