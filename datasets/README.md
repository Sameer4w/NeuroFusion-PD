# NeuroFusion-PD Dataset Information

## Purpose

This directory contains small dataset/feature artifacts required for
Fusion-stage reproducibility and demonstration.

The complete raw modality datasets are maintained separately and are
not included in the Fusion package.

## Included Artifacts

### Voice

No raw Voice dataset is included here.

The trained Voice model and its StandardScaler are stored in:

models/voice/

The Voice model expects 54 acoustic features.

### Handwriting

The exact processed feature table used to reproduce the saved
Handwriting fusion artifact is stored in:

models/handwriting/v2_fixed_features_all_files.csv

It contains:
- 40 samples
- 10 columns
- 8 modeled features
- 0 missing values

The eight modeled features are:

1. speed_mean
2. zero_movement_ratio
3. total_path_length
4. speed_max
5. movement_mean
6. channel2_unique
7. direction_change_mean
8. speed_std

The raw Handwriting TXT dataset is maintained separately.

### Gait

A single verified 5-second Gait feature sample is included:

gait/X_gait_sample.csv

It contains:
- 1 sample
- 63 model input features
- 0 missing values
- feature order matching the serialized Random Forest model

The complete raw Gait dataset is maintained separately.

## Important Scientific Note

The Voice, Handwriting, and Gait datasets used during model
development do not provide participant-paired measurements across
all three modalities.

Therefore, the included Gait sample and other modality examples
must not be interpreted as a patient-level multimodal evaluation.

The Fusion package validates the technical integration of the
independently trained modality models and the decision-level fusion
engine.

## Raw Dataset Storage

Complete raw datasets should be stored separately in the research
project archive, together with their original source information,
documentation, and applicable data-use/licensing requirements.
