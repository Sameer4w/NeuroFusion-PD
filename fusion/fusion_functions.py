import numpy as np

def fuse_probabilities(modalities, method="average", weights=None):
    """
    modalities:
        Dictionary such as:
        {
            "voice": 0.80,
            "handwriting": 0.70,
            "gait": 0.60
        }

    method:
        "average"
        "max"
        "reliability_weighted"
    """

    if not modalities:
        raise ValueError("At least one modality is required.")

    probabilities = list(modalities.values())

    if method == "average":
        final_probability = np.mean(probabilities)

    elif method == "max":
        final_probability = np.max(probabilities)

    elif method == "reliability_weighted":
        if weights is None:
            raise ValueError(
                "Weights are required for reliability-weighted fusion."
            )

        available_weights = {
            modality: weights[modality]
            for modality in modalities
        }

        weight_sum = sum(available_weights.values())

        if weight_sum <= 0:
            raise ValueError("Sum of available weights must be positive.")

        final_probability = sum(
            available_weights[m] * modalities[m]
            for m in modalities
        ) / weight_sum

    else:
        raise ValueError(
            "Method must be 'average', 'max', or "
            "'reliability_weighted'."
        )

    return {
        "modalities_used": list(modalities.keys()),
        "fusion_method": method,
        "final_probability": float(final_probability),
        "prediction": int(final_probability >= 0.50)
    }


def run_fusion(
    voice=None,
    handwriting=None,
    gait=None,
    method="average",
    weights=None
):
    """
    Decision-level fusion for NeuroFusion-PD.

    Each modality must be supplied as:
        {"probability": float, "prediction": 0 or 1}

    Missing modalities are ignored.
    """

    available = {}

    if voice is not None:
        available["voice"] = float(voice["probability"])

    if handwriting is not None:
        available["handwriting"] = float(
            handwriting["probability"]
        )

    if gait is not None:
        available["gait"] = float(
            gait["probability"]
        )

    if not available:
        raise ValueError(
            "At least one modality must be provided."
        )

    result = fuse_probabilities(
        available,
        method=method,
        weights=weights
    )

    configuration_names = {
        frozenset(["voice"]): "Voice",
        frozenset(["handwriting"]): "Handwriting",
        frozenset(["gait"]): "Gait",
        frozenset(["voice", "handwriting"]):
            "Voice + Handwriting",
        frozenset(["voice", "gait"]):
            "Voice + Gait",
        frozenset(["handwriting", "gait"]):
            "Handwriting + Gait",
        frozenset(["voice", "handwriting", "gait"]):
            "Voice + Handwriting + Gait"
    }

    result["configuration"] = configuration_names[
        frozenset(available.keys())
    ]

    # Put configuration first
    return {
        "configuration": result["configuration"],
        "modalities_used": result["modalities_used"],
        "fusion_method": result["fusion_method"],
        "final_probability": result["final_probability"],
        "prediction": result["prediction"]
    }


def explainable_fusion(
    voice=None,
    handwriting=None,
    gait=None,
    method="average",
    weights=None
):
    result = run_fusion(
        voice=voice,
        handwriting=handwriting,
        gait=gait,
        method=method,
        weights=weights
    )

    output = {
        "configuration": result["configuration"],
        "fusion_method": result["fusion_method"]
    }

    # Preserve modality-level probabilities
    if voice is not None:
        output["voice_probability"] = float(
            voice["probability"]
        )

    if handwriting is not None:
        output["handwriting_probability"] = float(
            handwriting["probability"]
        )

    if gait is not None:
        output["gait_probability"] = float(
            gait["probability"]
        )

    output["final_probability"] = result["final_probability"]
    output["prediction"] = result["prediction"]

    return output
