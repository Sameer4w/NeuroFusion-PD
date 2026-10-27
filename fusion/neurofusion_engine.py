import numpy as np
class NeuroFusionEngine:

    def __init__(self, weights=None, threshold=0.50):
        self.weights = weights
        self.threshold = threshold

    def fuse(self, voice=None, handwriting=None, gait=None,
             method="average"):

        available = {}

        if voice is not None:
            available["voice"] = float(voice["probability"])

        if handwriting is not None:
            available["handwriting"] = float(handwriting["probability"])

        if gait is not None:
            available["gait"] = float(gait["probability"])

        if not available:
            raise ValueError("At least one modality must be provided.")

        probabilities = list(available.values())

        if method == "average":
            final_probability = np.mean(probabilities)

        elif method == "max":
            final_probability = np.max(probabilities)

        elif method == "reliability_weighted":

            if self.weights is None:
                raise ValueError(
                    "Weights are required for reliability-weighted fusion."
                )

            available_weights = {
                m: self.weights[m]
                for m in available
            }

            weight_sum = sum(available_weights.values())

            if weight_sum <= 0:
                raise ValueError(
                    "Available weights must have positive sum."
                )

            final_probability = sum(
                available_weights[m] * available[m]
                for m in available
            ) / weight_sum

        else:
            raise ValueError(
                "Method must be 'average', 'max', "
                "or 'reliability_weighted'."
            )

        configuration = " + ".join(
            m.capitalize()
            for m in ["voice", "handwriting", "gait"]
            if m in available
        )

        return {
            "configuration": configuration,
            "modalities_used": list(available.keys()),
            "fusion_method": method,
            "voice_probability": available.get("voice"),
            "handwriting_probability": available.get("handwriting"),
            "gait_probability": available.get("gait"),
            "final_probability": float(final_probability),
            "prediction": int(final_probability >= self.threshold)
        }
