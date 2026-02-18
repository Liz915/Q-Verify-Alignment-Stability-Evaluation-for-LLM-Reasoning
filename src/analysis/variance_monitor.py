import numpy as np

class VarianceMonitor:
    """
    Tracks the variance and entropy of the agent's decision distribution.
    High entropy typically correlates with hallucination or uncertainty.
    """
    def __init__(self):
        self.variances = []

    def compute_entropy(self, probabilities):
        """
        Calculates Shannon Entropy: H(X) = -sum(p * log(p))
        """
        probs = np.array(probabilities)
        # Clip to avoid log(0)
        probs = np.clip(probs, 1e-9, 1.0)
        return -np.sum(probs * np.log(probs))

    def log_step(self, thought_vector):
        """
        Logs the variance of the thought vector components.
        """
        var = np.var(thought_vector)
        self.variances.append(var)
        return var