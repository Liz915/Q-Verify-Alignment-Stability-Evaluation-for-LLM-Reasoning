import numpy as np

class BoltzmannSampler:
    """
    Handles probabilistic sampling strategies for the Agent.
    """
    def __init__(self, temperature=1.0):
        self.temperature = max(temperature, 1e-5) # Prevent division by zero

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def sample_discrete(self, options, probs):
        """
        Samples an option based on probabilities scaled by temperature.
        """
        probs = np.array(probs)
        
        # Apply temperature sharpening
        # Higher temp -> Uniform distribution (Exploration)
        # Lower temp -> Argmax (Exploitation)
        logits = np.log(probs + 1e-9) / self.temperature
        scaled_probs = self.softmax(logits)
        
        return np.random.choice(options, p=scaled_probs)