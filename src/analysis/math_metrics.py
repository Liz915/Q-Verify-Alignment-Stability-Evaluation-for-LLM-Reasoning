import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_ftle(embeddings):
    """
    [Research Grade] Calculates Lyapunov-inspired Divergence on Real Embeddings.
    
    Metric: Local Expansion Rate of the Thought Trajectory.
    lambda_t = log( ||h_{t+1} - h_t|| / ||h_t - h_{t-1}|| )
    
    If lambda > 0: The thought process is accelerating/diverging (Chaos/Hallucination).
    If lambda < 0: The thought process is braking/converging (Stable Reasoning).
    """
    if len(embeddings) < 3:
        return 0.0
    
    embeddings = np.array(embeddings)
    
    # 1. Compute displacements (Velocity vectors in Latent Space)
    displacements = np.diff(embeddings, axis=0) # Shape: (T-1, Dim)
    
    # 2. Compute norms (Speed of thought)
    norms = np.linalg.norm(displacements, axis=1)
    
    # 3. Compute Log-Ratio of consecutive speeds
    # Add epsilon to avoid division by zero
    ratios = (norms[1:] + 1e-9) / (norms[:-1] + 1e-9)
    lambdas = np.log(ratios)
    
    return np.mean(lambdas)