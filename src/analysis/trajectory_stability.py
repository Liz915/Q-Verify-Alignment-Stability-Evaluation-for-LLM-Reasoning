import numpy as np

def calculate_lyapunov_exponent(variance_history):
    """
    Calculates the contraction rate (Lyapunov exponent approximation).
    Negative values indicate convergence (stability).
    """
    if len(variance_history) < 2:
        return 0.0
    
    # Calculate log-ratio of variances between steps
    log_divergences = []
    for i in range(len(variance_history) - 1):
        v_t = variance_history[i] + 1e-9
        v_t1 = variance_history[i+1] + 1e-9
        lambda_i = np.log(v_t1 / v_t)
        log_divergences.append(lambda_i)
        
    return np.mean(log_divergences)
