import torch


def _apply_noise(x, noise_std, noise_type):
    if noise_type == "gaussian":
        return x + torch.randn_like(x) * noise_std
    if noise_type == "uniform":
        return x + (torch.rand_like(x) * 2.0 - 1.0) * noise_std
    if noise_type == "signflip":
        p = max(0.0, min(1.0, float(noise_std)))
        flips = torch.rand_like(x) < p
        return torch.where(flips, -x, x)
    if noise_type == "dropout":
        p = max(0.0, min(0.95, float(noise_std)))
        if p <= 0:
            return x
        keep = (torch.rand_like(x) >= p).to(x.dtype)
        return (x * keep) / (1.0 - p)
    raise ValueError(f"Unsupported noise_type: {noise_type}")


def get_noise_hook(noise_std, noise_type="gaussian"):
    def hook(module, input, output):
        if isinstance(output, tuple):
            return (_apply_noise(output[0], noise_std, noise_type), *output[1:])
        return _apply_noise(output, noise_std, noise_type)

    return hook


def get_target_layer_robust(model, layer_idx=12):
    """
    Robustly find the target layer for both Base and Adapter models.
    """
    candidates = []

    # Strategy 1: Unwrapped model (or base_model access)
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base = model.base_model.model
        if hasattr(base, "model") and hasattr(base.model, "layers"):
            candidates.append(base.model.layers)
        if hasattr(base, "layers"):
            candidates.append(base.layers)

    # Strategy 2: Standard HF structure
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        candidates.append(model.model.layers)

    # Strategy 3: Direct access
    if hasattr(model, "layers"):
        candidates.append(model.layers)

    for layers in candidates:
        if len(layers) > layer_idx:
            return layers[layer_idx]

    raise AttributeError(f"Could not locate layer {layer_idx} in model type: {type(model)}")


class LayerNoiseController:
    def __init__(self, layer):
        self.layer = layer
        self.handle = None

    def set_noise(self, std, noise_type="gaussian"):
        self.clear()
        if std > 0.0:
            self.handle = self.layer.register_forward_hook(get_noise_hook(std, noise_type=noise_type))

    def clear(self):
        if self.handle:
            self.handle.remove()
            self.handle = None
