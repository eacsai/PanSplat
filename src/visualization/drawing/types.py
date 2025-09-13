import torch
from einops import repeat
from torch import Tensor


def sanitize_vector(
    vector,
    dim,
    device,
):
    if isinstance(vector, Tensor):
        vector = vector.type(torch.float32).to(device)
    else:
        vector = torch.tensor(vector, dtype=torch.float32, device=device)
    while vector.ndim < 2:
        vector = vector[None]
    if vector.shape[-1] == 1:
        vector = repeat(vector, "... () -> ... c", c=dim)

    return vector


def sanitize_scalar(scalar, device):
    if isinstance(scalar, Tensor):
        scalar = scalar.type(torch.float32).to(device)
    else:
        scalar = torch.tensor(scalar, dtype=torch.float32, device=device)
    while scalar.ndim < 1:
        scalar = scalar[None]
    return scalar

def sanitize_pair(pair, device):
    if isinstance(pair, Tensor):
        pair = pair.type(torch.float32).to(device)
    else:
        pair = torch.tensor(pair, dtype=torch.float32, device=device)
    return pair
