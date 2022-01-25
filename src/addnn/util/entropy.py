import torch
import math


def normalized_entropy(x: torch.Tensor) -> float:
    x = x.clone()
    x = x.softmax(0)
    x = x * x.log()
    x = x.nan_to_num()
    entropy = -x.sum() / torch.tensor(float(x.size()[0])).log()
    return float(entropy.item())
