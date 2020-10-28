import torch
import numpy as np
import pandas as pd

def to_device(inputs, device='cuda:0'):
    if isinstance(inputs, (list, tuple)):
        outputs = []
        for i, item in enumerate(inputs):
            outputs += to_device(item, device)
    elif isinstance(inputs, (torch.Tensor, torch.nn.Module)):
        return inputs.to(device)
    elif isinstance(inputs, dict):
        outputs = {}
        for key in inputs.keys():
            outputs[key] = to_device(inputs[key], device)
    return outputs