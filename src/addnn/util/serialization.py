import addnn
import addnn.model
import io
import pickle
import torch


def serialize_module(module: torch.nn.Module) -> bytes:
    if isinstance(module, torch.jit.ScriptModule):
        layer_bytes = io.BytesIO()
        torch.jit.save(module, layer_bytes)
        layer_bytes.seek(0)
        return layer_bytes.read()
    else:
        # TODO use torch.save?
        return pickle.dumps(module)


def deserialize_module(serialized_module: bytes, is_torchscript: bool) -> addnn.model.Layer:
    if is_torchscript:
        layer_bytes = io.BytesIO(serialized_module)
        layer_bytes.seek(0)
        return torch.jit.load(layer_bytes)
    else:
        # TODO use torch.load?
        return pickle.loads(serialized_module)
