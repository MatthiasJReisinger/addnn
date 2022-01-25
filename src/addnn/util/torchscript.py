import addnn
import addnn.model
import torch
from pathlib import Path
from typing import List, Iterable, Union


def read_model_from_preprocessed_script(filename: str, input_shape: Union[str, Iterable[int]]) -> addnn.model.Model:
    layers = _read_script_layers(filename)
    model = addnn.model.Model(layers)

    if isinstance(input_shape, str):
        # pass a random input through the model to compute the input size of each layer
        dimensions = [int(dimension) for dimension in input_shape.split(",")]
    else:
        dimensions = list(input_shape)

    x = torch.rand(*dimensions)
    x = x.unsqueeze(0)
    model(x)

    return model


def _read_script_layers(filename: str) -> List[addnn.model.Layer]:
    script_module = torch.jit.load(filename)
    graph = script_module.inlined_graph
    nodes = list(graph.nodes())
    number_of_nodes = len(nodes)

    script_layers: List[torch.jit.ScriptModule] = []
    for node_index in range(number_of_nodes):
        node_script_path = "{}_node_{}".format(filename, node_index)
        if Path(node_script_path).exists():
            node_script = torch.jit.load(node_script_path)
            script_layers.append(node_script)
    layers = [addnn.model.Layer(script_layer) for script_layer in script_layers[0:-1]]
    layers.append(addnn.model.Layer(None, addnn.model.Exit(script_layers[-1], 1.0)))
    return layers
