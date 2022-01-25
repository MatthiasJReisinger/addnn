import addnn
import addnn.model
import copy
import fvcore.nn
import io
import pypapi
import pickle
import ptflops
import torch
from addnn.util.serialization import serialize_module
from typing import List, Optional


class LayerProfile:
    """
    Specifies the resource requirements of a certain ADDNN layer.
    """
    def __init__(self, flops: int, in_memory_size: int, storage_size: int, marshalled_input_size: int, has_exit: bool,
                 number_of_exited_samples: int, exit_probability: float, execution_probability: float,
                 operators: List[str]):
        self._flops = flops
        self._in_memory_size = in_memory_size
        self._storage_size = storage_size
        self._marshalled_input_size = marshalled_input_size
        self._has_exit = has_exit
        self._number_of_exited_samples = number_of_exited_samples
        self._exit_probability = exit_probability
        self._execution_probability = execution_probability
        self._operators = operators

    @property
    def flops(self) -> int:
        """
        The number of floating point operations that are performed by the layer when processing an input tensor.
        """
        return self._flops

    @property
    def in_memory_size(self) -> int:
        """
        The in-memory size (in bytes) of the layer during inference.
        """
        return self._in_memory_size

    @property
    def storage_size(self) -> int:
        """
        The size of the layer (in bytes) when it is saved to disk.
        """
        return self._storage_size

    @property
    def marshalled_input_size(self) -> int:
        """
        The size of the layer's input tensor (in bytes) when for transfer between nodes.
        """
        return self._marshalled_input_size

    @property
    def has_exit(self) -> bool:
        """
        Determines whether this layer has an attached exit branch.
        """
        return self._has_exit

    @property
    def number_of_exited_samples(self) -> int:
        """
        The number of samples that exited at this layer.
        """
        return self._number_of_exited_samples

    @property
    def exit_probability(self) -> float:
        """
        The probability that the exit at this layer is taken.
        """
        return self._exit_probability

    @property
    def execution_probability(self) -> float:
        """
        The probability that this layer is executed.
        """
        return self._execution_probability

    @property
    def operators(self) -> List[str]:
        """
        The PyTorch operators that are used in this layer.
        """
        return self._operators


def get_layer_profiles(layers: List[addnn.model.Layer]) -> List[LayerProfile]:
    exit_probabilities = _get_exit_probabilities(layers)
    execution_probabilities = _get_execution_probabilities(exit_probabilities)

    layer_profiles: List[LayerProfile] = []
    for layer_index in range(len(layers)):
        layer_profile = _get_layer_profile(layers[layer_index], exit_probabilities[layer_index],
                                           execution_probabilities[layer_index])
        layer_profiles.append(layer_profile)
    return layer_profiles


def _get_exit_probabilities(layers: List[addnn.model.Layer]) -> List[float]:
    total_number_of_classified_samples = sum([layer.number_of_exited_samples for layer in layers])

    exit_probabilities = []

    if total_number_of_classified_samples > 0:
        for layer in layers:
            exit_probability = float(layer.number_of_exited_samples) / total_number_of_classified_samples
            exit_probabilities.append(exit_probability)
    else:
        # if no samples have been classified yet, just assume that all samples exit at the final classifier
        exit_probabilities = [0.0] * len(layers)
        exit_probabilities[-1] = 1.0

    return exit_probabilities


def _get_execution_probabilities(exit_probabilities: List[float]) -> List[float]:
    execution_probabilities = [0.0] * len(exit_probabilities)
    execution_probabilities[-1] = exit_probabilities[-1]
    for layer_index in range(len(exit_probabilities) - 2, -1, -1):
        execution_probabilities[layer_index] = exit_probabilities[layer_index] + execution_probabilities[layer_index +
                                                                                                         1]
    return execution_probabilities


def _get_layer_profile(layer: addnn.model.Layer, exit_probability: float, execution_probability: float) -> LayerProfile:
    flops = _estimate_flops(layer)
    in_memory_size = _estimate_in_memory_size(layer)
    storage_size = _estimate_storage_size(layer)
    marshalled_input_size = _estimate_marshalled_input_size(layer)
    has_exit = layer.exit_branch is not None
    operators = _get_operators(layer)
    layer_profile = LayerProfile(flops, in_memory_size, storage_size, marshalled_input_size, has_exit,
                                 layer.number_of_exited_samples, exit_probability, execution_probability, operators)
    return layer_profile


def _estimate_storage_size(layer: addnn.model.Layer) -> int:
    storage_size = 0

    if layer.main_branch is not None:
        pickled_model = serialize_module(layer.main_branch)
        storage_size += len(pickled_model)

    if layer.exit_branch is not None:
        pickled_model = serialize_module(layer.exit_branch.classifier)
        storage_size += len(pickled_model)

    return storage_size


def _estimate_flops(layer: addnn.model.Layer) -> int:
    if _is_torchscript_layer(layer):
        torchscript_flops = _profile_flops(layer)
        if torchscript_flops is not None:
            return torchscript_flops
        else:
            raise Exception("flops profiling is not possible on the current platform")

    if layer.input_size is None:
        raise Exception("cannot compute flops of layer with unknown input size")

    flops = 0

    input_tensor = torch.rand(layer.input_size)  # type: ignore
    input_tensor = input_tensor.unsqueeze(0)  # add batch dimension

    # compute the FLOPs for the layer's main branch
    if layer.main_branch is not None:
        analysis_inputs = (input_tensor, )
        main_branch_flops = fvcore.nn.FlopCountAnalysis(layer.main_branch, analysis_inputs).total()
        flops += main_branch_flops

    # compute the FLOPs for the layer's exit branch
    if layer.exit_branch is not None:
        if layer.main_branch is not None:
            main_branch_output = layer.main_branch(input_tensor)
            classifier_input = main_branch_output
        else:
            classifier_input = input_tensor

        analysis_inputs = (classifier_input, )
        exit_branch_flops = fvcore.nn.FlopCountAnalysis(layer.exit_branch, analysis_inputs).total()
        flops += exit_branch_flops

    return flops


def _is_torchscript_layer(layer: addnn.model.Layer) -> bool:
    return isinstance(layer.main_branch,
                      torch.jit.ScriptModule) or (layer.exit_branch is not None
                                                  and isinstance(layer.exit_branch.classifier, torch.jit.ScriptModule))


def _profile_flops(layer: addnn.model.Layer) -> Optional[int]:
    if layer.input_size is None:
        raise Exception("cannot profile flops of layer with unknown input size")

    input_tensor = torch.rand(layer.input_size)  # type: ignore
    input_tensor = input_tensor.unsqueeze(0)  # add batch dimension

    total_flops = None

    floating_point_events = [pypapi.events.PAPI_FP_OPS, pypapi.events.PAPI_SP_OPS, pypapi.events.PAPI_DP_OPS]
    for floating_point_event in floating_point_events:
        try:
            pypapi.papi_high.start_counters([floating_point_event])
            layer(input_tensor)
            flops = sum(pypapi.papi_high.stop_counters())
            if total_flops is None:
                total_flops = flops
            else:
                total_flops += flops
        except pypapi.exceptions.PapiNoEventError as e:
            # ignore error
            pass

    return total_flops


def _estimate_marshalled_input_size(layer: addnn.model.Layer) -> int:
    if layer.input_size is None:
        raise Exception("cannot compute marshalled input size of layer")

    layer_input = torch.rand(layer.input_size)
    layer_input = layer_input.unsqueeze(0)  # add batch dimension
    pickled_input = pickle.dumps(layer_input)
    marshalled_input_size = len(pickled_input)
    # TODO size of InferRequest?
    return marshalled_input_size


def _estimate_in_memory_size(layer: addnn.model.Layer) -> int:
    if layer.input_size is None:
        raise Exception("cannot compute in-memory size of layer with unknown input size")

    main_branch_in_memory_size = 0
    exit_classifier_input_size = layer.input_size
    if layer.main_branch is not None:
        main_branch_in_memory_size = _estimate_module_in_memory_size(layer.main_branch, layer.input_size)

        x = torch.rand(layer.input_size)
        x = x.unsqueeze(0)  # add batch dimension
        y = layer.main_branch(x)
        exit_classifier_input_size = y.shape[1:]

    exit_classifier_in_memory_size = 0
    if layer.exit_branch is not None:
        exit_classifier_in_memory_size = _estimate_module_in_memory_size(layer.exit_branch.classifier,
                                                                         exit_classifier_input_size)

    # TODO find leave modules in the module hierarchy
    # TODO find input size of each leaf module

    # TODO when used on GPU they're probably not loaded into the GPU memory at the same time?
    in_memory_size = main_branch_in_memory_size + exit_classifier_in_memory_size
    return in_memory_size


def _estimate_module_in_memory_size(module: torch.nn.Module, input_size: torch.Size) -> int:
    if isinstance(module, torch.jit.ScriptModule):
        return _estimate_script_module_in_memory_size(module, input_size)
    else:
        return _estimate_torch_module_in_memory_size(module, input_size)


def _estimate_script_module_in_memory_size(module: torch.jit.ScriptModule, input_size: torch.Size) -> int:
    input_tensor = torch.rand(input_size)
    input_tensor = input_tensor.unsqueeze(0)  # add batch dimension
    parameter_sizes = [_tensor_size_in_bytes(parameter) for parameter in module.parameters()]
    total_parameter_size = sum(parameter_sizes)
    output_tensor = module(input_tensor)
    return _tensor_size_in_bytes(input_tensor) + total_parameter_size + _tensor_size_in_bytes(output_tensor)


def _estimate_torch_module_in_memory_size(module: torch.nn.Module, input_size: torch.Size) -> int:
    # avoid changing the original module instance due to side-effects of the forward hook
    module = copy.deepcopy(module)

    # compute the size of module's (and all its sub-modules') parameters (i.e., the size of the weight matrix)
    parameter_sizes = [_tensor_size_in_bytes(parameter) for parameter in module.parameters()]
    total_parameter_size = sum(parameter_sizes)

    # register a hook to record the size of sub-module's output
    for sub_module in module.modules():
        sub_module.register_forward_hook(_record_output_size)

    # pass random input through the module to trigger forward-hooks recursively on all sub-modules
    input_tensor = torch.rand(input_size)
    input_tensor = input_tensor.unsqueeze(0)  # add batch dimension
    module(input_tensor)

    # sum up the size of all intermediate outputs as well as the size of the final output
    total_output_size = 0
    for sub_module in module.modules():
        total_output_size += sub_module.output_size_in_bytes  # type: ignore

    in_memory_size = _tensor_size_in_bytes(input_tensor) + total_parameter_size + total_output_size
    return in_memory_size


def _tensor_size_in_bytes(tensor: torch.Tensor) -> int:
    number_of_elements = int(torch.tensor(tensor.size()).prod().item())
    size_in_bytes = number_of_elements * int(_bitwidth(tensor.dtype) / 8)
    return size_in_bytes


def _bitwidth(dtype: torch.dtype) -> int:
    if dtype is torch.float16:
        return 16
    elif dtype is torch.float32:
        return 32
    elif dtype is torch.float64:
        return 64
    elif dtype is torch.bfloat16:
        return 16
    elif dtype is torch.complex32:
        return 32
    elif dtype is torch.complex64:
        return 64
    elif dtype is torch.complex128:
        return 128
    elif dtype is torch.uint8:
        return 8
    elif dtype is torch.int8:
        return 8
    elif dtype is torch.int16:
        return 16
    elif dtype is torch.int32:
        return 32
    elif dtype is torch.int64:
        return 64
    elif dtype is torch.bool:
        return 8
    else:
        raise Exception("unknown dtype {}".format(dtype))


# callback function for the forward-pre-hook
def _record_output_size(module: torch.nn.Module, inputs: torch.Tensor, outputs: List[torch.Tensor]) -> None:
    if len(outputs) != 1:
        raise Exception("module {} has {} outputs, expected single output of type torch.Tensor".format(
            type(module), len(outputs)))

    if type(outputs[0]) is not torch.Tensor:
        raise Exception("module {} output is of type {}, expected torch.Tensor".format(type(module), type(outputs[0])))

    output_tensor = outputs[0]
    module.output_size_in_bytes = _tensor_size_in_bytes(output_tensor)  # type: ignore


def _get_operators(layer: addnn.model.Layer) -> List[str]:
    operators: List[str] = []

    input_tensor = torch.rand(layer.input_size)  # type: ignore
    input_tensor = input_tensor.unsqueeze(0)  # add batch dimension

    if layer.main_branch is not None:
        if isinstance(layer.main_branch, torch.jit.ScriptModule):
            main_branch_trace = layer.main_branch
        else:
            main_branch_trace = torch.jit.trace(layer.main_branch, input_tensor)
            input_tensor = layer.main_branch(input_tensor)

        operators.extend(_get_operators_in_script_module(main_branch_trace))

    if layer.exit_branch is not None:
        if isinstance(layer.exit_branch, torch.jit.ScriptModule):
            exit_branch_trace = layer.exit_branch
        else:
            exit_branch_trace = torch.jit.trace(layer.exit_branch, input_tensor)

        operators.extend(_get_operators_in_script_module(exit_branch_trace))

    return operators


def _get_operators_in_script_module(layer: torch.jit.ScriptModule) -> List[str]:
    operators: List[str] = []
    graph = layer.inlined_graph

    for node in graph.nodes():
        schema = node.schema()

        if schema == "(no schema)":
            continue

        # the node's "schema" is a string representation that looks similar to a function signature, so ignore the part
        # that starts at the opening bracket to obtain the node's operator name
        operator = schema[:schema.find("(")]

        # prune a few "uninteresting" parts to make the operator more readable
        operator = operator.replace("aten::", "")
        operator = operator.replace(".Tensor", "")
        operator = operator.replace(".using_ints", "")
        operator = operator.strip("_")

        operators.append(operator)

    return operators
