import addnn
import addnn.model
import addnn.util.entropy
import grpc
import logging
import pickle
import threading
import torch
from addnn.node.proto import node_pb2
from addnn.node.proto.neural_network_pb2 import InferRequest
from addnn.node.proto.neural_network_pb2_grpc import NeuralNetworkStub
from typing import Any, List, Optional, Union

logger = logging.getLogger(__name__)


class Model:
    """
    Represents a node's local portion of a DNN model.
    """
    def __init__(self, layers: List[addnn.model.Layer], remote_layer: Optional[node_pb2.RemoteLayer],
                 start_layer_index: int) -> None:
        self._layers = layers
        self._remote_layer = remote_layer
        self._start_layer_index = start_layer_index

    @property
    def layers(self) -> List[addnn.model.Layer]:
        """
        Returns the model's layers.
        """
        return self._layers

    @property
    def remote_layer(self) -> Optional[node_pb2.RemoteLayer]:
        """
        Returns the entrypoint to the next layer that is hosted at a different node.
        """
        return self._remote_layer

    @property
    def start_layer_index(self) -> int:
        """
        Returns the index of the node's bottommost layer that globally identifies its position in the ADDNN.
        """
        return self._start_layer_index


class NeuralNetwork:
    """
    Represents a neural network that is hosted by a compute node.
    """
    def __init__(self) -> None:
        self._model: Optional[Model] = None
        self._lock = threading.Lock()

    @property
    def model(self) -> Optional[Model]:
        """
        Returns neural network's model.
        """
        with self._lock:
            return self._model

    @model.setter
    def model(self, model: Optional[Model]) -> None:
        """
        Update the node's local portion of the neural network's model.
        """
        with self._lock:
            self._model = model

    def infer(self, x: Union[bytes, torch.Tensor]) -> int:
        """
        Infer a classifiction for the given input tensor.
        """

        with self._lock:
            if self._model is None:
                raise Exception("node does not host a DNN model")

            if len(self._model.layers) == 0 and self._model.remote_layer is None:
                raise Exception("node does not host any DNN layers")

            if len(self._model.layers) == 0 and self._model.remote_layer is not None:
                if not isinstance(x, bytes):
                    raise Exception("expected serialized tensor on proxy node")

                # if the model does not have any local layers then, suppose that x is a serialized tensor (i.e. a bytes
                # instance), and send the serialized tensor to the remote layer without deserializing it
                remote_classification = _invoke_remote_layer(self._model.remote_layer, x)
                return remote_classification

            if isinstance(x, torch.Tensor):
                deserialized_x = x
            else:
                deserialized_x = pickle.loads(x)

            logger.debug("classify input of shape {}".format(deserialized_x.shape))

            with torch.no_grad():
                prediction = None
                local_layer_index = 0
                for layer in self._model.layers:
                    if layer.main_branch is not None:
                        deserialized_x = layer.main_branch(deserialized_x)

                    if layer.exit_branch is not None:
                        batch_predictions = layer.exit_branch.classifier(deserialized_x)
                        sample_prediction = batch_predictions[0]  # assumes batches of size 1
                        exit_entropy = addnn.util.entropy.normalized_entropy(sample_prediction)
                        logger.debug("normalized entropy at exit classifier is {}".format(exit_entropy))
                        if exit_entropy <= layer.exit_branch.confidence_threshold:
                            logger.debug("normalized entropy below threshold {}, take exit".format(
                                layer.exit_branch.confidence_threshold))
                            layer.exit_branch.number_of_exited_samples = layer.exit_branch.number_of_exited_samples + 1
                            classification = sample_prediction.argmax()
                            return int(classification.item())

                    local_layer_index += 1

                # if the model is connected to a further model that contains the upper layers of the DNN then pass on the
                # intermediate result
                if self._model.remote_layer is not None:
                    pickled_x = pickle.dumps(deserialized_x)
                    classification = _invoke_remote_layer(self._model.remote_layer, pickled_x)
                    return classification
                else:
                    raise Exception("DNN has no exit")


def _invoke_remote_layer(remote_layer: node_pb2.RemoteLayer, x: bytes) -> int:
    logger.debug("invoke next layer at {}:{}".format(remote_layer.host, remote_layer.port))
    with addnn.grpc.create_channel(remote_layer.host, remote_layer.port) as channel:
        neural_network_stub = NeuralNetworkStub(channel)
        infer_request = InferRequest()
        infer_request.tensor = x
        infer_response = neural_network_stub.Infer(infer_request)  # type: ignore
        return infer_response.classification
