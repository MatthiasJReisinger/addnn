import grpc
import pickle
import time
from addnn.node.proto import neural_network_pb2, neural_network_pb2_grpc, node_pb2
from addnn.node.neural_network import NeuralNetwork


class NeuralNetworkServicer(neural_network_pb2_grpc.NeuralNetworkServicer):
    """
    Implementation of the NeuralNetwork service interface.
    """
    def __init__(self, neural_network: NeuralNetwork):
        super().__init__()
        self._neural_network = neural_network

    def Infer(self, request, context) -> neural_network_pb2.InferResponse:  # type: ignore
        response = neural_network_pb2.InferResponse()
        response.start_timestamp = time.time_ns()
        response.classification = self._neural_network.infer(request.tensor)
        response.end_timestamp = time.time_ns()
        return response
