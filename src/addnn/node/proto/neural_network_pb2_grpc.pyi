"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import abc
import addnn.node.proto.neural_network_pb2
import grpc

class NeuralNetworkStub:
    def __init__(self, channel: grpc.Channel) -> None: ...
    Infer: grpc.UnaryUnaryMultiCallable[
        addnn.node.proto.neural_network_pb2.InferRequest,
        addnn.node.proto.neural_network_pb2.InferResponse] = ...


class NeuralNetworkServicer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def Infer(self,
        request: addnn.node.proto.neural_network_pb2.InferRequest,
        context: grpc.ServicerContext,
    ) -> addnn.node.proto.neural_network_pb2.InferResponse: ...


def add_NeuralNetworkServicer_to_server(servicer: NeuralNetworkServicer, server: grpc.Server) -> None: ...
