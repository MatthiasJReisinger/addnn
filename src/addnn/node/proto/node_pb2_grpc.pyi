"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import abc
import addnn.node.proto.node_pb2
import google.protobuf.empty_pb2
import grpc
import typing

class NodeStub:
    """Provides means to configure and monitor an ADDNN compute node."""
    def __init__(self, channel: grpc.Channel) -> None: ...
    DeployModel: grpc.StreamUnaryMultiCallable[
        addnn.node.proto.node_pb2.LocalLayer,
        google.protobuf.empty_pb2.Empty] = ...
    """Deploy parts of a DNN model to this compute node."""

    DeleteModel: grpc.UnaryUnaryMultiCallable[
        google.protobuf.empty_pb2.Empty,
        google.protobuf.empty_pb2.Empty] = ...
    """Delete the deployed DNN model parts from this compute node."""

    ActivateLayers: grpc.UnaryUnaryMultiCallable[
        addnn.node.proto.node_pb2.ActivateLayersRequest,
        google.protobuf.empty_pb2.Empty] = ...

    DeactivateLayers: grpc.UnaryUnaryMultiCallable[
        google.protobuf.empty_pb2.Empty,
        google.protobuf.empty_pb2.Empty] = ...

    ReadNodeState: grpc.UnaryUnaryMultiCallable[
        addnn.node.proto.node_pb2.ReadNodeStateRequest,
        addnn.node.proto.node_pb2.ReadNodeStateResponse] = ...
    """Read the current state of this compute node."""

    UpdateResourceState: grpc.UnaryUnaryMultiCallable[
        addnn.node.proto.node_pb2.UpdateResourceStateRequest,
        google.protobuf.empty_pb2.Empty] = ...
    """Update the resource state of this compute node."""

    ReadNeighbourNodes: grpc.UnaryUnaryMultiCallable[
        google.protobuf.empty_pb2.Empty,
        addnn.node.proto.node_pb2.ReadNeighbourNodesResponse] = ...
    """Read the current neighbours of this compute node."""


class NodeServicer(metaclass=abc.ABCMeta):
    """Provides means to configure and monitor an ADDNN compute node."""
    @abc.abstractmethod
    def DeployModel(self,
        request: typing.Iterator[addnn.node.proto.node_pb2.LocalLayer],
        context: grpc.ServicerContext,
    ) -> google.protobuf.empty_pb2.Empty:
        """Deploy parts of a DNN model to this compute node."""
        pass

    @abc.abstractmethod
    def DeleteModel(self,
        request: google.protobuf.empty_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> google.protobuf.empty_pb2.Empty:
        """Delete the deployed DNN model parts from this compute node."""
        pass

    @abc.abstractmethod
    def ActivateLayers(self,
        request: addnn.node.proto.node_pb2.ActivateLayersRequest,
        context: grpc.ServicerContext,
    ) -> google.protobuf.empty_pb2.Empty: ...

    @abc.abstractmethod
    def DeactivateLayers(self,
        request: google.protobuf.empty_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> google.protobuf.empty_pb2.Empty: ...

    @abc.abstractmethod
    def ReadNodeState(self,
        request: addnn.node.proto.node_pb2.ReadNodeStateRequest,
        context: grpc.ServicerContext,
    ) -> addnn.node.proto.node_pb2.ReadNodeStateResponse:
        """Read the current state of this compute node."""
        pass

    @abc.abstractmethod
    def UpdateResourceState(self,
        request: addnn.node.proto.node_pb2.UpdateResourceStateRequest,
        context: grpc.ServicerContext,
    ) -> google.protobuf.empty_pb2.Empty:
        """Update the resource state of this compute node."""
        pass

    @abc.abstractmethod
    def ReadNeighbourNodes(self,
        request: google.protobuf.empty_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> addnn.node.proto.node_pb2.ReadNeighbourNodesResponse:
        """Read the current neighbours of this compute node."""
        pass


def add_NodeServicer_to_server(servicer: NodeServicer, server: grpc.Server) -> None: ...
