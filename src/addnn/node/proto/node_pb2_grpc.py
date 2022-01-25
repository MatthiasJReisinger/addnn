# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from addnn.node.proto import node_pb2 as addnn_dot_node_dot_proto_dot_node__pb2
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


class NodeStub(object):
    """Provides means to configure and monitor an ADDNN compute node.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.DeployModel = channel.stream_unary(
                '/addnn.grpc.node.Node/DeployModel',
                request_serializer=addnn_dot_node_dot_proto_dot_node__pb2.LocalLayer.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )
        self.DeleteModel = channel.unary_unary(
                '/addnn.grpc.node.Node/DeleteModel',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )
        self.ActivateLayers = channel.unary_unary(
                '/addnn.grpc.node.Node/ActivateLayers',
                request_serializer=addnn_dot_node_dot_proto_dot_node__pb2.ActivateLayersRequest.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )
        self.DeactivateLayers = channel.unary_unary(
                '/addnn.grpc.node.Node/DeactivateLayers',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )
        self.ReadNodeState = channel.unary_unary(
                '/addnn.grpc.node.Node/ReadNodeState',
                request_serializer=addnn_dot_node_dot_proto_dot_node__pb2.ReadNodeStateRequest.SerializeToString,
                response_deserializer=addnn_dot_node_dot_proto_dot_node__pb2.ReadNodeStateResponse.FromString,
                )
        self.UpdateResourceState = channel.unary_unary(
                '/addnn.grpc.node.Node/UpdateResourceState',
                request_serializer=addnn_dot_node_dot_proto_dot_node__pb2.UpdateResourceStateRequest.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )
        self.ReadNeighbourNodes = channel.unary_unary(
                '/addnn.grpc.node.Node/ReadNeighbourNodes',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=addnn_dot_node_dot_proto_dot_node__pb2.ReadNeighbourNodesResponse.FromString,
                )


class NodeServicer(object):
    """Provides means to configure and monitor an ADDNN compute node.
    """

    def DeployModel(self, request_iterator, context):
        """Deploy parts of a DNN model to this compute node.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteModel(self, request, context):
        """Delete the deployed DNN model parts from this compute node.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ActivateLayers(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeactivateLayers(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ReadNodeState(self, request, context):
        """Read the current state of this compute node.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateResourceState(self, request, context):
        """Update the resource state of this compute node.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ReadNeighbourNodes(self, request, context):
        """Read the current neighbours of this compute node.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_NodeServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'DeployModel': grpc.stream_unary_rpc_method_handler(
                    servicer.DeployModel,
                    request_deserializer=addnn_dot_node_dot_proto_dot_node__pb2.LocalLayer.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
            'DeleteModel': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteModel,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
            'ActivateLayers': grpc.unary_unary_rpc_method_handler(
                    servicer.ActivateLayers,
                    request_deserializer=addnn_dot_node_dot_proto_dot_node__pb2.ActivateLayersRequest.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
            'DeactivateLayers': grpc.unary_unary_rpc_method_handler(
                    servicer.DeactivateLayers,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
            'ReadNodeState': grpc.unary_unary_rpc_method_handler(
                    servicer.ReadNodeState,
                    request_deserializer=addnn_dot_node_dot_proto_dot_node__pb2.ReadNodeStateRequest.FromString,
                    response_serializer=addnn_dot_node_dot_proto_dot_node__pb2.ReadNodeStateResponse.SerializeToString,
            ),
            'UpdateResourceState': grpc.unary_unary_rpc_method_handler(
                    servicer.UpdateResourceState,
                    request_deserializer=addnn_dot_node_dot_proto_dot_node__pb2.UpdateResourceStateRequest.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
            'ReadNeighbourNodes': grpc.unary_unary_rpc_method_handler(
                    servicer.ReadNeighbourNodes,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=addnn_dot_node_dot_proto_dot_node__pb2.ReadNeighbourNodesResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'addnn.grpc.node.Node', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Node(object):
    """Provides means to configure and monitor an ADDNN compute node.
    """

    @staticmethod
    def DeployModel(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/addnn.grpc.node.Node/DeployModel',
            addnn_dot_node_dot_proto_dot_node__pb2.LocalLayer.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/addnn.grpc.node.Node/DeleteModel',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ActivateLayers(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/addnn.grpc.node.Node/ActivateLayers',
            addnn_dot_node_dot_proto_dot_node__pb2.ActivateLayersRequest.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeactivateLayers(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/addnn.grpc.node.Node/DeactivateLayers',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ReadNodeState(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/addnn.grpc.node.Node/ReadNodeState',
            addnn_dot_node_dot_proto_dot_node__pb2.ReadNodeStateRequest.SerializeToString,
            addnn_dot_node_dot_proto_dot_node__pb2.ReadNodeStateResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpdateResourceState(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/addnn.grpc.node.Node/UpdateResourceState',
            addnn_dot_node_dot_proto_dot_node__pb2.UpdateResourceStateRequest.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ReadNeighbourNodes(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/addnn.grpc.node.Node/ReadNeighbourNodes',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            addnn_dot_node_dot_proto_dot_node__pb2.ReadNeighbourNodesResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
