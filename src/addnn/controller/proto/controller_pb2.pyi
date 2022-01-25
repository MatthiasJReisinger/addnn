"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import addnn.node.proto.node_state_pb2
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor = ...

class Node(google.protobuf.message.Message):
    """Represents a compute node of an ADDNN."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    HOST_FIELD_NUMBER: builtins.int
    PORT_FIELD_NUMBER: builtins.int
    TIER_FIELD_NUMBER: builtins.int
    IS_INPUT_FIELD_NUMBER: builtins.int
    STATE_FIELD_NUMBER: builtins.int
    IPERF_PORT_FIELD_NUMBER: builtins.int
    host: typing.Text = ...
    """The host name or IP of the node."""

    port: builtins.int = ...
    """The port at which the node's APIs can be reached."""

    tier: builtins.int = ...
    """The tier that the node is part of."""

    is_input: builtins.bool = ...
    """Whether the node is the input source for the neural network (i.e., the
    node represents an end device or sensor).
    """

    @property
    def state(self) -> addnn.node.proto.node_state_pb2.NodeState:
        """The current resource state of the node."""
        pass
    iperf_port: builtins.int = ...
    """The port at which the node's iperf server is exposed"""

    def __init__(self,
        *,
        host : typing.Text = ...,
        port : builtins.int = ...,
        tier : builtins.int = ...,
        is_input : builtins.bool = ...,
        state : typing.Optional[addnn.node.proto.node_state_pb2.NodeState] = ...,
        iperf_port : builtins.int = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"state",b"state"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"host",b"host",u"iperf_port",b"iperf_port",u"is_input",b"is_input",u"port",b"port",u"state",b"state",u"tier",b"tier"]) -> None: ...
global___Node = Node

class RegisterNodeRequest(google.protobuf.message.Message):
    """Request message of the RegisterNode method of the Controller service."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    NODE_FIELD_NUMBER: builtins.int
    @property
    def node(self) -> global___Node:
        """The node that should be registered."""
        pass
    def __init__(self,
        *,
        node : typing.Optional[global___Node] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"node",b"node"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"node",b"node"]) -> None: ...
global___RegisterNodeRequest = RegisterNodeRequest

class RegisterNodeResponse(google.protobuf.message.Message):
    """Response message of the RegisterNode method of the Controller service."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    UUID_FIELD_NUMBER: builtins.int
    uuid: typing.Text = ...
    """The UUID that the controller assigned to the new node."""

    def __init__(self,
        *,
        uuid : typing.Text = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"uuid",b"uuid"]) -> None: ...
global___RegisterNodeResponse = RegisterNodeResponse

class DeregisterNodeRequest(google.protobuf.message.Message):
    """Request message of the DeregisterNode method of the Controller service."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    UUID_FIELD_NUMBER: builtins.int
    uuid: typing.Text = ...
    """The UUID of the node that should be deregistered."""

    def __init__(self,
        *,
        uuid : typing.Text = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"uuid",b"uuid"]) -> None: ...
global___DeregisterNodeRequest = DeregisterNodeRequest

class UpdateNodeStateRequest(google.protobuf.message.Message):
    """Request message of the UpdateNodeState method of the Controller service."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    UUID_FIELD_NUMBER: builtins.int
    NODE_STATE_FIELD_NUMBER: builtins.int
    uuid: typing.Text = ...
    """The UUID of the node that should be deregistered."""

    @property
    def node_state(self) -> addnn.node.proto.node_state_pb2.NodeState:
        """The updated state of the compute node."""
        pass
    def __init__(self,
        *,
        uuid : typing.Text = ...,
        node_state : typing.Optional[addnn.node.proto.node_state_pb2.NodeState] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"node_state",b"node_state"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"node_state",b"node_state",u"uuid",b"uuid"]) -> None: ...
global___UpdateNodeStateRequest = UpdateNodeStateRequest

class RegisteredNode(google.protobuf.message.Message):
    """Represents a compute node that is registered at the controller."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    UUID_FIELD_NUMBER: builtins.int
    NODE_FIELD_NUMBER: builtins.int
    uuid: typing.Text = ...
    """The UUID that the controller assigned to the node."""

    @property
    def node(self) -> global___Node:
        """The node that is registered at the controller."""
        pass
    def __init__(self,
        *,
        uuid : typing.Text = ...,
        node : typing.Optional[global___Node] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"node",b"node"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"node",b"node",u"uuid",b"uuid"]) -> None: ...
global___RegisteredNode = RegisteredNode

class ListNodesResponse(google.protobuf.message.Message):
    """Request message of the ListNodes method of the Controller service."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    NODES_FIELD_NUMBER: builtins.int
    @property
    def nodes(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___RegisteredNode]:
        """The nodes that are currently registered at the controller."""
        pass
    def __init__(self,
        *,
        nodes : typing.Optional[typing.Iterable[global___RegisteredNode]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"nodes",b"nodes"]) -> None: ...
global___ListNodesResponse = ListNodesResponse
