# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: addnn/controller/proto/controller.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from addnn.node.proto import node_state_pb2 as addnn_dot_node_dot_proto_dot_node__state__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='addnn/controller/proto/controller.proto',
  package='addnn.grpc.controller',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\'addnn/controller/proto/controller.proto\x12\x15\x61\x64\x64nn.grpc.controller\x1a\x1bgoogle/protobuf/empty.proto\x1a!addnn/node/proto/node_state.proto\"\x87\x01\n\x04Node\x12\x0c\n\x04host\x18\x01 \x01(\t\x12\x0c\n\x04port\x18\x02 \x01(\r\x12\x0c\n\x04tier\x18\x03 \x01(\r\x12\x10\n\x08is_input\x18\x04 \x01(\x08\x12/\n\x05state\x18\x05 \x01(\x0b\x32 .addnn.grpc.node_state.NodeState\x12\x12\n\niperf_port\x18\x06 \x01(\r\"@\n\x13RegisterNodeRequest\x12)\n\x04node\x18\x01 \x01(\x0b\x32\x1b.addnn.grpc.controller.Node\"$\n\x14RegisterNodeResponse\x12\x0c\n\x04uuid\x18\x01 \x01(\t\"%\n\x15\x44\x65registerNodeRequest\x12\x0c\n\x04uuid\x18\x01 \x01(\t\"\\\n\x16UpdateNodeStateRequest\x12\x0c\n\x04uuid\x18\x01 \x01(\t\x12\x34\n\nnode_state\x18\x02 \x01(\x0b\x32 .addnn.grpc.node_state.NodeState\"I\n\x0eRegisteredNode\x12\x0c\n\x04uuid\x18\x01 \x01(\t\x12)\n\x04node\x18\x02 \x01(\x0b\x32\x1b.addnn.grpc.controller.Node\"I\n\x11ListNodesResponse\x12\x34\n\x05nodes\x18\x01 \x03(\x0b\x32%.addnn.grpc.controller.RegisteredNode2\xe9\x02\n\nController\x12T\n\x0cRegisterNode\x12*.addnn.grpc.controller.RegisterNodeRequest\x1a\x16.google.protobuf.Empty\"\x00\x12X\n\x0e\x44\x65registerNode\x12,.addnn.grpc.controller.DeregisterNodeRequest\x1a\x16.google.protobuf.Empty\"\x00\x12Z\n\x0fUpdateNodeState\x12-.addnn.grpc.controller.UpdateNodeStateRequest\x1a\x16.google.protobuf.Empty\"\x00\x12O\n\tListNodes\x12\x16.google.protobuf.Empty\x1a(.addnn.grpc.controller.ListNodesResponse\"\x00\x62\x06proto3'
  ,
  dependencies=[google_dot_protobuf_dot_empty__pb2.DESCRIPTOR,addnn_dot_node_dot_proto_dot_node__state__pb2.DESCRIPTOR,])




_NODE = _descriptor.Descriptor(
  name='Node',
  full_name='addnn.grpc.controller.Node',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='host', full_name='addnn.grpc.controller.Node.host', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='port', full_name='addnn.grpc.controller.Node.port', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='tier', full_name='addnn.grpc.controller.Node.tier', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='is_input', full_name='addnn.grpc.controller.Node.is_input', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='state', full_name='addnn.grpc.controller.Node.state', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='iperf_port', full_name='addnn.grpc.controller.Node.iperf_port', index=5,
      number=6, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=131,
  serialized_end=266,
)


_REGISTERNODEREQUEST = _descriptor.Descriptor(
  name='RegisterNodeRequest',
  full_name='addnn.grpc.controller.RegisterNodeRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='node', full_name='addnn.grpc.controller.RegisterNodeRequest.node', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=268,
  serialized_end=332,
)


_REGISTERNODERESPONSE = _descriptor.Descriptor(
  name='RegisterNodeResponse',
  full_name='addnn.grpc.controller.RegisterNodeResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='uuid', full_name='addnn.grpc.controller.RegisterNodeResponse.uuid', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=334,
  serialized_end=370,
)


_DEREGISTERNODEREQUEST = _descriptor.Descriptor(
  name='DeregisterNodeRequest',
  full_name='addnn.grpc.controller.DeregisterNodeRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='uuid', full_name='addnn.grpc.controller.DeregisterNodeRequest.uuid', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=372,
  serialized_end=409,
)


_UPDATENODESTATEREQUEST = _descriptor.Descriptor(
  name='UpdateNodeStateRequest',
  full_name='addnn.grpc.controller.UpdateNodeStateRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='uuid', full_name='addnn.grpc.controller.UpdateNodeStateRequest.uuid', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='node_state', full_name='addnn.grpc.controller.UpdateNodeStateRequest.node_state', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=411,
  serialized_end=503,
)


_REGISTEREDNODE = _descriptor.Descriptor(
  name='RegisteredNode',
  full_name='addnn.grpc.controller.RegisteredNode',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='uuid', full_name='addnn.grpc.controller.RegisteredNode.uuid', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='node', full_name='addnn.grpc.controller.RegisteredNode.node', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=505,
  serialized_end=578,
)


_LISTNODESRESPONSE = _descriptor.Descriptor(
  name='ListNodesResponse',
  full_name='addnn.grpc.controller.ListNodesResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='nodes', full_name='addnn.grpc.controller.ListNodesResponse.nodes', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=580,
  serialized_end=653,
)

_NODE.fields_by_name['state'].message_type = addnn_dot_node_dot_proto_dot_node__state__pb2._NODESTATE
_REGISTERNODEREQUEST.fields_by_name['node'].message_type = _NODE
_UPDATENODESTATEREQUEST.fields_by_name['node_state'].message_type = addnn_dot_node_dot_proto_dot_node__state__pb2._NODESTATE
_REGISTEREDNODE.fields_by_name['node'].message_type = _NODE
_LISTNODESRESPONSE.fields_by_name['nodes'].message_type = _REGISTEREDNODE
DESCRIPTOR.message_types_by_name['Node'] = _NODE
DESCRIPTOR.message_types_by_name['RegisterNodeRequest'] = _REGISTERNODEREQUEST
DESCRIPTOR.message_types_by_name['RegisterNodeResponse'] = _REGISTERNODERESPONSE
DESCRIPTOR.message_types_by_name['DeregisterNodeRequest'] = _DEREGISTERNODEREQUEST
DESCRIPTOR.message_types_by_name['UpdateNodeStateRequest'] = _UPDATENODESTATEREQUEST
DESCRIPTOR.message_types_by_name['RegisteredNode'] = _REGISTEREDNODE
DESCRIPTOR.message_types_by_name['ListNodesResponse'] = _LISTNODESRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Node = _reflection.GeneratedProtocolMessageType('Node', (_message.Message,), {
  'DESCRIPTOR' : _NODE,
  '__module__' : 'addnn.controller.proto.controller_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.controller.Node)
  })
_sym_db.RegisterMessage(Node)

RegisterNodeRequest = _reflection.GeneratedProtocolMessageType('RegisterNodeRequest', (_message.Message,), {
  'DESCRIPTOR' : _REGISTERNODEREQUEST,
  '__module__' : 'addnn.controller.proto.controller_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.controller.RegisterNodeRequest)
  })
_sym_db.RegisterMessage(RegisterNodeRequest)

RegisterNodeResponse = _reflection.GeneratedProtocolMessageType('RegisterNodeResponse', (_message.Message,), {
  'DESCRIPTOR' : _REGISTERNODERESPONSE,
  '__module__' : 'addnn.controller.proto.controller_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.controller.RegisterNodeResponse)
  })
_sym_db.RegisterMessage(RegisterNodeResponse)

DeregisterNodeRequest = _reflection.GeneratedProtocolMessageType('DeregisterNodeRequest', (_message.Message,), {
  'DESCRIPTOR' : _DEREGISTERNODEREQUEST,
  '__module__' : 'addnn.controller.proto.controller_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.controller.DeregisterNodeRequest)
  })
_sym_db.RegisterMessage(DeregisterNodeRequest)

UpdateNodeStateRequest = _reflection.GeneratedProtocolMessageType('UpdateNodeStateRequest', (_message.Message,), {
  'DESCRIPTOR' : _UPDATENODESTATEREQUEST,
  '__module__' : 'addnn.controller.proto.controller_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.controller.UpdateNodeStateRequest)
  })
_sym_db.RegisterMessage(UpdateNodeStateRequest)

RegisteredNode = _reflection.GeneratedProtocolMessageType('RegisteredNode', (_message.Message,), {
  'DESCRIPTOR' : _REGISTEREDNODE,
  '__module__' : 'addnn.controller.proto.controller_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.controller.RegisteredNode)
  })
_sym_db.RegisterMessage(RegisteredNode)

ListNodesResponse = _reflection.GeneratedProtocolMessageType('ListNodesResponse', (_message.Message,), {
  'DESCRIPTOR' : _LISTNODESRESPONSE,
  '__module__' : 'addnn.controller.proto.controller_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.controller.ListNodesResponse)
  })
_sym_db.RegisterMessage(ListNodesResponse)



_CONTROLLER = _descriptor.ServiceDescriptor(
  name='Controller',
  full_name='addnn.grpc.controller.Controller',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=656,
  serialized_end=1017,
  methods=[
  _descriptor.MethodDescriptor(
    name='RegisterNode',
    full_name='addnn.grpc.controller.Controller.RegisterNode',
    index=0,
    containing_service=None,
    input_type=_REGISTERNODEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='DeregisterNode',
    full_name='addnn.grpc.controller.Controller.DeregisterNode',
    index=1,
    containing_service=None,
    input_type=_DEREGISTERNODEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='UpdateNodeState',
    full_name='addnn.grpc.controller.Controller.UpdateNodeState',
    index=2,
    containing_service=None,
    input_type=_UPDATENODESTATEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ListNodes',
    full_name='addnn.grpc.controller.Controller.ListNodes',
    index=3,
    containing_service=None,
    input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    output_type=_LISTNODESRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_CONTROLLER)

DESCRIPTOR.services_by_name['Controller'] = _CONTROLLER

# @@protoc_insertion_point(module_scope)
