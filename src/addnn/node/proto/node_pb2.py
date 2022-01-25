# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: addnn/node/proto/node.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from addnn.node.proto import node_state_pb2 as addnn_dot_node_dot_proto_dot_node__state__pb2
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from addnn.controller.proto import controller_pb2 as addnn_dot_controller_dot_proto_dot_controller__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='addnn/node/proto/node.proto',
  package='addnn.grpc.node',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x1b\x61\x64\x64nn/node/proto/node.proto\x12\x0f\x61\x64\x64nn.grpc.node\x1a!addnn/node/proto/node_state.proto\x1a\x1bgoogle/protobuf/empty.proto\x1a\'addnn/controller/proto/controller.proto\"8\n\x04\x45xit\x12\x12\n\nclassifier\x18\x01 \x01(\x0c\x12\x1c\n\x14\x63onfidence_threshold\x18\x02 \x01(\x02\")\n\x0bRemoteLayer\x12\x0c\n\x04host\x18\x01 \x01(\t\x12\x0c\n\x04port\x18\x02 \x01(\x05\"e\n\nLocalLayer\x12\x13\n\x0bmain_branch\x18\x01 \x01(\x0c\x12*\n\x0b\x65xit_branch\x18\x02 \x01(\x0b\x32\x15.addnn.grpc.node.Exit\x12\x16\n\x0eis_torchscript\x18\x03 \x01(\x08\"|\n\x05Layer\x12\x30\n\x0blocal_layer\x18\x01 \x01(\x0b\x32\x1b.addnn.grpc.node.LocalLayer\x12\x32\n\x0cremote_layer\x18\x02 \x01(\x0b\x32\x1c.addnn.grpc.node.RemoteLayer\x12\r\n\x05index\x18\x03 \x01(\r\"4\n\nLayerRange\x12\x13\n\x0bstart_index\x18\x01 \x01(\r\x12\x11\n\tend_index\x18\x02 \x01(\r\"\x7f\n\x15\x41\x63tivateLayersRequest\x12\x32\n\ractive_layers\x18\x01 \x01(\x0b\x32\x1b.addnn.grpc.node.LayerRange\x12\x32\n\x0cremote_layer\x18\x02 \x01(\x0b\x32\x1c.addnn.grpc.node.RemoteLayer\"L\n\x14ReadNodeStateRequest\x12\x34\n\x0fneighbour_nodes\x18\x01 \x03(\x0b\x32\x1b.addnn.grpc.controller.Node\"M\n\x15ReadNodeStateResponse\x12\x34\n\nnode_state\x18\x01 \x01(\x0b\x32 .addnn.grpc.node_state.NodeState\"Z\n\x1aUpdateResourceStateRequest\x12<\n\x0eresource_state\x18\x01 \x01(\x0b\x32$.addnn.grpc.node_state.ResourceState\"R\n\x1aReadNeighbourNodesResponse\x12\x34\n\x0fneighbour_nodes\x18\x01 \x03(\x0b\x32\x1b.addnn.grpc.controller.Node2\xc6\x04\n\x04Node\x12\x46\n\x0b\x44\x65ployModel\x12\x1b.addnn.grpc.node.LocalLayer\x1a\x16.google.protobuf.Empty\"\x00(\x01\x12?\n\x0b\x44\x65leteModel\x12\x16.google.protobuf.Empty\x1a\x16.google.protobuf.Empty\"\x00\x12R\n\x0e\x41\x63tivateLayers\x12&.addnn.grpc.node.ActivateLayersRequest\x1a\x16.google.protobuf.Empty\"\x00\x12\x44\n\x10\x44\x65\x61\x63tivateLayers\x12\x16.google.protobuf.Empty\x1a\x16.google.protobuf.Empty\"\x00\x12`\n\rReadNodeState\x12%.addnn.grpc.node.ReadNodeStateRequest\x1a&.addnn.grpc.node.ReadNodeStateResponse\"\x00\x12\\\n\x13UpdateResourceState\x12+.addnn.grpc.node.UpdateResourceStateRequest\x1a\x16.google.protobuf.Empty\"\x00\x12[\n\x12ReadNeighbourNodes\x12\x16.google.protobuf.Empty\x1a+.addnn.grpc.node.ReadNeighbourNodesResponse\"\x00\x62\x06proto3'
  ,
  dependencies=[addnn_dot_node_dot_proto_dot_node__state__pb2.DESCRIPTOR,google_dot_protobuf_dot_empty__pb2.DESCRIPTOR,addnn_dot_controller_dot_proto_dot_controller__pb2.DESCRIPTOR,])




_EXIT = _descriptor.Descriptor(
  name='Exit',
  full_name='addnn.grpc.node.Exit',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='classifier', full_name='addnn.grpc.node.Exit.classifier', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='confidence_threshold', full_name='addnn.grpc.node.Exit.confidence_threshold', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=153,
  serialized_end=209,
)


_REMOTELAYER = _descriptor.Descriptor(
  name='RemoteLayer',
  full_name='addnn.grpc.node.RemoteLayer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='host', full_name='addnn.grpc.node.RemoteLayer.host', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='port', full_name='addnn.grpc.node.RemoteLayer.port', index=1,
      number=2, type=5, cpp_type=1, label=1,
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
  serialized_start=211,
  serialized_end=252,
)


_LOCALLAYER = _descriptor.Descriptor(
  name='LocalLayer',
  full_name='addnn.grpc.node.LocalLayer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='main_branch', full_name='addnn.grpc.node.LocalLayer.main_branch', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='exit_branch', full_name='addnn.grpc.node.LocalLayer.exit_branch', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='is_torchscript', full_name='addnn.grpc.node.LocalLayer.is_torchscript', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
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
  serialized_start=254,
  serialized_end=355,
)


_LAYER = _descriptor.Descriptor(
  name='Layer',
  full_name='addnn.grpc.node.Layer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='local_layer', full_name='addnn.grpc.node.Layer.local_layer', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='remote_layer', full_name='addnn.grpc.node.Layer.remote_layer', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='index', full_name='addnn.grpc.node.Layer.index', index=2,
      number=3, type=13, cpp_type=3, label=1,
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
  serialized_start=357,
  serialized_end=481,
)


_LAYERRANGE = _descriptor.Descriptor(
  name='LayerRange',
  full_name='addnn.grpc.node.LayerRange',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='start_index', full_name='addnn.grpc.node.LayerRange.start_index', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='end_index', full_name='addnn.grpc.node.LayerRange.end_index', index=1,
      number=2, type=13, cpp_type=3, label=1,
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
  serialized_start=483,
  serialized_end=535,
)


_ACTIVATELAYERSREQUEST = _descriptor.Descriptor(
  name='ActivateLayersRequest',
  full_name='addnn.grpc.node.ActivateLayersRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='active_layers', full_name='addnn.grpc.node.ActivateLayersRequest.active_layers', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='remote_layer', full_name='addnn.grpc.node.ActivateLayersRequest.remote_layer', index=1,
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
  serialized_start=537,
  serialized_end=664,
)


_READNODESTATEREQUEST = _descriptor.Descriptor(
  name='ReadNodeStateRequest',
  full_name='addnn.grpc.node.ReadNodeStateRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='neighbour_nodes', full_name='addnn.grpc.node.ReadNodeStateRequest.neighbour_nodes', index=0,
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
  serialized_start=666,
  serialized_end=742,
)


_READNODESTATERESPONSE = _descriptor.Descriptor(
  name='ReadNodeStateResponse',
  full_name='addnn.grpc.node.ReadNodeStateResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='node_state', full_name='addnn.grpc.node.ReadNodeStateResponse.node_state', index=0,
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
  serialized_start=744,
  serialized_end=821,
)


_UPDATERESOURCESTATEREQUEST = _descriptor.Descriptor(
  name='UpdateResourceStateRequest',
  full_name='addnn.grpc.node.UpdateResourceStateRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='resource_state', full_name='addnn.grpc.node.UpdateResourceStateRequest.resource_state', index=0,
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
  serialized_start=823,
  serialized_end=913,
)


_READNEIGHBOURNODESRESPONSE = _descriptor.Descriptor(
  name='ReadNeighbourNodesResponse',
  full_name='addnn.grpc.node.ReadNeighbourNodesResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='neighbour_nodes', full_name='addnn.grpc.node.ReadNeighbourNodesResponse.neighbour_nodes', index=0,
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
  serialized_start=915,
  serialized_end=997,
)

_LOCALLAYER.fields_by_name['exit_branch'].message_type = _EXIT
_LAYER.fields_by_name['local_layer'].message_type = _LOCALLAYER
_LAYER.fields_by_name['remote_layer'].message_type = _REMOTELAYER
_ACTIVATELAYERSREQUEST.fields_by_name['active_layers'].message_type = _LAYERRANGE
_ACTIVATELAYERSREQUEST.fields_by_name['remote_layer'].message_type = _REMOTELAYER
_READNODESTATEREQUEST.fields_by_name['neighbour_nodes'].message_type = addnn_dot_controller_dot_proto_dot_controller__pb2._NODE
_READNODESTATERESPONSE.fields_by_name['node_state'].message_type = addnn_dot_node_dot_proto_dot_node__state__pb2._NODESTATE
_UPDATERESOURCESTATEREQUEST.fields_by_name['resource_state'].message_type = addnn_dot_node_dot_proto_dot_node__state__pb2._RESOURCESTATE
_READNEIGHBOURNODESRESPONSE.fields_by_name['neighbour_nodes'].message_type = addnn_dot_controller_dot_proto_dot_controller__pb2._NODE
DESCRIPTOR.message_types_by_name['Exit'] = _EXIT
DESCRIPTOR.message_types_by_name['RemoteLayer'] = _REMOTELAYER
DESCRIPTOR.message_types_by_name['LocalLayer'] = _LOCALLAYER
DESCRIPTOR.message_types_by_name['Layer'] = _LAYER
DESCRIPTOR.message_types_by_name['LayerRange'] = _LAYERRANGE
DESCRIPTOR.message_types_by_name['ActivateLayersRequest'] = _ACTIVATELAYERSREQUEST
DESCRIPTOR.message_types_by_name['ReadNodeStateRequest'] = _READNODESTATEREQUEST
DESCRIPTOR.message_types_by_name['ReadNodeStateResponse'] = _READNODESTATERESPONSE
DESCRIPTOR.message_types_by_name['UpdateResourceStateRequest'] = _UPDATERESOURCESTATEREQUEST
DESCRIPTOR.message_types_by_name['ReadNeighbourNodesResponse'] = _READNEIGHBOURNODESRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Exit = _reflection.GeneratedProtocolMessageType('Exit', (_message.Message,), {
  'DESCRIPTOR' : _EXIT,
  '__module__' : 'addnn.node.proto.node_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.node.Exit)
  })
_sym_db.RegisterMessage(Exit)

RemoteLayer = _reflection.GeneratedProtocolMessageType('RemoteLayer', (_message.Message,), {
  'DESCRIPTOR' : _REMOTELAYER,
  '__module__' : 'addnn.node.proto.node_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.node.RemoteLayer)
  })
_sym_db.RegisterMessage(RemoteLayer)

LocalLayer = _reflection.GeneratedProtocolMessageType('LocalLayer', (_message.Message,), {
  'DESCRIPTOR' : _LOCALLAYER,
  '__module__' : 'addnn.node.proto.node_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.node.LocalLayer)
  })
_sym_db.RegisterMessage(LocalLayer)

Layer = _reflection.GeneratedProtocolMessageType('Layer', (_message.Message,), {
  'DESCRIPTOR' : _LAYER,
  '__module__' : 'addnn.node.proto.node_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.node.Layer)
  })
_sym_db.RegisterMessage(Layer)

LayerRange = _reflection.GeneratedProtocolMessageType('LayerRange', (_message.Message,), {
  'DESCRIPTOR' : _LAYERRANGE,
  '__module__' : 'addnn.node.proto.node_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.node.LayerRange)
  })
_sym_db.RegisterMessage(LayerRange)

ActivateLayersRequest = _reflection.GeneratedProtocolMessageType('ActivateLayersRequest', (_message.Message,), {
  'DESCRIPTOR' : _ACTIVATELAYERSREQUEST,
  '__module__' : 'addnn.node.proto.node_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.node.ActivateLayersRequest)
  })
_sym_db.RegisterMessage(ActivateLayersRequest)

ReadNodeStateRequest = _reflection.GeneratedProtocolMessageType('ReadNodeStateRequest', (_message.Message,), {
  'DESCRIPTOR' : _READNODESTATEREQUEST,
  '__module__' : 'addnn.node.proto.node_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.node.ReadNodeStateRequest)
  })
_sym_db.RegisterMessage(ReadNodeStateRequest)

ReadNodeStateResponse = _reflection.GeneratedProtocolMessageType('ReadNodeStateResponse', (_message.Message,), {
  'DESCRIPTOR' : _READNODESTATERESPONSE,
  '__module__' : 'addnn.node.proto.node_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.node.ReadNodeStateResponse)
  })
_sym_db.RegisterMessage(ReadNodeStateResponse)

UpdateResourceStateRequest = _reflection.GeneratedProtocolMessageType('UpdateResourceStateRequest', (_message.Message,), {
  'DESCRIPTOR' : _UPDATERESOURCESTATEREQUEST,
  '__module__' : 'addnn.node.proto.node_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.node.UpdateResourceStateRequest)
  })
_sym_db.RegisterMessage(UpdateResourceStateRequest)

ReadNeighbourNodesResponse = _reflection.GeneratedProtocolMessageType('ReadNeighbourNodesResponse', (_message.Message,), {
  'DESCRIPTOR' : _READNEIGHBOURNODESRESPONSE,
  '__module__' : 'addnn.node.proto.node_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.node.ReadNeighbourNodesResponse)
  })
_sym_db.RegisterMessage(ReadNeighbourNodesResponse)



_NODE = _descriptor.ServiceDescriptor(
  name='Node',
  full_name='addnn.grpc.node.Node',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=1000,
  serialized_end=1582,
  methods=[
  _descriptor.MethodDescriptor(
    name='DeployModel',
    full_name='addnn.grpc.node.Node.DeployModel',
    index=0,
    containing_service=None,
    input_type=_LOCALLAYER,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='DeleteModel',
    full_name='addnn.grpc.node.Node.DeleteModel',
    index=1,
    containing_service=None,
    input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ActivateLayers',
    full_name='addnn.grpc.node.Node.ActivateLayers',
    index=2,
    containing_service=None,
    input_type=_ACTIVATELAYERSREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='DeactivateLayers',
    full_name='addnn.grpc.node.Node.DeactivateLayers',
    index=3,
    containing_service=None,
    input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ReadNodeState',
    full_name='addnn.grpc.node.Node.ReadNodeState',
    index=4,
    containing_service=None,
    input_type=_READNODESTATEREQUEST,
    output_type=_READNODESTATERESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='UpdateResourceState',
    full_name='addnn.grpc.node.Node.UpdateResourceState',
    index=5,
    containing_service=None,
    input_type=_UPDATERESOURCESTATEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ReadNeighbourNodes',
    full_name='addnn.grpc.node.Node.ReadNeighbourNodes',
    index=6,
    containing_service=None,
    input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    output_type=_READNEIGHBOURNODESRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_NODE)

DESCRIPTOR.services_by_name['Node'] = _NODE

# @@protoc_insertion_point(module_scope)
