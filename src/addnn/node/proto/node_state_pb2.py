# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: addnn/node/proto/node_state.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='addnn/node/proto/node_state.proto',
  package='addnn.grpc.node_state',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n!addnn/node/proto/node_state.proto\x12\x15\x61\x64\x64nn.grpc.node_state\"\x92\x01\n\tNodeState\x12<\n\x0eresource_state\x18\x01 \x01(\x0b\x32$.addnn.grpc.node_state.ResourceState\x12G\n\x14neural_network_state\x18\x02 \x01(\x0b\x32).addnn.grpc.node_state.NeuralNetworkState\"\xdd\x01\n\rResourceState\x12\x0e\n\x06memory\x18\x01 \x01(\x04\x12\x0f\n\x07storage\x18\x02 \x01(\x04\x12\x0f\n\x07\x63ompute\x18\x03 \x01(\x04\x12\x11\n\tbandwidth\x18\x04 \x01(\x04\x12\x45\n\x13network_throughputs\x18\x05 \x03(\x0b\x32(.addnn.grpc.node_state.NetworkThroughput\x12@\n\x11network_latencies\x18\x06 \x03(\x0b\x32%.addnn.grpc.node_state.NetworkLatency\"5\n\x11NetworkThroughput\x12\x0c\n\x04host\x18\x01 \x01(\t\x12\x12\n\nthroughput\x18\x02 \x01(\x04\"/\n\x0eNetworkLatency\x12\x0c\n\x04host\x18\x01 \x01(\t\x12\x0f\n\x07latency\x18\x02 \x01(\x02\"M\n\x12NeuralNetworkState\x12\x37\n\x0clayer_states\x18\x01 \x03(\x0b\x32!.addnn.grpc.node_state.LayerState\"S\n\nLayerState\x12\x13\n\x0blayer_index\x18\x01 \x01(\r\x12 \n\x18number_of_exited_samples\x18\x02 \x01(\r\x12\x0e\n\x06\x61\x63tive\x18\x03 \x01(\x08\x62\x06proto3'
)




_NODESTATE = _descriptor.Descriptor(
  name='NodeState',
  full_name='addnn.grpc.node_state.NodeState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='resource_state', full_name='addnn.grpc.node_state.NodeState.resource_state', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='neural_network_state', full_name='addnn.grpc.node_state.NodeState.neural_network_state', index=1,
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
  serialized_start=61,
  serialized_end=207,
)


_RESOURCESTATE = _descriptor.Descriptor(
  name='ResourceState',
  full_name='addnn.grpc.node_state.ResourceState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='memory', full_name='addnn.grpc.node_state.ResourceState.memory', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='storage', full_name='addnn.grpc.node_state.ResourceState.storage', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='compute', full_name='addnn.grpc.node_state.ResourceState.compute', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bandwidth', full_name='addnn.grpc.node_state.ResourceState.bandwidth', index=3,
      number=4, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='network_throughputs', full_name='addnn.grpc.node_state.ResourceState.network_throughputs', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='network_latencies', full_name='addnn.grpc.node_state.ResourceState.network_latencies', index=5,
      number=6, type=11, cpp_type=10, label=3,
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
  serialized_start=210,
  serialized_end=431,
)


_NETWORKTHROUGHPUT = _descriptor.Descriptor(
  name='NetworkThroughput',
  full_name='addnn.grpc.node_state.NetworkThroughput',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='host', full_name='addnn.grpc.node_state.NetworkThroughput.host', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='throughput', full_name='addnn.grpc.node_state.NetworkThroughput.throughput', index=1,
      number=2, type=4, cpp_type=4, label=1,
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
  serialized_start=433,
  serialized_end=486,
)


_NETWORKLATENCY = _descriptor.Descriptor(
  name='NetworkLatency',
  full_name='addnn.grpc.node_state.NetworkLatency',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='host', full_name='addnn.grpc.node_state.NetworkLatency.host', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='latency', full_name='addnn.grpc.node_state.NetworkLatency.latency', index=1,
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
  serialized_start=488,
  serialized_end=535,
)


_NEURALNETWORKSTATE = _descriptor.Descriptor(
  name='NeuralNetworkState',
  full_name='addnn.grpc.node_state.NeuralNetworkState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='layer_states', full_name='addnn.grpc.node_state.NeuralNetworkState.layer_states', index=0,
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
  serialized_start=537,
  serialized_end=614,
)


_LAYERSTATE = _descriptor.Descriptor(
  name='LayerState',
  full_name='addnn.grpc.node_state.LayerState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='layer_index', full_name='addnn.grpc.node_state.LayerState.layer_index', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='number_of_exited_samples', full_name='addnn.grpc.node_state.LayerState.number_of_exited_samples', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='active', full_name='addnn.grpc.node_state.LayerState.active', index=2,
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
  serialized_start=616,
  serialized_end=699,
)

_NODESTATE.fields_by_name['resource_state'].message_type = _RESOURCESTATE
_NODESTATE.fields_by_name['neural_network_state'].message_type = _NEURALNETWORKSTATE
_RESOURCESTATE.fields_by_name['network_throughputs'].message_type = _NETWORKTHROUGHPUT
_RESOURCESTATE.fields_by_name['network_latencies'].message_type = _NETWORKLATENCY
_NEURALNETWORKSTATE.fields_by_name['layer_states'].message_type = _LAYERSTATE
DESCRIPTOR.message_types_by_name['NodeState'] = _NODESTATE
DESCRIPTOR.message_types_by_name['ResourceState'] = _RESOURCESTATE
DESCRIPTOR.message_types_by_name['NetworkThroughput'] = _NETWORKTHROUGHPUT
DESCRIPTOR.message_types_by_name['NetworkLatency'] = _NETWORKLATENCY
DESCRIPTOR.message_types_by_name['NeuralNetworkState'] = _NEURALNETWORKSTATE
DESCRIPTOR.message_types_by_name['LayerState'] = _LAYERSTATE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

NodeState = _reflection.GeneratedProtocolMessageType('NodeState', (_message.Message,), {
  'DESCRIPTOR' : _NODESTATE,
  '__module__' : 'addnn.node.proto.node_state_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.node_state.NodeState)
  })
_sym_db.RegisterMessage(NodeState)

ResourceState = _reflection.GeneratedProtocolMessageType('ResourceState', (_message.Message,), {
  'DESCRIPTOR' : _RESOURCESTATE,
  '__module__' : 'addnn.node.proto.node_state_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.node_state.ResourceState)
  })
_sym_db.RegisterMessage(ResourceState)

NetworkThroughput = _reflection.GeneratedProtocolMessageType('NetworkThroughput', (_message.Message,), {
  'DESCRIPTOR' : _NETWORKTHROUGHPUT,
  '__module__' : 'addnn.node.proto.node_state_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.node_state.NetworkThroughput)
  })
_sym_db.RegisterMessage(NetworkThroughput)

NetworkLatency = _reflection.GeneratedProtocolMessageType('NetworkLatency', (_message.Message,), {
  'DESCRIPTOR' : _NETWORKLATENCY,
  '__module__' : 'addnn.node.proto.node_state_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.node_state.NetworkLatency)
  })
_sym_db.RegisterMessage(NetworkLatency)

NeuralNetworkState = _reflection.GeneratedProtocolMessageType('NeuralNetworkState', (_message.Message,), {
  'DESCRIPTOR' : _NEURALNETWORKSTATE,
  '__module__' : 'addnn.node.proto.node_state_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.node_state.NeuralNetworkState)
  })
_sym_db.RegisterMessage(NeuralNetworkState)

LayerState = _reflection.GeneratedProtocolMessageType('LayerState', (_message.Message,), {
  'DESCRIPTOR' : _LAYERSTATE,
  '__module__' : 'addnn.node.proto.node_state_pb2'
  # @@protoc_insertion_point(class_scope:addnn.grpc.node_state.LayerState)
  })
_sym_db.RegisterMessage(LayerState)


# @@protoc_insertion_point(module_scope)
