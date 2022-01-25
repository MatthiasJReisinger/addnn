from addnn.controller.proto import controller_pb2, controller_pb2_grpc
from google.protobuf.empty_pb2 import Empty


def get_input_node(controller_stub: controller_pb2_grpc.ControllerStub) -> controller_pb2.Node:
    # retrieve input node from controller
    list_nodes_response = controller_stub.ListNodes(Empty())
    input_nodes = list(filter(lambda registered_node: registered_node.node.is_input, list_nodes_response.nodes))
    number_of_input_nodes = len(input_nodes)

    if number_of_input_nodes != 1:
        raise Exception(
            "there has to be exactly one input node, but currenly there are {}".format(number_of_input_nodes))

    input_node = input_nodes[0].node
    return input_node
