import addnn
import addnn.dataset
import addnn.grpc
import click
import grpc
import io
import pickle
import PIL
import requests
import time
import torch
from addnn.cli import cli
from addnn.controller.proto.controller_pb2_grpc import ControllerStub
from addnn.node.proto.neural_network_pb2 import InferRequest
from addnn.node.proto.neural_network_pb2_grpc import NeuralNetworkStub
from google.protobuf.empty_pb2 import Empty
from torchvision import transforms


@cli.command("infer", help="Trigger DNN inference for an input sample.")
@click.option("--controller-host", required=True, help="The host that runs the controller.")
@click.option("--controller-port", type=int, required=True, help="The port at which to reach the controller.")
@click.option("--image-url", help="URL of an image to classify.")
@click.option("--image-file", type=click.Path(exists=True), help="File path of an image to classify.")
@click.option("--dataset", "dataset_name", type=click.Choice(addnn.dataset.datasets.keys()), help="The dataset to use.")
@click.option("--random", help="Classify a randomly generated tensor with the given dimensions (e.g., '3,224,224').")
def run(controller_host: str, controller_port: int, image_url: str, image_file: str, dataset_name: str, random: str) -> None:
    if image_url is not None or image_file is not None:
        if image_url is not None:
            print("classifying image at URL {}".format(image_url))
            response = requests.get(image_url)
            input_image = PIL.Image.open(io.BytesIO(response.content))
        else:
            print("classifying image file {}".format(image_file))
            input_image = PIL.Image.open(image_file)

        if dataset_name:
            dataset = addnn.dataset.datasets[dataset_name]
            input_tensor = dataset.test_set_normalization(input_image)
        else:
            input_tensor = transforms.functional.to_tensor(input_image)
    elif random is not None:
        dimensions = [int(dimension) for dimension in random.split(",")]
        input_tensor = torch.rand(*dimensions)
        print("classifying random input of shape {}".format(random))
    else:
        raise Exception("no input specified")

    # add batch dimension
    input_tensor = input_tensor.unsqueeze(0)

    # retrieve input node from controller
    controller_stub = addnn.grpc.create_tcp_stub(ControllerStub, controller_host, controller_port)
    list_nodes_response = controller_stub.ListNodes(Empty())
    input_nodes = list(filter(lambda registered_node: registered_node.node.is_input, list_nodes_response.nodes))
    number_of_input_nodes = len(input_nodes)

    if number_of_input_nodes != 1:
        raise Exception(
                "there has to be exactly one input node, but currenly there are {}".format(number_of_input_nodes))

    input_node = input_nodes[0].node

    # connect to input node
    neural_network_stub = addnn.grpc.create_tcp_stub(NeuralNetworkStub, input_node.host, input_node.port)

    # send inference request
    infer_request = InferRequest()
    infer_request.tensor = pickle.dumps(input_tensor)
    request_time = time.time_ns()
    infer_response = neural_network_stub.Infer(infer_request)
    response_time = time.time_ns()

    if dataset_name == "imagenet":
        response = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
        imagenet_classes = [line.strip() for line in response.text.splitlines()]
        imagenet_class = imagenet_classes[infer_response.classification]
        print("classified image as '{}' (ImageNet class label {})".format(imagenet_class,
            infer_response.classification))
    else:
        print("inferred {}".format(infer_response.classification))

    latency = float(response_time - request_time) / 1000 **3
    print("latency: {} seconds".format(latency))
