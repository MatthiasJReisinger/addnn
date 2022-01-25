import ipaddress
import grpc
import grpc.aio
from typing import Callable, TypeVar

StubType = TypeVar('StubType')

def create_tcp_stub(stub_type: Callable[..., StubType], host: str, port: int) -> StubType:
    endpoint = create_endpoint(host, port)
    channel = grpc.insecure_channel(endpoint)
    stub = stub_type(channel)
    return stub

def create_channel(host: str, port: int) -> grpc.Channel:
    endpoint = create_endpoint(host, port)
    channel = grpc.insecure_channel(endpoint)
    return channel

def create_async_channel(host: str, port: int) -> grpc.aio.Channel:
    endpoint = create_endpoint(host, port)
    channel = grpc.aio.insecure_channel(endpoint)
    return channel

def create_endpoint(bind_ip: str, bind_port: int) -> str:
    bind_ip_version = ipaddress.ip_address(bind_ip).version

    if bind_ip_version == 6:
        formatted_bind_ip = "[{}]".format(bind_ip)
    else:
        formatted_bind_ip = bind_ip

    endpoint = "{}:{}".format(formatted_bind_ip, bind_port)
    return endpoint
