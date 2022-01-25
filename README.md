# addnn

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A library and CLI toolchain for implementation, training, and serving of adaptive distributed deep neural networks.

## Setup

This project requires Python 3.9 and uses [poetry](https://python-poetry.org/) for dependency managemend (see [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation) for installation instructions).
In order to set up a virtual environment and install all dependencies, run the following command:

```
poetry install
```

One project dependency, namely `python_papi`, comes with native code that requires `gcc` for compilation.
So if your system's default compiler isn't `gcc`, it might be necessary override your compiler settings when installing the project:

```
CC=gcc CXX=g++ poetry install
```

Also, if new dependencies have been added to [pyproject.toml](pyproject.toml), just rerun the above command to install those dependencies in your virtual environment.

## Example: Classifying images with ResNet

Export a pretrained ResNet model:

```
poetry run addnn example --model=resnet18 --pretrained ./resnet18.pt
```

Start an ADDNN controller:

```
poetry run addnn controller --bind-port=42424
```

Start an ADDNN node with the `--is-input` flag to start an ADDNN node that is marked as "input source" for the neural network:

```
poetry run addnn node --controller-host 127.0.0.1 --controller-port 42424 --bind-port=24242 --memory=10000000 --storage=10000000 --compute=1000000000 --bandwidth=1000000 --tier=0 --is-input
```

Start another node on a second compute tier:

```
poetry run addnn node --controller-host 127.0.0.1 --controller-port 42424 --bind-port=24243 --memory=10000000 --storage=10000000 --compute=1000000000 --bandwidth=2000000 --tier=1
```

Serve the pretrained ResNet model to the available nodes based on an "optimal" deployment strategy:

```
poetry run addnn serve --controller-host=127.0.0.1 --controller-port=42424 --mapping=optimal resnet18.pt
```

Classify the image of a dog:

```
poetry run addnn infer --dnn-host=127.0.0.1 --dnn-port=24242 --image-url=https://github.com/pytorch/hub/raw/master/images/dog.jpg
```

## torchscript-atomize

This repository also contains a utility named `torchscript-atomize` that splits TorchScript models into atomic sequential components.
It is based on PyTorch's C++ API and needs to be built separately from the Python based `addnn` tool.

`torchscript-atomize` can be built as follows:

```
cd cpp/torchscript-atomize
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER="/usr/lib/ccache/bin/clang++" ..
cmake --build .
```

The `torchscript-atomize` binary is then located in the `build` folder.

## Contributing

### Static type checking

This project uses [mypy](https://mypy.readthedocs.io/en/latest/index.html) for static type checking.
Where possible, and where meaningful, we use type hints to annote method arguments, returns types, and variables with type information.
For code parts that do not provide any type hints, mypy at least performs syntax checks, which is also quite helpful and improves development experience.

The following command runs mypy for both the `src` and `tests` directory:

```
poetry run mypy
```

### Generate gRPC code

The following command generates gRPC code for all `.proto` files that are placed within the `src` diretory or one of its sub-directories.
Generated python files will be placed in the same directory as their corresponding `.proto` files.

```
poetry run python -m grpc_tools.protoc -Isrc --python_out=src --grpc_python_out=src --mypy_out=src --mypy_grpc_out=src **/*.proto
```

### Run tests

```
poetry run pytest
```

### Code formatting

Automatic code formatting can be done conveniently via [yapf](https://github.com/google/yapf), which can also be used with common editors, e.g., vim ([vim-autoformat](https://github.com/Chiel92/vim-autoformat) and Visual Studio Code.
