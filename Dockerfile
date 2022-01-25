FROM ubuntu:latest

# Install required system packages.
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y software-properties-common && \
    apt-get install -y curl python3.9 python3.9-dev python3.9-distutils gcc make iperf3

# Install poetry as described in https://python-poetry.org/docs/#installation.
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
ENV PATH=/root/.poetry/bin:$PATH

# Copy the project sources to their destination.
COPY src /addnn/src
COPY pyproject.toml /addnn
COPY poetry.lock /addnn

# Set the project root as working directory.
WORKDIR /addnn

# Install the project dependencies.
RUN poetry install

# Set the root 'addnn' command as entry point. When starting a container, the desired sub-commands, as well as their
# arguments, can simply be appended as arguments to `docker run`.
ENTRYPOINT ["poetry", "run", "addnn"]
