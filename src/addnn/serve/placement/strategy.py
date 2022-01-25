from addnn.controller.proto.controller_pb2 import Node
from addnn.profile.layer_profile import LayerProfile
from addnn.serve.placement.placement import Placement
from abc import ABC, abstractmethod
from typing import List


class Strategy(ABC):
    """
    Base class for placement strategies.
    """
    @abstractmethod
    def name(self) -> str:
        """
        A name that uniquely identifies this strategy.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_placement(self, nodes: List[Node], layers: List[LayerProfile]) -> Placement:
        """
        Compute a placement that assigns each layer to a compute node.
        """
        raise NotImplementedError
