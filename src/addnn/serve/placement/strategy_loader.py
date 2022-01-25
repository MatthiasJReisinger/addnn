import addnn.serve.placement.strategies
import importlib
import pkgutil
import types
from addnn.serve.placement.strategy import Strategy
from inspect import isclass
from typing import List


def load_strategies() -> List[Strategy]:
    strategies = []
    strategy_plugin_namespace = addnn.serve.placement.strategies

    for module_info in pkgutil.iter_modules(
            strategy_plugin_namespace.__path__,  # type: ignore
            strategy_plugin_namespace.__name__ + "."):
        strategy_module = importlib.import_module(module_info.name)
        strategies_in_module = _load_strategies_from_module(strategy_module)
        strategies.extend(strategies_in_module)

    return strategies


def _load_strategies_from_module(module: types.ModuleType) -> List[Strategy]:
    strategies = []
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if isclass(attribute):
            module_class = attribute
            if issubclass(module_class, Strategy) and not module_class is Strategy:
                strategy = module_class()
                strategies.append(strategy)
    return strategies


def get_available_strategy_names() -> List[str]:
    strategies = load_strategies()
    return [strategy.name() for strategy in strategies]


def load_strategy(strategy_name: str) -> Strategy:
    strategies = load_strategies()
    for strategy in strategies:
        if strategy.name() == strategy_name:
            return strategy
    raise Exception("no strategy found with name '{}'".format(strategy_name))
