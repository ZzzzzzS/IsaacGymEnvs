from abc import ABC
from abc import abstractclassmethod, abstractproperty


class BasePlayer(ABC):
    def __init__(self, params):
        pass

    @abstractclassmethod
    def restore(self, fn: str):
        pass

    @abstractproperty
    def device(self):
        pass

    @abstractclassmethod
    def set_weights(self, weights: str):
        pass

    @abstractclassmethod
    def run(self):
        pass

    @abstractclassmethod
    def run_step(self):
        pass
