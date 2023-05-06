from abc import ABC, abstractmethod


class Graph(ABC):

    pt_curve = None

    @abstractmethod
    def __init__(self, curve):
        pass

    @abstractmethod
    def show(self):
        pass
