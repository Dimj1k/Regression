from abc import ABC, abstractmethod
from .ABCRegression import Regression


class Graph(ABC):

    pt_curve: Regression

    @abstractmethod
    def __init__(self, curve):
        pass

    @abstractmethod
    def show(self):
        pass
