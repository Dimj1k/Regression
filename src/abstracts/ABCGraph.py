from abc import ABC, abstractmethod
from .ABCRegression import AbstractOneDRegression, AbstractMultiplyDRegression


class Graph(ABC):

    pt_curve: AbstractOneDRegression | AbstractMultiplyDRegression

    @abstractmethod
    def __init__(self, curve):
        pass

    @abstractmethod
    def show(self):
        pass
