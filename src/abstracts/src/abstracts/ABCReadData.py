from abc import ABC, abstractmethod
from numpy import ndarray


class AbstractData(ABC):

    Xnames: list
    Yname: str
    x: ndarray
    y: ndarray
    allNames: list
    allData: list

    @abstractmethod
    def __init__(self, io):
        pass

    @abstractmethod
    def dim(self):
        pass

    @abstractmethod
    def get_min_x(self):
        pass

    @abstractmethod
    def get_max_x(self):
        pass

    @abstractmethod
    def select_spec_x_y(self, choisen_x, choisen_y):
        pass

    @abstractmethod
    def names(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self):
        pass
