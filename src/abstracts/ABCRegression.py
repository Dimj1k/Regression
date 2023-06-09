from abc import ABC, abstractmethod
from pandas import read_csv
from numpy import ndarray
from pathlib import Path


def get_f_table_func(alpha, k1, k2):
    alpha = alpha.replace(".", ",", 1)
    pathtofile = Path(__file__).parent / ".." / "Таблицы распределений" / f"alpha {alpha}.csv"
    if k1 > 30:
        if 30 <= k1 <= 35:
            k1 = 31
        elif 35 < k1 < 50:
            k1 = 32
        elif 50 <= k1 <= 80:
            k1 = 33
        elif k1 > 80:
            k1 = 34
    if k2 > 30:
        if 30 <= k2 <= 35:
            k2 = 40
        elif 35 < k2 <= 50:
            k2 = 60
        elif 50 <= k2 <= 150:
            k2 = 120
        else:
            k2 = "4294967296"
    try:
        csv = read_csv(pathtofile, delimiter=";")
    except FileNotFoundError:
        return 4
    return float(csv[str(k2)].iloc[k1 - 1].replace(",", '.'))


class AbstractOneDRegression(ABC):

    f_table: float
    f_fact: float
    r: float
    nonlin_r: float
    r2: float
    reg_x: ndarray
    reg_y: ndarray
    rv_up: ndarray
    rv_down: ndarray
    pred_up: ndarray
    pred_down: ndarray

    @staticmethod
    def get_f_table(alpha, k1, k2):
        return get_f_table_func(alpha, k1, k2)

    @abstractmethod
    def __init__(self, x, y, xname, yname):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def points(self, p1, p2, step):
        pass

    @abstractmethod
    def dim(self):
        pass

    @abstractmethod
    def prediction(self):
        pass

    @abstractmethod
    def correlation_f(self, alpha):
        pass

    @abstractmethod
    def is_norm(self):
        pass

    @abstractmethod
    def approx_error(self):
        pass

    @abstractmethod
    def can_be_linear(self):
        pass


class AbstractMultiplyDRegression(ABC):

    r: float
    r2: float
    adjR2: float
    adjR2arr: list
    f_table_all: float
    f_fact_all: float
    f_table_each: float
    f_fact_each: list
    reg_xy: ndarray
    rv_up: ndarray
    rv_down: ndarray
    pred_up: ndarray
    pred_down: ndarray

    @staticmethod
    def get_f_table(alpha, k1, k2):
        return get_f_table_func(alpha, k1, k2)

    @abstractmethod
    def approx_error(self):
        pass

    @abstractmethod
    def params_is_norm(self):
        pass

    @abstractmethod
    def __init__(self, x, y):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def points(self, p1, p2):
        pass

    @abstractmethod
    def dim(self):
        pass

    @abstractmethod
    def prediction(self):
        pass

    @abstractmethod
    def correlation_f(self, alpha):
        pass

    @abstractmethod
    def is_norm(self):
        pass

    @abstractmethod
    def get_used_variables(self):
        pass

    @abstractmethod
    def get_unused_variables(self):
        pass
