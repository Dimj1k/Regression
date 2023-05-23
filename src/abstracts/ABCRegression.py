from abc import ABC, abstractmethod
from pandas import read_csv
from numpy import array
from pathlib import Path


class Regression(ABC):

    f_table: float
    f_fact: float
    r: float
    nonlin_r: float
    r2: float
    rv_up: array
    rv_down: array
    pred_up: array
    pred_down: array
    
    @staticmethod
    def get_f_table(alpha, k1, k2):
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
            return None
        return float(csv[str(k2)].iloc[k1 - 1].replace(",", '.'))

    @abstractmethod
    def __init__(self):
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
    def correlation_f(self, alpha: str):
        pass

    @abstractmethod
    def is_norm(self):
        pass

    @abstractmethod
    def approx_error(self):
        pass


class AbstractOneDRegression(Regression, ABC):

    @abstractmethod
    def can_be_linear(self):
        pass


class AbstractMultiplyDRegression(Regression, ABC):

    f_table_all: float
    f_fact_all: float
    f_table_each: float
    f_fact_each: list
    
    @abstractmethod
    def get_used_variables(self):
        pass

    @abstractmethod
    def get_unused_variables(self):
        pass

    @abstractmethod
    def best_three_adjR2(self):
        pass
