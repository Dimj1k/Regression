from .abstracts.ABCRegression import *
from numpy import (
    matrix,
    sum as npsum,
    array,
    arange,
    min as npmin,
    max as npmax,
    mean,
    sqrt,
    exp,
    float64,
    append,
    square,
    linspace,
    meshgrid,
    abs as npabs,
    log,
    any as npany,
    argwhere,
    sin
)
from scipy.optimize import curve_fit
from operator import itemgetter


class OneDPolynomialRegression(AbstractOneDRegression):

    def names(self):
        return {"x": self.name_x, "y": self.name_y}

    def _least_squares(self):
        return (matrix([[npsum(self.x ** i) for i in range(j, self.pol + j + 1)] for j in range(self.pol + 1)]) ** -1) \
               * [[npsum(self.x ** i * self.y)] for i in range(self.pol + 1)]

    def _function_points(self, x):
        return npsum([self.coeffs[i] * x ** (self.pol - i) for i in range(self.pol + 1)], axis=0)

    def __init__(self, pol, x, y, name_x="x", name_y="y"):
        self.pol = pol
        self.name_y = name_y
        self.x, self.y, self.name_x = array(x).flatten(), array(y).flatten(), name_x[0]
        self.k = len(self.x)
        self.coeffs = array(self._least_squares()).flatten()[::-1]
        self.islinear = True if pol == 1 else False

    def __str__(self):
        coeffs = self.coeffs[::-1]
        return "".join("+".join([f"{coeffs[self.pol - i]:.2f}x^{self.pol - i}" for i in range(self.pol + 1)])
                       .rstrip("x^0").replace("+-", "-").rsplit("^1", maxsplit=1))

    def points(self, p1=None, p2=None, step=0.1):
        x = arange(p1, p2 + step, step=step) if p1 and p2 else arange(npmin(self.x), npmax(self.x) + step, step=step)
        self.reg_x = x
        self.reg_y = self._function_points(x)
        return self

    def correlation_f(self, alpha: str):
        self.r = npsum((self.x - mean(self.x)) * (self.y - mean(self.y))) / \
                 sqrt(npsum((self.x - mean(self.x)) ** 2) * npsum((self.y - mean(self.y)) ** 2))
        self.r2 = self.r ** 2
        m = str(self).replace("exp", "").count("x")
        if self.pol > 1:
            y = self._function_points(self.x)
            self.nonlin_r = sqrt(1 - npsum((self.y - y) ** 2) / npsum((self.y - mean(self.y)) ** 2))
        self.f_table = self.get_f_table(alpha, self.k - 1 - m, m)
        if self.pol == 1:
            self.f_fact = (self.r2 / (1 - self.r2)) * (self.k - m - 1) / m
        else:
            self.f_fact = (self.nonlin_r ** 2 / (1 - self.nonlin_r ** 2)) * (self.k - m - 1) / m
        return self

    def can_be_linear(self):
        assert self.pol > 1
        return f"{abs(self.nonlin_r - abs(self.r)) * 100:.4f} %"

    def is_norm(self):
        assert self.f_table is not None and self.f_fact is not None, "f_table or f_fact is None"
        return abs(self.f_fact) > self.f_table

    def prediction(self):
        mean_x = mean(self.x)
        y = self._function_points(self.x)
        pred = sqrt(self.f_table * (1/self.k + (self.reg_x - mean_x)**2/npsum((self.x - mean_x)**2)) *
                    npsum((self.y - y)**2)/(self.k - 2))
        rv = sqrt(self.f_table * (1 + 1/self.k + (self.reg_x - mean_x)**2/npsum((self.x - mean_x)**2)) *
                  npsum((self.y - y)**2)/(self.k - 2))
        del y, mean_x
        self.pred_up = self.reg_y + pred
        self.pred_down = self.reg_y - pred
        self.rv_up = self.reg_y + rv
        self.rv_down = self.reg_y - rv
        return self

    def approx_error(self):
        y = self._function_points(self.x)
        indexes = argwhere(self.y != 0).flatten()
        f = self.y[indexes]
        return f"{mean(npabs((f - y[indexes]) / f)) * 100:.4f} %"

    def dim(self):
        return 1


class OneDExponentRegression(OneDPolynomialRegression):

    def _least_squares(self):
        coeffs = curve_fit(lambda x, a, b, c: a * exp(b * x) + c, self.x, self.y)[0]
        return [[coeffs[2]], [coeffs[1]], [coeffs[0]]]

    def _function_points(self, x):
        return self.coeffs[0] * exp(self.coeffs[1] * x) + self.coeffs[2]

    def __init__(self, x, y, name_x="x", name_y="y"):
        super().__init__(2, x, y, name_x, name_y)

    def __str__(self):
        return f"{self.coeffs[0]:.2f}e^({self.coeffs[1]:.2f}x)+{self.coeffs[2]:.2f}".replace("+-", "-")


class OneDLogarithmicRegression(OneDPolynomialRegression):

    def _least_squares(self):
        coeffs = curve_fit(lambda x, a, b: a + b * log(x), self.x, self.y)[0]
        return [[coeffs[1]], [coeffs[0]]]

    def _function_points(self, x):
        return self.coeffs[0] + self.coeffs[1] * log(x)

    def __init__(self, x, y, name_x="x", name_y="y"):
        super().__init__(2, x, y, name_x, name_y)

    def __str__(self):
        return f"{self.coeffs[0]:.2f}+{self.coeffs[1]:.2f}*ln(x)".replace("+-", "-", 1)


class OneDExponencialRegression(OneDPolynomialRegression):

    def _least_squares(self):
        coeffs = curve_fit(lambda x, a, b, c: exp(a + b * x) + c, self.x, self.y)[0]
        return [[coeffs[2]], [coeffs[1]], [coeffs[0]]]

    def _function_points(self, x):
        return exp(self.coeffs[0] + self.coeffs[1] * x) + self.coeffs[2]

    def __init__(self, x, y, name_x="x", name_y="y"):
        super().__init__(2, x, y, name_x, name_y)

    def __str__(self):
        return f"exp({self.coeffs[0]:.2f}+{self.coeffs[1]:.2f}x)+{self.coeffs[2]:.2f}".replace("+-", "-")


class OneDPowerRegression(OneDPolynomialRegression):

    def _least_squares(self):
        coeffs = curve_fit(lambda x, a, b, c: a * (x ** b) + c, self.x, self.y)[0]
        return [[coeffs[2]], [coeffs[1]], [coeffs[0]]]

    def _function_points(self, x):
        return self.coeffs[0] * x ** self.coeffs[1] + self.coeffs[2]

    def __init__(self, x, y, name_x="x", name_y="y"):
        super().__init__(2, x, y, name_x, name_y)

    def __str__(self):
        return f"{self.coeffs[0]:.2f}x^({self.coeffs[1]:.2f})+{self.coeffs[2]:.2f}".replace("+-", "-", 1)


class OneDHyperbolaRegression(OneDPolynomialRegression):

    def _least_squares(self):
        coeffs = curve_fit(lambda x, a, b: a + b / x, self.x, self.y)[0]
        return [[coeffs[1]], [coeffs[0]]]

    def _function_points(self, x):
        return self.coeffs[0] + self.coeffs[1] / x

    def __init__(self, x, y, name_x="x", name_y="y"):
        super().__init__(2, x, y, name_x, name_y)

    def __str__(self):
        return f"{self.coeffs[0]:.2f}+{self.coeffs[1]:.2f}/x".replace("+-", "-", 1)


class OneDTrigonometricRegression(OneDPolynomialRegression):

    def _least_squares(self):
        coeffs = curve_fit(lambda x, a, b, c: a + b * sin(x) + c * x, self.x, self.y)[0]
        return [[coeffs[2]], [coeffs[1]], [coeffs[0]]]

    def _function_points(self, x):
        return self.coeffs[0] + self.coeffs[1] * sin(x) + self.coeffs[2] * x

    def __init__(self, x, y, name_x="x", name_y="y"):
        super().__init__(2, x, y, name_x, name_y)

    def __str__(self):
        return f"{self.coeffs[0]:.2f}+{self.coeffs[1]:.2f}sin(x)+{self.coeffs[2]:.2f}x".replace("+-", "-", 1)


class MultiplyDLinearRegression(AbstractMultiplyDRegression):

    __s: matrix

    def _leastsquares(self):
        atrix = matrix(append(self.data[0:-1], [[1 for _ in arange(self.data[0].size)]], axis=0))
        return (atrix * atrix.transpose()) ** -1 * atrix * matrix(self.data[-1]).transpose()

    @staticmethod
    def _tempname(vark, data):
        return [array(x, dtype=float64) for x in array(list(zip(*data)))[vark]]

    @staticmethod
    def _subarrays(num):
        from functools import reduce
        return reduce(lambda p, x: p + [subset | {x} for subset in p], list(range(num - 1)), [set()])[1:]

    def __solving(self, _atemp, _goodlst):
        self.variables = _goodlst[1] or [*_atemp, -1]
        self.data = self._tempname(self.variables, self.origdata)
        self.__n = self.data[0].size
        self.__m = len(self.data) - 1
        avy = self.data[-1].mean()
        coeffs = self._leastsquares()
        self.coeffs = array([coeffs.item(i) for i in arange(self.__m + 1)], dtype=float64)
        del coeffs
        self.f = array([npsum([self.coeffs[j] * self.data[j][i] for j in range(self.__m)]) + self.coeffs[-1]
                        for i in arange(self.__n)])
        syy = npsum(square(self.data[-1] - avy))
        self.__se = npsum(square(self.data[-1] - self.f))
        self.r2 = 1 - self.__se / syy
        self.r = sqrt(self.r2)
        self.adjR2 = 1 - (self.__se / (self.__n - self.__m - 1)) / (syy / (self.__n - 1))
        self.adjR2arr.append(self.adjR2)
        self.varsarr.append(self.variables)

    def _attempment(self, _bad=True, _goodlst=(False, False)):
        if not _bad:
            self.__solving(0, _goodlst)
            return self
        for _atemp in self._s:
            self.__solving(_atemp, _goodlst)
        return self

    def get_used_variables(self):
        return self.namesX[self.variables[:-1]]

    def get_unused_variables(self):
        used = self.get_used_variables()
        return [name_x for name_x in self.namesX if name_x not in used]

    def __init__(self, x, y, names_x, name_y="y", _tried=0):
        x = x.join(y)
        self.origdata = array(x)
        self.namesX = names_x
        self.nameY = name_y
        self.__k = len(self.origdata[0])
        self._s = self._subarrays(self.__k)
        self.adjR2arr = []
        self.varsarr = []
        self._attempment()
        self.bestadjR2arr = sorted(zip(self.adjR2arr, self.varsarr), key=itemgetter(0), reverse=True)[0:4]
        self._attempment(_bad=False, _goodlst=self.bestadjR2arr[0])
        del self.adjR2arr[-1], self.varsarr[-1], self._s
        self.reg_xy = None

    def best_three_adjR2(self):
        return list(map(lambda el: [el[0], el[1][0:-1]], self.bestadjR2arr[1:]))

    def __str__(self):
        return ('+'.join([f"{self.coeffs[i]:.2f}x{self.variables[i] + 1}" for i in arange(self.__m)]) +
                f"+{(self.coeffs[self.__m]):.2f}").replace("+-", '-')

    def correlation_f(self, alpha):
        self.f_table_all = self.get_f_table(alpha, self.__n - self.__m - 1, self.__m)
        self.f_fact_all = (self.r2 / (1 - self.r2)) * (self.__n - self.__m - 1) / self.__m
        atrix = matrix(append(self.data[0:-1], [[1 for _ in arange(self.data[0].size)]], axis=0))
        self.__s = (atrix * atrix.transpose()) ** -1
        s = self.__s
        se = self.__se
        self.f_table_each = self.get_f_table(alpha, self.__n - self.__m - 1, 1)
        self.f_fact_each = [(self.coeffs[i] ** 2 / s.item(i, i)) / (se / (self.__n - self.__m - 1))
                            for i in range(self.__m)]

    def is_norm(self):
        return self.f_fact_all > self.f_table_all

    def get_f_fact_each(self):
        return {f"x{var + 1}": f_fact_var for var, f_fact_var in zip(self.variables, self.f_fact_each)}

    def params_is_norm(self):
        return not npany(array(self.f_fact_each) < self.f_table_each)

    def points(self, p1, p2=None):
        if self.__k - 1 <= 2:
            regression_x = linspace(p1, p2, 15)
        else:
            regression_x = array(p1)
        coeffs = self.coeffs[:-1]
        if self.__k - 1 == 2:
            regression_x = meshgrid(*regression_x.T)
            regression_y = npsum([el1 * el2 for el1, el2 in zip(coeffs, regression_x)], axis=0) + self.coeffs[-1]
            self.reg_xy = {"x": regression_x, "y": regression_y}
        elif self.__k - 1 > 2:
            self.reg_xy = {"x": regression_x,
                           "y": npsum([el1 * el2 for el1, el2 in zip(coeffs, regression_x)], axis=0) + self.coeffs[-1]}
        else:
            self.reg_xy = {"x": regression_x.flatten(),
                           "y": (coeffs[0] * regression_x + self.coeffs[-1]).flatten()}
        return self

    def prediction(self):
        reg_x = self.reg_xy["x"]
        reg_y = self.reg_xy["y"]
        avx = array([self.data[i].mean() for i in range(self.__m)])
        if self.__k - 1 <= 2:
            D2 = array([[npsum(array([[(reg_x[i][k][m] - avx[i]) * (reg_x[j][k][m] - avx[j]) * self.__s.item(i, j)
                                       for i in range(self.__m)] for j in range(self.__m)])) * (self.__n - 1)
                         for k in range(15)] for m in range(15)])
        else:
            D2 = npsum(array([[(reg_x[i] - avx[i]) * (reg_x[j] - avx[j]) * self.__s.item(i, j)
                               for i in range(self.__m)] for j in range(self.__m)])) * (self.__n - 1)
        sigma_sq = self.__se / (self.__n - self.__m - 1)
        rv = sqrt(self.f_table_each * (1 / self.__n + D2 / (self.__n - 1)) * sigma_sq)
        self.rv_up, self.rv_down = reg_y + rv, reg_y - rv
        pred = sqrt(self.f_table_each * (1 + 1 / self.__n + D2 / (self.__n - 1)) * sigma_sq)
        self.pred_up, self.pred_down = reg_y + pred, reg_y - pred

    def approx_error(self):
        x = self._tempname(self.variables, self.origdata)[:-1]
        f = self.origdata.T[-1]
        indexes = argwhere(f != 0).flatten()
        coeffs = self.coeffs[:-1]
        y = npsum([el1 * el2 for el1, el2 in zip(coeffs, x)], axis=0) + self.coeffs[-1]
        f = f[indexes]
        return f"{mean(npabs((f - y[indexes]) / f)) * 100:.4f} %"

    def dim(self):
        return self.__m
