import matplotlib.pyplot as plt
from .abstracts import Graph


class OneDGraph(Graph):

    def __init__(self, curve):
        self.pt_curve = curve
        self.x_real, self.y_real = curve.x, curve.y
        self.x_reg, self.y_reg = curve.regression_x, curve.regression_y

    def show(self):
        fig = plt.figure()
        plt.plot(self.x_real, self.y_real, ".", color="k", label="Ориг.")
        plt.plot(self.x_reg, self.y_reg, color="tomato", label=f"{self.pt_curve}")
        if id(self.pt_curve.pred_up) != id(self.y_reg):
            plt.fill_between(self.x_reg, self.pt_curve.pred_up, self.pt_curve.pred_down, alpha=0.5, color="lime",
                             label="Доверительный интервал")
            plt.fill_between(self.x_reg, self.pt_curve.rv_up, self.pt_curve.rv_down, alpha=0.3,
                             color="aqua", label="Прогноз")
        names = self.pt_curve.names()
        plt.xlabel(names["x"])
        plt.ylabel(names["y"])
        plt.legend()
        fig.show()


class MultiplyDGraph(Graph):

    def __init__(self, curve):
        self.pt_curve = curve
        self.origdata = curve.origdata
        self.variables = curve.variables
        self.x_reg, self.y_reg = curve.regression_xy["x"], curve.regression_xy["y"]

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(*self.origdata.T, color='k', label="Ориг.")
        ax.plot_wireframe(*self.x_reg, self.y_reg, label=f"{self.pt_curve}", color="tomato")
        l1 = ax.plot_surface(*self.x_reg, self.pt_curve.rv_up, color="lime", alpha=0.5, label="Доверительный интервал")
        ax.plot_surface(*self.x_reg, self.pt_curve.rv_down, color="lime", alpha=0.5)
        ax.plot_surface(*self.x_reg, self.pt_curve.pred_up, color="aqua", alpha=0.4)
        l2 = ax.plot_surface(*self.x_reg, self.pt_curve.pred_down, color="aqua", alpha=0.4, label="Прогноз")
        names = self.pt_curve.get_used_variables()
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1] if len(names) == 2 else self.pt_curve.get_unused_variables()[0])
        ax.set_zlabel(self.pt_curve.nameY)
        l1._edgecolors2d, l1._facecolors2d = l1._edgecolor3d, l1._facecolor3d
        l2._edgecolors2d, l2._facecolors2d = l2._edgecolor3d, l2._facecolor3d
        ax.legend()
        plt.show()
