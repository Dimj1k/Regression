from src import *
from functools import partial
import tkinter as tk
import tkinter.filedialog as filedialog
from tkinter.messagebox import askyesno
from tkinter.ttk import Combobox as tkCombobox


class Main(tk.Tk):
    
    data: Data
    regression: Logic_Regression
    graph: Graph

    class LabelFrame(tk.LabelFrame):
        def __init__(self, frame, name, font, pack=None, **kwargs):
            super().__init__(frame, text=name, font=font, **kwargs)
            self.pack(pack or {})

    class Button(tk.Button):
        def __init__(self, frame, text, font, command, pack=None, **kwargs):
            super().__init__(frame, text=text, font=font, command=command, **kwargs)
            self.pack(pack or {"fill": tk.X})

    class Label(tk.Label):
        def __init__(self, frame, str_variable, font, pack=None, **kwargs):
            if type(str_variable) is not str:
                super().__init__(frame, textvariable=str_variable, font=font, **kwargs)
            else:
                super().__init__(frame, text=str_variable, font=font, **kwargs)
            self.pack(pack or {})

    class RadioButton(tk.Radiobutton):
        def __init__(self, frame, var, font, text, value, pack=None, **kwargs):
            super().__init__(frame, variable=var, text=text, value=value, font=font, **kwargs)
            self.pack(pack or {})

    class Combobox(tkCombobox):
        def __init__(self, frame, values, var, font, pack=None, **kwargs):
            super().__init__(frame, values=values, textvariable=var, font=font, **kwargs)
            self.pack(pack or {})

    class Frame(tk.Frame):
        def __init__(self, frame, pack=None, **kwargs):
            super().__init__(frame, **kwargs)
            self.pack(pack or {})

    class Entry(tk.Entry):
        def __init__(self, frame, var, font, pack=None, **kwargs):
            super().__init__(frame, font=font, textvariable=var, **kwargs)
            self.pack(pack or {})

    class TopLevel(tk.Toplevel):

        @staticmethod
        def __show_data(data):
            for idx, data in data:
                yield idx, str(data).rsplit("Name", maxsplit=1)[0], data.dtype not in ("object", "datetime64[ns]")

        @staticmethod
        def __create_checkbutton(main_frame_idx, idx, is_num):
            var = tk.IntVar()
            if is_num:
                label = tk.Checkbutton(main_frame_idx, variable=var)
                label.select()
                label.pack(fill=tk.BOTH, expand=1)
            else:
                tk.Frame(main_frame_idx, height=25).pack()
            return {"id": idx, "var": var}

        @staticmethod
        def __create_radio_button(main_frame_idx, idx, is_num, var):
            if is_num:
                tk.Radiobutton(main_frame_idx, variable=var, value=idx).pack()
                var.set(idx)
            else:
                tk.Frame(main_frame_idx, height=25).pack()

        def __init__(self, data, font):
            super().__init__()
            self.grab_set()
            self.resizable(False, False)
            self.wm_title("Выбор конкректных данных для регрессии")
            frame = tk.Frame(self)
            frame.pack()
            outside_frame = tk.Frame(frame)
            outside_frame.pack(side=tk.LEFT)
            tk.Frame(outside_frame, height=30).pack()
            tk.Label(outside_frame, text="Переменные", font=font).pack()
            tk.Frame(outside_frame, height=270).pack()
            tk.Label(outside_frame, text="Зависимая\nпеременная", font=font).pack()
            main_frame = tk.LabelFrame(frame, text="Выберите конкретные данные из файла", font=font)
            main_frame.pack(fill=tk.BOTH, expand=1, side=tk.LEFT)
            self.radio_variable = tk.StringVar()
            self.checkboxes = []
            for i, (idx, d, is_num) in enumerate(self.__show_data(data)):
                main_frame_idx = tk.LabelFrame(main_frame)
                main_frame_idx.pack(side=tk.LEFT, pady=1, padx=5)
                tk.Label(main_frame_idx, text=idx, font=font).pack()
                self.checkboxes.append(self.__create_checkbutton(main_frame_idx, idx, is_num))
                tk.Label(main_frame_idx, text=d, font=font, height=16).pack(expand=1)
                self.__create_radio_button(main_frame_idx, idx, is_num, self.radio_variable)
            tk.Button(self, text="Выбрать указанные столбцы", command=lambda: self.destroy()).pack(fill=tk.X)

        def choice_x_y(self):
            self.wait_window()
            choisen = [el["id"] for el in filter(lambda checkbox: checkbox["var"].get(), self.checkboxes)]
            return choisen, self.radio_variable.get()

    def __input_default(self, dont_update=False):
        list(map(lambda btn: btn.config(state=tk.NORMAL), self.__btns))
        if self.data.dim() >= 2:
            self.__functions_alpha["text"] = "Функция и уровень доверия"
            if self.is_packed(self.btns_functions[0]):
                list(map(self.__unvisible_packed_widget, self.btns_functions))
                self.__visible_packed_widget(self.__multiLinear, dict(expand=1))
        else:
            self.__functions_alpha["text"] = "Парные функции и уровень доверия"
            if not self.is_packed(self.btns_functions[0]):
                list(map(lambda x: self.__visible_packed_widget(x, dict(expand=1, side=tk.LEFT)), self.btns_functions))
                self.__unvisible_packed_widget(self.__multiLinear)
        if self.data.dim() > 2:
            self.__btns[2]["text"] = "Отобразить значение уравнения в интерфейсе"
            [[self.__unvisible_packed_widget(widget) for widget in widgets] for widgets in self.__points[1:]]
            self.__points[0][2].config(width=34)
            self.__points[0][0].pack_configure(padx=6)
            self.__points[0][1]["text"] = "X: "
            self.x0.set("")
            self.input_points["text"] = "Рассчитать значение в многомерной точке X={x1 x2 ... xn}:"
        else:
            if not self.is_packed(self.__points[1][0]):
                [[self.__visible_packed_widget(widget) for widget in widgets] for widgets in self.__points[1:]]
                self.__points[0][2].config(width=10)
                self.__points[0][0].pack_configure(padx=0)
            self.__points[0][1]["text"] = "x0:"
            self.__points[1][1]["text"] = "xk:"
            if self.data.dim() != 1:
                [self.__unvisible_packed_widget(widget) for widget in self.__points[2]]
                self.__btns[2]["text"] = "Отобразить значения уравнения в пространстве"
                self.input_points["text"] = "Рассчитать значения во множестве точек {x10 x20}, {x1k x2k}:"
            else:
                self.__btns[2]["text"] = "Отобразить значения уравнения на плоскости"
                self.input_points["text"] = "Рассчитать значения во множестве точек x1, xk с шагом:"
                if not self.is_packed(self.__points[2][0]):
                    [self.__visible_packed_widget(widget) for widget in self.__points[2]]
            if not dont_update:
                self.x0.set(" ".join(map(str, self.data.get_min_x())))
                self.xk.set(" ".join(map(str, self.data.get_max_x())))
        self.__set_info()

    @staticmethod
    def __visible_packed_widget(widget, pack=None):
        widget.pack(pack or dict(side=tk.LEFT))

    @staticmethod
    def __unvisible_packed_widget(widget):
        widget.pack_forget()

    def __set_info(self):
        self.answer_str = f"Файл: {self.data.filename}\n(Обозначение) - (Переменная в уравнении)\n"
        for key, el in self.data.names().items():
            self.answer_str += f"{el} - {key}\n"
        self.answer.set(self.answer_str.strip())

    def __path_to_file(self):
        self.path.set(filedialog.askopenfilename(**self.opts_for_filename) or self.path.get())
        if not self.path.get():
            return
        self.data = Data(self.path.get())
        regression_x, regression_y = self.TopLevel(self.data, self.font).choice_x_y()
        self.grab_set()
        regression_x = list(filter(lambda el: el != regression_y, regression_x))
        self.data.select_spec_x_y(regression_x, regression_y)
        self.__input_default()

    def __change_clrs_radio(self):
        func_choisen = self.func_var.get()
        if func_choisen == self.__old_func_var:
            return
        for i, radio_btn in enumerate(self.btns_functions):
            radio_btn["fg"] = "green" if i == func_choisen else "black"
        if self.__btns[0]["state"] == tk.DISABLED:
            if askyesno(title="Пересчитать уравнение регрессии",
                        message=f"Найти другое уравнение регрессии\n({self.functions_str[func_choisen]})?"):
                self.__input_default(True)
                self.__old_func_var = func_choisen
            else:
                for i, radio_btn in enumerate(self.btns_functions):
                    radio_btn["fg"] = "green" if i == self.__old_func_var else "black"
                self.func_var.set(self.__old_func_var)
        else:
            self.__old_func_var = func_choisen

    def __create_input_points_frames(self, text, var, **kwargs):
        frame = self.Frame(self.input_points, pack=dict(side=tk.LEFT))
        return (frame, self.Label(frame, text + ":", self.font, pack=dict(side=tk.LEFT)),
                self.Entry(frame, var, self.font, **kwargs))

    def __change_alpha(self, event):
        alpha = self.alpha.get()
        if self.__old_alpha == alpha:
            return
        if self.__btns[0]["state"] == tk.DISABLED:
            if askyesno(title="Пересчитать интервалы",
                        message=f"Вы хотите пересчитать доверительный интервал\n"
                                f"и прогноз при уровне доверия: {alpha}?"):
                self.__R_and_f(True)
                self.__old_alpha = alpha
            else:
                self.alpha.set(self.__old_alpha)
        else:
            self.__old_alpha = alpha

    @staticmethod
    def is_packed(frame):
        try:
            frame.pack_info()
            return True
        except tk.TclError:
            return False

    def geometry(self):
        screen_width = int(self.winfo_screenwidth() // 1.5)
        screen_height = int(self.winfo_screenheight() // 1.5)
        if screen_width > 1060 and screen_height > 700:
            super().geometry(f"{screen_width}x{screen_height}+"
                             f"{int(int(screen_width) * 1.5 // 5)}+{int(int(screen_height) * 1.5 // 6)}")
        else:
            super().geometry("1060x700+265+175")
        self.minsize(1060, 700)

    def __find_eq(self):
        if self.__btns[0]["state"] == tk.DISABLED:
            return
        x, y, Xnames, Yname = self.data.x, self.data.y, self.data.Xnames, self.data.Yname
        if self.data.dim() == 1:
            func = self.functions[self.func_var.get()]
            self.curve = func(x, y, Xnames, Yname)
            self.answer_str += f"Уравнение регресии: y={self.curve}\n"
        else:
            self.__old_answer_str_for_multiDim = ""
            self.curve = MultiplyDLinearRegression(x, y, Xnames, Yname)
            self.answer_str += f"Уравнение регресии: y={self.curve}\n"
            used_vars = self.curve.get_used_variables()
            if len(used_vars) != self.data.dim():
                self.answer_str += f"Вы можете исключить переменные: {', '.join(self.curve.get_unused_variables())}" \
                                   f" из данных" + \
                                   (" и найти парное уравнение регрессии" if len(used_vars) == 1 else "") + "\n"
            self.answer_str += f"Объясняющие переменные: {', '.join(used_vars)}\n"
            self.answer_str += f"Коэффициент корреляции: {self.curve.r:.4f}\n" \
                               f"Коэффициент детерминации: {self.curve.r2:.4f}\n"
            self.answer_str += f"Скорректированный коэффициент детерминации: {self.curve.adjR2:.4f}\n"
            best3adjr2 = self.curve.best_three_adjR2()
            names_x = self.curve.namesX
            if self.data.dim() > 2:
                self.answer_str += "Лучшие три варианта используемых объясняющих переменных" \
                                   " и их скорректированный R^2:\n"
            else:
                self.answer_str += "Остальные варианты используемых объясняющих переменных и " \
                                   "их скорректированный R^2:\n"
            for i, el in enumerate(best3adjr2, start=1):
                self.answer_str += f"{i} вариант: " + ", ".join(names_x[array(el[1])]) + f": {el[0]:.4f}\n"
        self.answer.set(self.answer_str.strip())
        self.__btns[0].config(state=tk.DISABLED)

    def __R_and_f(self, again=False):
        if not again and self.__btns[1]["state"] == tk.DISABLED:
            return
        if not again:
            self.__find_eq()
            self.__old_answer_str = self.answer_str[:]
        else:
            self.answer_str = self.__old_answer_str[:]
        sign = str(round(1 - float(self.alpha.get()), 2))
        if self.data.dim() == 1:
            self.curve.correlation_f(sign)
            self.answer_str += f"Коэффициент корреляции Пирсона: {self.curve.r:.4f}\n"
            if not self.curve.islinear:
                self.answer_str += f"Коэффициент кореляции: {self.curve.nonlin_r:.4f}\n"
                self.answer_str += f"Разница между корреляцией Пирсона: {self.curve.can_be_linear()}\n"
            self.answer_str += f"Коэффициент детерминации:" \
                               f" {self.curve.nonlin_r ** 2 if not self.curve.islinear else self.curve.r2:.4f}\n"
            self.answer_str += f"f-критерий: {self.curve.f_fact:.4f}\nf-критерий табличный: {self.curve.f_table:.2f}\n"
            self.answer_str += f"Средняя ошибка аппроксимации: {self.curve.approx_error()}\n"
            self.answer_str += "Уравнение регрессии статически" + (" " if self.curve.is_norm() else " не ")+"надежно\n"
        else:
            self.curve.correlation_f(sign)
            self.answer_str += f"Средняя ошибка аппроксимации: {self.curve.approx_error()}\n"
            self.answer_str += f"f-критерий: {self.curve.f_fact_all:.4f}\n" \
                               f"f-критерий табличный: {self.curve.f_table_all:.2f}\n"
            self.answer_str += f"f-критерий табличный для каждого параметра: {self.curve.f_table_each}\n"
            self.answer_str += "f-критерии по параметрам:\n"
            f_var_fact = self.curve.get_f_fact_each()
            self.answer_str += "x: " + "\t".join(f_var_fact.keys()) + "\n"
            self.answer_str += "f: " + "\t".join([f"{f:.2f}" if f < 10e5 else "inf" for f in f_var_fact.values()]) \
                               + "\n"
            curve_is_norm, params_is_norm = self.curve.is_norm(), self.curve.params_is_norm()
            if curve_is_norm and params_is_norm:
                self.answer_str += "Уравнение регрессии в целом надёжно"
            elif curve_is_norm and not params_is_norm:
                self.answer_str += "Уравнение регрессии надежно, но модель не значима по параметрам"
            elif not curve_is_norm and params_is_norm:
                self.answer_str += "Уравнение регрессии не надежно, но модель значима по параметрам"
            else:
                self.answer_str += "Уравнение регрессии в целом не надежно"
            self.answer_str += "\n"
        self.answer.set(self.answer_str.strip())
        self.__btns[1].config(state=tk.DISABLED)
        if self.data.dim() > 2:
            self.__old_answer_str_for_multiDim = self.answer_str[:]

    def __graph(self):
        self.__R_and_f()
        if self.data.dim() == 1:
            x0 = float(self.x0.get())
            xk = float(self.xk.get())
            step = float(self.step.get().replace(",", ".", 1))
            self.curve.points(x0, xk, step).prediction()
            graph = OneDGraph(self.curve)
            graph.show()
        else:
            if self.data.dim() <= 2:
                x0 = array(list(map(lambda x: float(x.replace(",", ".", 1)), self.x0.get().split(" "))))
                xk = array(list(map(lambda x: float(x.replace(",", ".", 1)), self.xk.get().split(" "))))
                self.curve.points(x0, xk).prediction()
                graph = MultiplyDGraph(self.curve)
                graph.show()
            else:
                x0 = array(list(map(lambda x: float(x.replace(",", ".", 1)), self.x0.get().split(" "))))
                self.curve.points(x0).prediction()
                reg_y = self.curve.reg_xy["y"]
                self.answer_str = self.__old_answer_str_for_multiDim[:]
                self.answer_str += "Значение уравнения в точке {" + self.x0.get() + "}: " + f"{reg_y:.4f}\n"
                self.answer_str += f"Доверительный интервал в точке: от {self.curve.rv_down:.4f} до" \
                                   f" {self.curve.rv_up:.4f}\n"
                self.answer_str += f"Прогнозируемый интервал в точке: от {self.curve.pred_down:.4f} до" \
                                   f" {self.curve.pred_up:.4f}\n"
                self.answer.set(self.answer_str.strip())

    def __init__(self):
        super().__init__()
        self.geometry()
        file_types_supports = {"Поддерживающиеся файлы": "*.xlsx *.xls *.ods *.csv",
                               "Электронные таблицы": "*.xlsx *.xls *.ods", "csv": "*.csv"}.items()
        self.opts_for_filename = dict(filetypes=file_types_supports, title="Загрузка файла")
        self.functions = (partial(OneDPolynomialRegression, 1),
                          partial(OneDPolynomialRegression, 2),
                          partial(OneDPolynomialRegression, 3),
                          OneDExponentRegression,
                          OneDLogarithmicRegression,
                          OneDExponencialRegression,
                          OneDPowerRegression,
                          OneDHyperbolaRegression,
                          OneDTrigonometricRegression,
                          )
        self.__old_answer_str_for_multiDim = ""
        self.functions_str = ("y=ax+b", "y=ax^2+bx+c", "y=ax^3+bx^2+cx+d", "y=a*exp(b)", "y=a+b*ln(x)",
                              "y=exp(a+bx)", "y=exp(ax^b)", "y=a+b/x", "y=a+b*sin(x)+cx")
        self.__old_func_var = 0
        self.font = ("Times New Roman", 11)
        self.path = tk.StringVar()
        self.title("Регрессионный анализ")
        path_to_file_label = self.LabelFrame(self, "Путь до файла", self.font, pack=dict(fill=tk.X))
        self.Button(path_to_file_label, "Путь до файла", self.font, self.__path_to_file, pack=dict(side=tk.RIGHT))
        self.Label(path_to_file_label, self.path, self.font, bg="#ffffff",
                   pack=dict(side=tk.LEFT, padx=5, pady=5), width=self.winfo_screenwidth())
        self.__functions_alpha = self.LabelFrame(self, "Парные функции и уровень доверия", self.font,
                                                 pack=dict(fill=tk.X))
        self.func_var = tk.IntVar(value=0)
        self.btns_functions = [self.RadioButton(self.__functions_alpha, self.func_var, self.font, func_str, i,
                                                pack=dict(side=tk.LEFT, expand=1), command=self.__change_clrs_radio)
                               for i, func_str in enumerate(self.functions_str)]
        self.__multiLinear = self.Label(self.__functions_alpha, "Множественная линейная регрессия", self.font)
        self.__unvisible_packed_widget(self.__multiLinear)
        self.btns_functions[0]["fg"] = "green"
        alphas = ["0.9", "0.95", "0.98", "0.99"]
        self.alpha = tk.StringVar(value=alphas[0])
        self.__old_alpha = alphas[0]
        alpha_frame = self.Frame(self.__functions_alpha, pack=dict(side=tk.RIGHT))
        self.Label(alpha_frame, "Дов. вероятность", self.font)
        self.Combobox(alpha_frame, alphas, self.alpha, self.font, width=4, state="readonly")\
            .bind("<<ComboboxSelected>>", self.__change_alpha)
        frame_regression = self.Frame(self, pack=dict(pady=1, fill=tk.BOTH, anchor=tk.N))
        self.answer_str = "Загрузите файл в приложение"
        self.__old_answer_str = ""
        self.answer = tk.StringVar(value=self.answer_str)
        info_regression = self.LabelFrame(frame_regression, "Информация об уравнении регрессии", self.font,
                                          pack=dict(side=tk.LEFT, fill=tk.Y, anchor=tk.NW))
        self.Label(info_regression, self.answer, self.font, pack=dict(fill=tk.Y, anchor=tk.W), justify=tk.LEFT)
        input_frame = self.Frame(frame_regression, pack=dict(side=tk.RIGHT, anchor=tk.NE))
        self.input_points = self.LabelFrame(input_frame, "Рассчитать значение во множестве точек x1, xk с шагом:",
                                            self.font)
        self.x0, self.xk, self.step = tk.StringVar(), tk.StringVar(), tk.StringVar(value="0.1")
        self.__points = [self.__create_input_points_frames("x0", self.x0, width=10),
                         self.__create_input_points_frames("xk", self.xk, width=10),
                         self.__create_input_points_frames("Шаг", self.step, width=6)]
        btns_frame = self.Frame(input_frame, pack=dict(fill=tk.X))
        self.__btns = [self.Button(btns_frame, "Найти уравнение регрессии", self.font, self.__find_eq),
                       self.Button(btns_frame, "Коэффициент корреляции и критерий Фишера", self.font, self.__R_and_f),
                       self.Button(btns_frame, "Отобразить значения уравнения на плоскости", self.font, self.__graph)]
        self.Button(btns_frame, "Выйти из приложения", self.font, lambda: self.destroy(),
                    pack=dict(fill=tk.X))
        footer = self.Frame(self, bg="#ffffff", pack=dict(side=tk.BOTTOM, fill=tk.X))
        self.Label(footer, "Прикладная математика и информатика 2 курс. Тамбов 2023\n"
                           "https://github.com/Dimj1k/Regression/tree/CourseWorkCode\n"
                           "Саратов Дмитрий Александрович. Малютин Кирилл Александрович. Ожогин Дмитрий Александрович",
                   self.font, justify=tk.RIGHT, bg="#ffffff", pack=dict(side=tk.RIGHT))
        self.mainloop()


Main()
