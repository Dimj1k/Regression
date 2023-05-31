from pandas import (
    read_excel,
    read_csv,
    set_option
)
from pathlib import Path
from .abstracts import AbstractData


class Data(AbstractData):

    def __init__(self, io):
        set_option("display.max_rows", 20)
        set_option("display.min_rows", 15)
        self.io = Path(io)
        self.filename = self.io.name
        format_file = self.io.suffix
        if format_file in (".xlsx", ".ods", ".xls"):
            self.data = read_excel(io, sheet_name=0)
        elif format_file == ".csv":
            self.data = read_csv(io, encoding="utf-8")
        else:
            raise TypeError
        self.data.index = [""] * len(self.data)
        self.allNames = self.data.columns
        self.allData = self.data[self.allNames]

    def dim(self):
        return len(self.Xnames)

    def get_min_x(self):
        return self.x.min(axis=0)

    def get_max_x(self):
        return self.x.max(axis=0)

    def select_spec_x_y(self, choisen_x, choisen_y):
        self.data.index = list(range(len(self.data)))
        data = self.data[choisen_x]
        self.Yname = self.data[choisen_y].name
        self.Xnames = data.columns
        self.x, self.y = data[self.Xnames], self.data[self.Yname]
        del self.allData, self.allNames

    def names(self):
        if self.dim() > 1:
            return {**{f"x{i}": el for i, el in enumerate(self.Xnames, start=1)}, "y": self.Yname}
        else:
            return {"x": self.Xnames[0], "y": self.Yname}

    def __getitem__(self, item):
        return self.allNames[item], self.data[self.allNames[item]]

    def __len__(self):
        return len(self.data.columns)
