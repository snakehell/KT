import numpy as np
import math
import json


class Functions:

    @staticmethod
    def functions_t0(y, LS, EpSC, A, c):
        return Functions.functions(y, 0, LS, EpSC, A, c)

    @staticmethod
    def functions(y, t, LS, EpSC, A, c):
        f0 = (- LS[0][1] * (y[0] - y[1]) - EpSC[0] * (y[0] / 100) ** 4) / c[0]
        f1 = (- LS[1][0] * (y[1] - y[0]) - LS[1][2] * (y[1] - y[2]) - EpSC[1] * (y[1] / 100) ** 4) / c[1]
        f2 = (- LS[2][1] * (y[2] - y[1]) - LS[2][3] * (y[2] - y[3]) - EpSC[2] * (y[2] / 100) ** 4) / c[2]
        f3 = (- LS[3][2] * (y[3] - y[2]) - LS[3][4] * (y[3] - y[4]) - EpSC[3] * (y[3] / 100) ** 4) / c[3]
        f4 = (- LS[4][3] * (y[4] - y[3]) - EpSC[4] * (y[4] / 100) ** 4) / c[4]
        f4 += A * (20 + 3 * math.sin(t / 4)) / c[0]
        return [f0, f1, f2, f3, f4]

    @staticmethod
    def coef_init(filepath):
        with open(filepath) as f:
            coefs = json.load(f)
        for name, value in coefs.items():
            if name == "c":
                c = value
            elif name == "eps":
                eps = value
            elif name == "lambda":
                lambd = value
            else:
                print("ERROR in json file", filepath)
        return c, eps, lambd

    @staticmethod
    def temp_init(filepath):
        with open(filepath) as f:
            coefs = json.load(f)
        for name, value in coefs.items():
            if name == "T":
                T = value
            else:
                print("ERROR in json file", filepath)
        return T

    @staticmethod
    def time_init(filepath):
        with open(filepath) as f:
            coefs = json.load(f)
        for name, value in coefs.items():
            if name == "time_interval":
                t = np.linspace(0, value, 1001)
                v = value
            else:
                print("ERROR in json file", filepath)
        return t, v

