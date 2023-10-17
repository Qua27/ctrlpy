import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


class System:
    t, s = sp.symbols('t s')

    def __init__(self, tf):
        self.tf = sp.sympify(tf)

    def response(self, inp_signal_str):
        inp_signal = sp.sympify(inp_signal_str)
        U = sp.laplace_transform(inp_signal, self.t, self.s)[0]
        Y = U * self.tf
        return sp.inverse_laplace_transform(Y, self.s, self.t)

    def plot_response(self, inp_signal, function_to_compare_with=None):
        s = np.linspace(0, 20, 1000)
        out_signal = sp.lambdify(self.t, self.response(sp.sympify(inp_signal)), 'numpy')
        plt.figure()
        plt.plot(s, out_signal(s), label='Response through transfer function')
        if function_to_compare_with is not None:
            plt.plot(s, function_to_compare_with(s), label='Analytical response')
        plt.xlabel('t, sec')
        plt.ylabel('y')
        plt.legend()
        plt.grid()
        plt.show()
