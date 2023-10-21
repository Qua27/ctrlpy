import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from functools import wraps
from typing import Callable
from collections import namedtuple


def simplifier(func: Callable[..., sp.Expr]) -> Callable[..., sp.Expr]:
    @wraps(func)
    def wrapper(*args, **kwargs) -> sp.Expr:
        res = func(*args, **kwargs)
        return sp.simplify(res)

    return wrapper


FunctionData = namedtuple('FunctionData', ['xlabel', 'ylabel', 'title'])


class System:
    _s = sp.symbols('s')
    _t, _w = sp.symbols('t w', real=True)
    _names = {'response': FunctionData('t', 'y', 'Response to Input Signal'),
              'step_response': FunctionData('t', 'y', 'Step Response'),
              'delta_response': FunctionData('t', 'y', 'Delta Response'),
              're_freq': FunctionData('w', 'Re H(jw)', 'Real part of H(jw)'),
              'im_freq': FunctionData('w', 'Im H(jw)', 'Imaginary part of H(jw)'),
              'magnitude_freq': FunctionData('w', '|H(jw)|', 'Magnitude-Frequency Response'),
              'phase_freq': FunctionData('w', 'arg(H(jw))', 'Phase-Frequency Response'),
              'log_magnitude': FunctionData('w', 'dB', 'Logarithmic Magnitude-Frequency Response')
              }

    def __init__(self, tf: str):
        self.tf = sp.sympify(tf, locals={'s': self._s})

    def __repr__(self) -> str:
        return f'{{System with transfer function: {self.tf}}}'

    @simplifier
    def response(self, inp_signal_str: str) -> sp.Expr:
        inp_signal = sp.sympify(inp_signal_str, locals={'t': self._t})
        U = sp.laplace_transform(inp_signal, self._t, self._s)[0]
        Y = U * self.tf
        return sp.inverse_laplace_transform(Y, self._s, self._t)

    @simplifier
    def step_response(self) -> sp.Expr:
        return self.response('1')

    @simplifier
    def delta_response(self) -> sp.Expr:
        return self.response('DiracDelta(t)')

    def plot(self, function: Callable[[], sp.Expr], plot_range: tuple[float, float] = (-10, 10),
             func_to_compare_with: Callable[[float | np.ndarray], float | np.ndarray] = None,
             title: str = None) -> None:
        if not hasattr(self, function.__name__):
            raise AttributeError('Invalid function passed. You can only pass System class methods to plot')
        fdata = self._names.get(function.__name__)
        assert len(function().free_symbols) == 1, 'We cannot have more than one variable here'
        l_func = sp.lambdify(fdata.xlabel, function(), 'numpy')
        x = np.linspace(*plot_range, 1000)
        plt.plot(x, l_func(x), label='Response through transfer function')
        if func_to_compare_with is not None:
            plt.plot(x, func_to_compare_with(x), label='Analytical response')
            plt.legend()
        plt.xlabel(fdata.xlabel)
        plt.ylabel(fdata.ylabel)
        plt.grid()
        if title is not None:
            plt.title(title)
        else:
            plt.title(fdata.title)
        plt.show()

    @simplifier
    def freqs(self) -> sp.Expr:
        H_jw = self.tf.subs({self._s: sp.I * self._w})
        return H_jw

    @simplifier
    def re_freq(self) -> sp.Expr:
        return sp.re(self.freqs())

    @simplifier
    def im_freq(self) -> sp.Expr:
        return sp.im(self.freqs())

    @simplifier
    def magnitude_freq(self) -> sp.Expr:
        return sp.sqrt(self.re_freq() ** 2 + self.im_freq() ** 2)

    @simplifier
    def phase_freq(self) -> sp.Expr:
        return sp.atan2(self.im_freq(), self.re_freq())

    @simplifier
    def log_magnitude(self) -> sp.Expr:
        return 20 * sp.log(self.magnitude_freq()) / sp.log(10)

    def __nyquist(self):
        return sp.lambdify(self._w, self.re_freq(), 'numpy'), sp.lambdify(self._w, self.im_freq(), 'numpy')
