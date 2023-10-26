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
    _names = {'response': FunctionData('t, s', 'y', 'Response to Input Signal'),
              'step_response': FunctionData('t, s', 'y', 'Step Response'),
              'delta_response': FunctionData('t, s', 'y', 'Delta Response'),
              're_freq': FunctionData('w, rad/s', 'Re H(jw)', 'Real part of H(jw)'),
              'im_freq': FunctionData('w, rad/s', 'Im H(jw)', 'Imaginary part of H(jw)'),
              'magnitude_freq': FunctionData('w, rad/s', '|H(jw)|', 'Magnitude-Frequency Response'),
              'phase_freq': FunctionData('w, rad/s', 'arg(H(jw)), rad', 'Phase-Frequency Response'),
              'log_magnitude': FunctionData('w, rad/s', 'dB', 'Logarithmic Magnitude-Frequency Response')
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

    def plot(self, function: Callable[[], sp.Expr], plot_range: tuple[float, float] = (0, 10),
             log_scale: bool | None = None,
             func_to_compare_with: Callable[[float | np.ndarray], float | np.ndarray] | None = None,
             title: str | None = None) -> None:
        if log_scale is None and function.__name__ == 'log_magnitude':
            log_scale = True
        assert isinstance(function(), sp.Expr), 'We must have sympy.Expr returning functions'
        if not hasattr(self, function.__name__):
            raise AttributeError('Invalid function passed. You can only pass System class methods to plot')
        fdata = self._names.get(function.__name__)
        assert len(function().free_symbols) == 1, 'We cannot have more than one variable here'
        l_func = sp.lambdify(fdata.xlabel[0], function(), 'numpy')
        x = np.linspace(*plot_range, 1000)
        plt.plot(x, l_func(x), label='Response through transfer function')
        if log_scale:
            plt.xscale('log')

            # TODO: implement proper yticks including -3 and -17 and other for other cases for log_magnitude

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

    def __nyquist(self) -> tuple[
        Callable[[float | np.ndarray], float | np.ndarray],
        Callable[[float | np.ndarray], float | np.ndarray],
    ]:
        return sp.lambdify(self._w, self.re_freq(), 'numpy'), sp.lambdify(self._w, self.im_freq(), 'numpy')

    def plot_nyquist(self, freq_range: tuple[float, float] = (0, 20), title: str | None = None) -> None:
        w = np.linspace(*freq_range, 10000)
        plt.plot(self.__nyquist()[0](w), self.__nyquist()[1](w))
        plt.title(title) if title is not None else plt.title('Nyquist')
        plt.xlabel('Re H(jw)')
        plt.ylabel('Im H(jw)')
        plt.grid()
        plt.show()
