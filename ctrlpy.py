"""
This module is designed for solving control theory and signal theory basic problems.
"""

from functools import wraps
from typing import Callable
from collections import namedtuple

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy import signal

PlotData = namedtuple('PlotData', ['xlabel', 'ylabel', 'log_scale', 'title'])


def simplifier(func: Callable[..., sp.Expr]) -> Callable[..., sp.Expr]:
    """
    Decorator to simplify the result of a function that returns a SymPy expression.

    This decorator takes a function that returns a SymPy expression, applies
    SymPy's `simplify` function to the result, and returns the simplified expression.

    :param func: The function that returns a SymPy expression.

    :return: A decorated function that returns a simplified SymPy expression.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> sp.Expr:
        res = func(*args, **kwargs)
        return sp.simplify(res)

    return wrapper


def process(expr: sp.Expr, s: sp.Symbol) -> sp.Expr:
    """
    Process a SymPy expression representing a transfer function.

    This function simplifies and separates the given expression into its numerator and denominator.

    :param expr: The expression representing the transfer function.

    :param s: The symbol used in the transfer function.

    :return: The processed transfer function with its numerator and denominator separated.
    """

    simple = sp.simplify(expr)
    num = sp.collect(sp.expand(sp.numer(simple)), s)
    den = sp.collect(sp.expand(sp.denom(simple)), s)
    return num / den


class ExprMissingVariableError(ValueError):
    """
    Custom exception for cases where a SymPy expression is missing a required variable.

    :param variable_name: The variable that is missing from the expression.
    """

    def __init__(self, variable_name: sp.Symbol):
        """
        Initialize the exception.

        :param variable_name: The variable that is missing from the expression.
        """

        super().__init__(f'The expression must contain {variable_name}.')


class InvalidSymbolError(ValueError):
    """
    Exception raised when a symbol variable is invalid.

    This exception is raised when a symbol variable is considered invalid for use in the
    `System.from_string` method. A valid symbol variable must contain exactly one character
    and be an instance of either the `sympy.Symbol` class or a string of length 1.

    :param message: The optional custom error message. If not provided, a default message
                    is used.
    """

    def __init__(self, message: str | None = None):
        if message is None:
            message = 'A valid symbol variable must contain exactly one character ' \
                      'and be an instance of either the sympy.Symbol class or a string of length 1.'
        super().__init__(message)


class System:
    """
    Represents a linear time-invariant (LTI) system with a transfer function.

    This class encapsulates the properties and behavior of an LTI system, including its transfer function and various
    operations related to signal processing and control systems.

    :param tf: The transfer function of the system.

    Attributes:
        _s (sympy.Symbol): The Laplace variable 's' used in the transfer function.

        _t (sympy.Symbol): The time variable 't' used for time-domain signal representations.

        _w (sympy.Symbol): The angular frequency variable 'w' used in frequency domain operations.

    Methods:
        __init__(tf): Initialize the System with a transfer function.

        from_string(string, symbol): Create a System from a string representation.

        __repr__(): Return a string representation of the System.

        __str__(): Return a human-readable string representation of the System.

        __eq__(other): Check if two Systems are equal.

        tf_coefs: Get the coefficients of the transfer function as NumPy arrays.

        lti: Get the scipy.signal.lti instance of the System.

        response(inp_signal): Calculate the system's response to an input signal.

        step_response(): Calculate the system's step response.

        delta_response(): Calculate the system's delta (Dirac delta) response.

    Note: This class is primarily intended for symbolic math and control systems applications.
    """

    _s: sp.Symbol
    _t: sp.Symbol
    _w: sp.Symbol

    _s = sp.symbols('s')
    _t, _w = sp.symbols('t w', real=True)

    def __init__(self, tf: sp.Expr):
        """
        Initialize the System with a transfer function.

        :param tf: The transfer function of the system.
        :raises ExprMissingVariableError: If the Laplace variable 's' is not found in the transfer function.
        """

        if self._s not in tf.free_symbols:
            raise ExprMissingVariableError(self._s)
        self.tf = process(tf, self._s)

    @classmethod
    def from_string(cls, string: str, symbol: sp.Symbol | str) -> 'System':
        """
        Create a System from a string representation of the transfer function.

        This class method allows the creation of a System instance from a string representation of a transfer function.
        The string representation should use the specified symbol as a placeholder for the Laplace variable 's' in the
        transfer function.

        :param string: A string representation of the transfer function.
        :param symbol: The symbol variable representing the Laplace variable 's'.
        :return: A System instance with the transfer function.
        :raises InvalidSymbolError: If the symbol variable is invalid.
        """

        if isinstance(symbol, sp.Symbol) or isinstance(symbol, str) and len(symbol) == 1:
            return cls(process(sp.sympify(string, locals={symbol: cls._s}), cls._s))
        raise InvalidSymbolError

    def __repr__(self) -> str:
        """
        Return a string representation of the System.
        """

        return f'System({self.tf})'

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the System.
        """

        return f'{{System with transfer function: {str(self.tf)}}}'

    def __eq__(self, other: object) -> bool:
        """
        Check if two Systems are equal.

        :param other: The other System to compare with.
        :return: True if the Systems are equal, False if the Systems are not equal
                 and NotImplemented if comparing System with another type.
        """

        if isinstance(other, System):
            return self.tf == other.tf
        return NotImplemented

    @property
    def tf_coefs(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the coefficients of the transfer function as NumPy arrays.

        :return: A tuple containing NumPy arrays of numerator and denominator coefficients.
        """

        num, den = sp.fraction(self.tf)
        num_coeffs = np.array(sp.Poly(num, self._s).all_coeffs(), dtype=float)
        den_coeffs = np.array(sp.Poly(den, self._s).all_coeffs(), dtype=float)
        return num_coeffs, den_coeffs

    @property
    def lti(self) -> signal.lti:
        """
        Get the scipy.signal.lti instance of the System.

        :return: The scipy.signal.lti instance.
        """

        return signal.lti(*self.tf_coefs)

    @simplifier
    def response(self, inp_signal: sp.Expr | str) -> sp.Expr:
        """
        Calculate the system's response to an input signal.

        :param inp_signal: The input signal as a symbolic expression or a string representation.
        :type inp_signal: sympy.Expr | str
        :return: The symbolic expression representing the system's response.
        """

        if isinstance(inp_signal, str):
            inp_signal = sp.sympify(inp_signal, locals={'t': self._t})
        u = sp.laplace_transform(inp_signal, self._t, self._s)[0]
        y = u * self.tf
        return sp.inverse_laplace_transform(y, self._s, self._t)

    @simplifier
    def step_response(self) -> sp.Expr:
        """
        Calculate the system's step response.

        :return: The symbolic expression representing the system's step response.
        """

        return self.response('1')

    @simplifier
    def delta_response(self) -> sp.Expr:
        """
        Calculate the system's delta (Dirac delta) response.

        :return: The symbolic expression representing the system's delta response.
        """

        return self.response('DiracDelta(t)')


@simplifier
def s_freqs(sy: System) -> sp.Expr:
    """
    Calculate the symbolic frequency response of a system in terms of a frequency variable 'w'.

    This function computes the symbolic frequency response of a given system in terms of a frequency variable 'w'.
    It replaces the Laplace variable 's' with 'jw', where 'j' is the imaginary unit,
    and then returns the symbolic frequency response expression.

    :param sy: The system for which to calculate the symbolic frequency response.
    :return: The symbolic frequency response expression in terms of the complex frequency variable 'w'.
    """

    w = sp.symbols('w', real=True)
    h_jw = sy.tf.subs({'s': sp.I * w})
    return h_jw


def create_plot_data(xlabel: str, ylabel: str, log_scale: bool, title: str) -> PlotData:
    """
    Create a `PlotData` instance with the specified plot settings.

    :param xlabel: The label for the x-axis.
    :param ylabel: The label for the y-axis.
    :param log_scale: Whether to use a logarithmic scale for the plot.
    :param title: The title of the plot.

    :return: A `PlotData` instance representing the plot settings.
    """

    return PlotData(xlabel, ylabel, log_scale, title)


def plot(
        sy: System,
        function: Callable[[System, int | np.ndarray | None], tuple[np.ndarray, np.ndarray]],
        plot_range: tuple[float, float],
        plot_data: PlotData,
        func_to_compare_with: Callable[[float | np.ndarray], float | np.ndarray] | None = None,
):
    """
    Plot the response of a system using the specified function.

    :param sy: The system for which to plot the response.
    :param function: A function that calculates the response given the system and a frequency range.
    :param plot_range: A tuple specifying the start and end values of the x-axis.
    :param plot_data: A `PlotData` instance containing plot settings, including labels and scale information.
    :param func_to_compare_with: A function to compare with the numerical plot (optional).

    **Note**:

    - If `func_to_compare_with` is provided, both the numerical and analytical plots will be displayed with a legend.

    - When `plot_data.log_scale` is `True`, the x-axis will be displayed in a logarithmic scale.

    - TODO: Implement proper yticks, including -3 and -17, for log_magnitude and other cases.
    """

    x = np.linspace(*plot_range, 1000)
    plt.plot(*function(sy, x), label='Numerical')
    if func_to_compare_with is not None:
        plt.plot(x, func_to_compare_with(x), label='Analytical')
        plt.legend()
    if plot_data.log_scale:
        plt.xscale('log')
    plt.xlabel(plot_data.xlabel)
    plt.ylabel(plot_data.ylabel)
    plt.grid()
    plt.title(plot_data.title)

    plt.show()


@simplifier
def s_re_freq(sy: System) -> sp.Expr:
    """
    Calculate the real part of the symbolic frequency response for a system.

    :param sy: The system for which to calculate the real part of the symbolic frequency response.
    :return: The symbolic expression representing the real part of the frequency response.
    """

    return sp.re(s_freqs(sy))


@simplifier
def s_im_freq(sy: System) -> sp.Expr:
    """
    Calculate the imaginary part of the symbolic frequency response for a system.

    :param sy: The system for which to calculate the imaginary part of the symbolic frequency response.
    :return: The symbolic expression representing the imaginary part of the frequency response.
    """

    return sp.im(s_freqs(sy))


@simplifier
def s_magnitude_freq(sy: System) -> sp.Expr:
    """
    Calculate the magnitude of the symbolic frequency response for a system.

    :param sy: The system for which to calculate the magnitude of the symbolic frequency response.
    :return: The symbolic expression representing the magnitude of the frequency response.
    """

    return sp.sqrt(s_re_freq(sy) ** 2 + s_im_freq(sy) ** 2)


@simplifier
def s_phase_freq(sy: System) -> sp.Expr:
    """
    Calculate the phase of the symbolic frequency response for a system.

    :param sy: The system for which to calculate the phase of the symbolic frequency response.
    :return: The symbolic expression representing the phase of the frequency response.
    """

    return sp.atan2(s_im_freq(sy), s_re_freq(sy))


@simplifier
def s_log_magnitude(sy: System) -> sp.Expr:
    """
    Calculate the logarithm of the magnitude of the symbolic frequency response for a system.

    :param sy: The system for which to calculate the logarithm of the magnitude of the symbolic frequency response.
    :return: The symbolic expression representing the logarithm of the magnitude of the frequency response.
    """

    return 20 * sp.log(s_magnitude_freq(sy)) / sp.log(10)


def n_freqs(sy: System, w: int | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate frequency response data for a system.

    This function computes the frequency response data for a given system in a complex form.

    :param sy: The system for which to calculate the frequency response.
    :param w: The angular frequencies at which to calculate the response. If None, the default range is used.
    :return: A tuple containing the angular frequencies and frequency response magnitudes.
    """

    return signal.freqs(*sy.tf_coefs, worN=w)


def n_re_freq(sy: System, w: int | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the real part of the frequency response data for a system.

    This function computes the real part of the frequency response data for a given system.

    :param sy: The system for which to calculate the real part of the frequency response.
    :param w: The angular frequencies at which to calculate the response. If None, the default range is used.
    :return: A tuple containing the angular frequencies and the real part of the frequency response magnitudes.
    """

    w, h = n_freqs(sy, w)
    return w, np.real(h)


def n_im_freq(sy: System, w: int | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the imaginary part of the frequency response data for a system.

    This function computes the imaginary part of the frequency response data for a given system.

    :param sy: The system for which to calculate the imaginary part of the frequency response.
    :param w: The angular frequencies at which to calculate the response. If None, the default range is used.
    :return: A tuple containing the angular frequencies and the imaginary part of the frequency response magnitudes.
    """

    w, h = n_freqs(sy, w)
    return w, np.imag(h)


def n_magnitude_freq(sy: System, w: int | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the magnitude-frequency response data for a system.

    This function computes the magnitude of the frequency response data for a given system.

    :param sy: The system for which to calculate the magnitude of the frequency response.
    :param w: The angular frequencies at which to calculate the response. If None, the default range is used.
    :return: A tuple containing the angular frequencies and the magnitude of the frequency response magnitudes.
    """

    w, h = n_freqs(sy, w)
    return w, np.abs(h)


def n_phase_freq(sy: System, w: int | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the phase-frequency response data for a system.

    This function computes the phase of the frequency response data for a given system.

    :param sy: The system for which to calculate the phase of the frequency response.
    :param w: The angular frequencies at which to calculate the response. If None, the default range is used.
    :return: A tuple containing the angular frequencies and the phase of the frequency response magnitudes.
    """

    w, h = n_freqs(sy, w)
    return w, np.angle(h)


def n_log_magnitude_freq(sy: System, w: int | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the logarithmic magnitude response data for a system.

    This function computes the logarithmic magnitude response data for a given system.

    :param sy: The system for which to calculate the logarithm of the magnitude of the frequency response.
    :param w: The angular frequencies at which to calculate the response. If None, the default range is used.
    :return: A tuple containing the angular frequencies and the logarithm of the magnitude of the frequency response
             magnitudes.
    """

    w, h = n_freqs(sy, w)
    return w, 20 * np.log10(np.abs(h))


def __nyquist(sy: System, w: int | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Nyquist plot data for a system.

    This function computes the Nyquist plot data for a given system by separately calculating
    the real and imaginary parts of the frequency response at specified angular frequencies.

    :param sy: The system for which to calculate the Nyquist plot data.
    :param w: The angular frequencies at which to calculate the response. If None, the default range is used.
    :return: A tuple containing the real and imaginary parts of the Nyquist plot data.
    """

    _, h1 = n_re_freq(sy, w)
    _, h2 = n_im_freq(sy, w)
    return h1, h2


def plot_nyquist(sy: System, w: int | np.ndarray | None = None, title: str | None = None) -> None:
    """
    Plot the Nyquist diagram for a system.

    This function generates a Nyquist plot for a given system by plotting the real and imaginary parts
    of the frequency response at specified angular frequencies.

    :param sy: The system for which to plot the Nyquist diagram.
    :param w: The angular frequencies at which to calculate the Nyquist plot. If None, the default range is used.
    :param title: Optional title for the Nyquist plot. If None, a default title is used.
    """

    re, im = __nyquist(sy, w)
    plt.plot(re, im)
    if title is not None:
        plt.title(title)
    else:
        plt.title('Nyquist')
    plt.xlabel('Re H(jw)')
    plt.ylabel('Im H(jw)')
    plt.grid()
    plt.show()
