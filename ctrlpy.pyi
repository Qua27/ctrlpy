import numpy as np
import sympy as sp
from scipy import signal
from typing import Callable, NamedTuple


class PlotData(NamedTuple):
    xlabel: str
    ylabel: str
    log_scale: bool
    title: str


def simplifier(func: Callable[..., sp.Expr]) -> Callable[..., sp.Expr]: ...


def process(expr: sp.Expr, s: sp.Symbol) -> sp.Expr: ...


class ExprMissingVariableError(ValueError):
    def __init__(self, variable_name: sp.Symbol) -> None: ...


class InvalidSymbolError(ValueError):
    def __init__(self, message: str | None = ...) -> None: ...


class System:
    tf: sp.Expr

    def __init__(self, tf: sp.Expr) -> None: ...

    @classmethod
    def from_string(cls, string: str, symbol: sp.Symbol | str) -> System: ...

    def __eq__(self, other: object) -> bool: ...

    @property
    def tf_coefs(self) -> tuple[np.ndarray, np.ndarray]: ...

    @property
    def lti(self) -> signal.lti: ...

    def response(self, inp_signal: sp.Expr | str) -> sp.Expr: ...

    def step_response(self) -> sp.Expr: ...

    def delta_response(self) -> sp.Expr: ...


def s_freqs(sy: System) -> sp.Expr: ...


def create_plot_data(xlabel: str, ylabel: str, log_scale: bool, title: str) -> PlotData: ...


def plot(sy: System, function: Callable[[System, int | np.ndarray | None], tuple[np.ndarray, np.ndarray]],
         plot_range: tuple[float, float], plot_data: PlotData,
         func_to_compare_with: Callable[[float | np.ndarray], float | np.ndarray] | None = ...): ...


def s_re_freq(sy: System) -> sp.Expr: ...


def s_im_freq(sy: System) -> sp.Expr: ...


def s_magnitude_freq(sy: System) -> sp.Expr: ...


def s_phase_freq(sy: System) -> sp.Expr: ...


def s_log_magnitude(sy: System) -> sp.Expr: ...


def n_freqs(sy: System, w: int | np.ndarray | None = ...) -> tuple[np.ndarray, np.ndarray]: ...


def n_re_freq(sy: System, w: int | np.ndarray | None = ...) -> tuple[np.ndarray, np.ndarray]: ...


def n_im_freq(sy: System, w: int | np.ndarray | None = ...) -> tuple[np.ndarray, np.ndarray]: ...


def n_magnitude_freq(sy: System, w: int | np.ndarray | None = ...) -> tuple[np.ndarray, np.ndarray]: ...


def n_phase_freq(sy: System, w: int | np.ndarray | None = ...) -> tuple[np.ndarray, np.ndarray]: ...


def n_log_magnitude_freq(sy: System, w: int | np.ndarray | None = ...) -> tuple[np.ndarray, np.ndarray]: ...


def plot_nyquist(sy: System, w: int | np.ndarray | None = ..., title: str | None = ...) -> None: ...
