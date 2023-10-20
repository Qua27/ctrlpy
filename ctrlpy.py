import sympy as sp
from functools import wraps


def simplifier(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        return sp.simplify(res)

    return wrapper


class System:
    t, s = sp.symbols('t s')
    w = sp.symbols('w', real=True)

    def __init__(self, tf: str):
        self.tf = sp.sympify(tf)

    def __repr__(self) -> str:
        return f'{{System with transfer function: {self.tf}}}'

    @simplifier
    def response(self, inp_signal_str: str) -> sp.Expr:
        inp_signal = sp.sympify(inp_signal_str)
        U = sp.laplace_transform(inp_signal, self.t, self.s)[0]
        Y = U * self.tf
        return sp.inverse_laplace_transform(Y, self.s, self.t)

    @simplifier
    def step_response(self) -> sp.Expr:
        return self.response('1')

    @simplifier
    def delta_response(self) -> sp.Expr:
        return self.response('DiracDelta(t)')

    @simplifier
    def freqs(self) -> sp.Expr:
        H_jw = self.tf.subs({self.s: sp.I * self.w})
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

    def log_magnitude(self):
        pass

    def __nyquist(self):
        return sp.lambdify(self.w, self.re_freq(), 'numpy'), sp.lambdify(self.w, self.im_freq(), 'numpy')
