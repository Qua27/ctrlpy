import unittest
import sympy as sp
import ctrlpy as cp


class CtrlPyTestCase(unittest.TestCase):
    t = sp.symbols('t', real=True)
    w = sp.symbols('w', real=True)
    s = cp.System.from_string('1/s', 's')

    def test_response(self):
        self.assertEqual(self.t ** 2 * sp.Heaviside(self.t) / 2, self.s.response('t'))

    def test_step_response(self):
        self.assertEqual(self.t * sp.Heaviside(self.t), self.s.step_response())

    def test_delta_response(self):
        self.assertEqual(sp.Heaviside(self.t), self.s.delta_response())

    def test_re_freq(self):
        self.assertEqual(0, cp.s_re_freq(self.s))

    def test_im_freq(self):
        self.assertEqual(-1 / self.w, cp.s_im_freq(self.s))

    def test_magnitude_freq(self):
        self.assertEqual(1 / sp.Abs(self.w), cp.s_magnitude_freq(self.s))

    def test_phase_freq(self):
        self.assertEqual(sp.atan2(-1 / self.w, 0), cp.s_phase_freq(self.s))

    def test_log_magnitude(self):
        self.assertEqual(20 * sp.log(1 / sp.Abs(self.w)) / sp.log(10), cp.s_log_magnitude(self.s))


if __name__ == '__main__':
    unittest.main()
