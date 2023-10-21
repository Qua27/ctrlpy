from ctrlpy import System


def an_re_freq(w):
    return 1 / (4 * w * w + 1)


s = System('1/(2*s+1)')
print(s)
print(s.re_freq())
print(s.magnitude_freq())
print(s.log_magnitude())
s.plot(s.re_freq, plot_range=(0, 1000), func_to_compare_with=an_re_freq)
s.plot(s.log_magnitude, (0, 1000))
s.plot(s.delta_response)
