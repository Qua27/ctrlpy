from ctrlpy import System


def an_magnitude(w):
    return 1 / (4 * w * w + 1)


s = System('1/(2*s+1)')
print(s.re_freq())
print(s)
print(s.magnitude_freq())
