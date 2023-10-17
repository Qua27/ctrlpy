from ctrlpy import System
from numpy import exp

s = System('1/(2*s+1)')
print(s.response('1'))
s.plot_response('1')
s.plot_response('1', function_to_compare_with=lambda t: 1 - exp(-t / 2))
