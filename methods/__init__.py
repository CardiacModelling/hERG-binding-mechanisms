#
# Methods module
#

# Get methods directory
import os, inspect
frame = inspect.currentframe()
DIR_METHOD = os.path.abspath(os.path.dirname(inspect.getfile(frame)))
del(os, inspect, frame)

#
# Settings
#
run = 1

concentrations = {
    #'Na_i': 10,
    #'Na_o': 140,
    # FDA; Li et al. 2017
    'K_i': 140,
    'K_o': 5,
    #'Ca_i': 1e-5,
    #'Ca_o': 2,
}

if run == 1:
    # 10 seconds holding time between episodes
    results = 'results'

    t_hold = 10e3  # ms
    v_hold = -80   # mV

else:
    raise ValueError('Unknown run number')

# Make results dir
import os
if not os.path.isdir(results):
    os.makedirs(results)
del(os)
