#!/usr/bin/env python3
import sys
sys.path.append('..')
import pints
import numpy as np
import matplotlib.pyplot as plt
import methods.models as models
import methods.parameters as parameters

# models
m0 = models.APModel('dutta', 'li', cl=2000, parameters=['conductance'])
m1 = models.APModel('dutta', 'm1', cl=2000, parameters=['conductance'])
dt = 0.01
#t = np.arange(0, 4000, 0.1)
t = np.arange(0, 2000, dt)

# default values
g_li = 0.046585
g_lei = 0.1524

# Just so the model works; we are still using the control model
compound = 'dofetilide'
non_hERG_ic50s = parameters.non_hERG_ic50['li'][compound]
non_hERG_hs = parameters.non_hERG_hill['li'][compound]
m0.set_non_hERG_parameters(non_hERG_ic50s, non_hERG_hs)
m1.set_non_hERG_parameters(non_hERG_ic50s, non_hERG_hs)
m0.set_dose(0)
m1.set_dose(0)


def APD90(trace, dt=dt, offset=50):
    APA = max(trace) - min(trace)
    APD90_v = min(trace) + 0.1 * APA
    offidx = int((offset + 50) / dt)  # avoid upstroke
    index = np.abs(np.array(trace[offidx:]) - APD90_v).argmin()
    APD90 = (index + offidx) * dt - offset
    return APD90


debug = '--debug' in sys.argv
if debug:
    mt = models.APModel('dutta', 'm1', cl=2000, parameters=['binding'])
    ikr_conductance = parameters._dutta_ikr_conductance['lei']
    mt.set_ikr_conductance(ikr_conductance)
    mt.set_non_hERG_parameters(non_hERG_ic50s, non_hERG_hs)
    mt.set_dose(0)
    c = mt.simulate(np.ones(mt.n_parameters()), t)

    plt.figure()
    plt.xlabel('Time (ms)'); plt.ylabel('Membrane potential (mV)')
    a0 = m0.simulate([g_li], t)
    a1 = m1.simulate([g_lei], t)
    a2 = m1.simulate([g_lei / 1.5], t)
    a3 = m1.simulate([g_lei / 2.5], t)
    b = m1.simulate([parameters._dutta_ikr_conductance['lei']], t)
    print(APD90(a0), APD90(a1), APD90(a2), APD90(a3), APD90(b), APD90(c))
    plt.plot(t, a0, c='C0')
    plt.plot(t, a1, c='C1')
    plt.plot(t, a2, c='C2')
    plt.plot(t, a3, c='C3')
    plt.plot(t, b, c='k', ls='--')
    plt.plot(t, b, c='#7f7f7f', ls=':')
    plt.show()
    sys.exit()


class ErrorMeasure(pints.ProblemErrorMeasure):
    """
    Parameters
    ----------
    problem: pints.SingleOutputProblem or pints.MultiOutputProblem.
    metric: A function that takes an AP trace and returns a metric (eg APD).
    """
    def __init__(self, problem, metric=APD90):
        super(ErrorMeasure, self).__init__(problem)
        self._m = metric
        self._m_values = self._m(self._values)

    def __call__(self, x):
        return np.abs(self._m(self._problem.evaluate(x)) - self._m_values)


data = m0.simulate([g_li], t)

problem = pints.SingleOutputProblem(m1, t, data)
error = ErrorMeasure(problem, metric=APD90)
bounds = pints.RectangularBoundaries([g_lei / 2.5], [g_lei])

q0 = g_lei / 1.5

# Create optimiser
opt = pints.OptimisationController(
    error, q0, boundaries=bounds, method=pints.NelderMead)
opt.set_max_iterations(None)
opt.set_max_unchanged_iterations(iterations=50, threshold=1e-3)
#opt.set_parallel(True)

# Run optimisation
try:
    with np.errstate(all='ignore'): # Tell numpy not to issue warnings
        p, s = opt.run()
except ValueError:
    import traceback
    traceback.print_exc()

print(p, s)
