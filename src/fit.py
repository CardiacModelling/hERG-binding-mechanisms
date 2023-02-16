#!/usr/bin/env python3
import sys
import os
import argparse
import numpy as np
import pints

sys.path.append('..')
import methods.models as models
import methods.parameters as parameters
import methods.boundaries as boundaries
from methods import results

results = os.path.join(results, 'fits')
if not os.path.isdir(results):
    os.makedirs(results)

def n_choices(n):
    return np.linspace(1, n, n, dtype=int)

# Check input arguments
parser = argparse.ArgumentParser(description='Fit models to data.')
parser.add_argument('-m', '--model', help='Binding model for optimisation')
parser.add_argument('-d', '--drug', type=str,
        choices=list(parameters._drug_list), default='quinidine',
        help='Drug name')
parser.add_argument('-b', '--base_model', type=str, choices=['lei', 'li'],
        default='li', help='Base hERG model')
parser.add_argument('-fh', '--fix_hill', action='store_true', default=False,
        help='Fix the Hill coefficient to 1 during fitting')
parser.add_argument('-r', '--repeats', type=int, default=10, metavar='N',
        help='Number of optimisation runs from different initial guesses')
parser.add_argument('-v', '--verbose', action='store_true', default=False,
        help='Output detailed information (optimisation logs)')
args = parser.parse_args()

# Get model string and params
drug_str = args.drug
base_model = args.base_model
if base_model == 'lei':
    model_str = f'm{args.model}'
else:
    model_str = f'{base_model}-m{args.model}'

#protocol = '../protocols/protocol-Milnes.csv'
protocol = '../protocols/protocol-Milnes.mmt'
times = np.arange(0, 15e3, 10)
# Get time window where V = 0 mV
#p = np.loadtxt(protocol, delimiter=',', skiprows=1)
#win = p[:, 0][np.abs(p[:, 1] - 0) < 1e-5] * 1e3  # s -> ms
#win = (times >= win[0]) & (times < win[-1])
#win = (times >= 1e3) & (times < 10.9e3)
win = (times >= 1.1e3) & (times < 11e3)

filename = os.path.join(results, f'{drug_str}-{model_str}')
print('=' * 79)
print('Selected drug', drug_str)
print('Selected base model', base_model)
print('Selected model', model_str)
print('Storing results to', f'{filename}.txt')
print('-' * 79)

# PINTS model
class ConcatMilnesModel(pints.ForwardModel):
    """A PINTS model simulating concatenated Milnes protocol."""
    def __init__(self, model, protocol, times, win, conc):
        self._model = models.Model(model,
                                   protocol,
                                   parameters=['binding'],
                                   analytical=True)
        if args.fix_hill:
            self._model.fix_hill()
        if args.model in ['12', '13']:
            self._model.fix_kt()
        self._win = win
        self._conc = conc
        self.n_pulses = 10
        self._times = times
        # Simulate dose free (control)
        self._model.set_dose(0)
        z = np.ones(self._model.n_parameters())
        self._before = self._model.simulate(z, self._times)[self._win]
    def n_parameters(self):
        return self._model.n_parameters()
    def simulate(self, parameters, times):
        self._model.set_dose(self._conc)
        try:
            out = []
            after = self._model.simulate(parameters, self._times)
            out = np.append(out, after[self._win] / self._before)
            for i in range(self.n_pulses - 1):
                after = self._model.simulate(parameters, self._times, reset=False)
                out = np.append(out, after[self._win] / self._before)
        except:  # TODO?
            out = np.ones(times.shape) * float('inf')
        assert(len(out) == len(times))
        return out


# Set individual errors and weights
weights = []
errors = []
concs = parameters.drug_concs[drug_str]
for conc in concs:

    # Create forward models
    model = ConcatMilnesModel(
        model_str,
        protocol,
        times,
        win,
        conc
    )

    # Load data
    u = np.loadtxt(
        f'../data/Milnes-data/drug-{drug_str}-conc-{conc}-Milnes.csv',
        delimiter=',',
        skiprows=1
    )
    concat_time = u[:, 0]
    concat_milnes = u[:, 1]

    # Create single output problem
    problem = pints.SingleOutputProblem(model, concat_time, concat_milnes)

    # Define error function
    errors.append(pints.RootMeanSquaredError(problem))

    # Add weighting based on range
    weights.append(1 / len(concs))

    # Debug
    debug = False
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(concat_time, concat_milnes)
        p = parameters.binding[model_str][drug_str]
        plt.plot(concat_time, model.simulate(p, concat_time))
        plt.show()
        sys.exit()

# Create weighted sum of errors
f = pints.SumOfErrors(errors, weights)

bounds = boundaries.Boundaries(args.model, fix_hill=args.fix_hill)

# Fix random seed for reproducibility
np.random.seed(100)

# Transformation
if args.model in ['12', '13']:
    transform = pints.ComposedTransformation(
        pints.LogTransformation(f.n_parameters() - 1),
        pints.IdentityTransformation(1),
    )
else:
    transform = pints.LogTransformation(f.n_parameters())

#
# Run
#
repeats = args.repeats
params, scores = [], []
if args.verbose:
    import myokit
    b = myokit.Benchmarker()
    opt_times = []
for i in range(repeats):
    print('Repeat ' + str(1 + i))

    # Choose random starting point
    #if i < 3:
    #    baselines = ['lei','whittaker','bowtie']
    #    q0 = np.log(np.loadtxt('fits_cmaes/Drug-' + drug_str + '-' + \
    #        baselines[i] + '-' + model_str + '-protocol-' + protocol_str + '-fit-RMSE.txt'))
    # if i == 0:
    #     q0 = np.log(lib_drug.get_FDA_model_parameters(args.drug))
    #     q0[-1] = np.exp(q0[-1])
    #     # model 12: 'ikr.kmax', 'ikr.ku', 'ikr.halfmax', 'ikr.n', 'ikr.vhalf'
    #     # model 13: 'ikr.kforward', 'ikr.ku', 'ikr.hill', 'ikr.vhalf'
    #     if args.model == 13:
    #         q0[0] = q0[1] * q0[0] / q0[2]
    #         q0 = np.delete(q0, 2)
    #     # q0[:len(q0)-1] = np.log(1e-6) # Force initial guesses to take on low values
    if False:
        pass
    else:
        q0 = bounds.sample()
    print('Starting parameters: ')
    print(q0)
    print('Initial score: ')
    print(f(q0))

    # Create optimiser
    opt = pints.OptimisationController(
        f, q0, boundaries=bounds, transformation=transform, method=pints.CMAES)
    if args.verbose:
        opt.set_log_to_file(filename + '-log-' + str(i) + '.txt')
        b.reset()
    opt.set_max_iterations(None)
    opt.set_parallel(True)

    # Run optimisation
    try:
        with np.errstate(all='ignore'): # Tell numpy not to issue warnings
            p, s = opt.run()
            params.append(p)
            scores.append(s)
            if args.verbose:
                opt_times.append(b.time())
    except ValueError:
        import traceback
        traceback.print_exc()

# Order from best to worst
order = np.argsort(scores)
scores = np.asarray(scores)[order]
params = np.asarray(params)[order]
if args.verbose:
    opt_times = np.asarray(opt_times)[order]

# Show results
print('Best scores:')
for score in scores[:10]:
    print(score)
print('Mean & std of score:')
print(np.mean(scores))
print(np.std(scores))
print('Worst score:')
print(scores[-1])

# Extract best
obtained_score = scores[0]
obtained_parameters = params[0]

# Store results
print('Storing best result...')
with open(filename + '.txt', 'w') as f:
    for x in obtained_parameters:
        f.write(pints.strfloat(x) + '\n')

print('Storing all errors')
with open(filename + '-errors.txt', 'w') as f:
    for score in scores:
        f.write(pints.strfloat(score) + '\n')

print('Storing all parameters')
for i, param in enumerate(params):
    with open(filename + '-parameters-' + str(1 + i) + '.txt', 'w') as f:
        for x in param:
            f.write(pints.strfloat(x) + '\n')

if args.verbose:
    print('Storing all simulation times')
    with open(filename + '-opt_times.txt', 'w') as f:
        for t in opt_times:
            f.write(pints.strfloat(t) + '\n')

print('Done.\n')
