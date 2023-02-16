#!/usr/bin/env python3
import sys
sys.path.append('..')
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import methods.models as models
import methods.parameters as parameters
from methods import results

results = os.path.join(results, 'Milnes-fits')
if not os.path.isdir(results):
    os.makedirs(results)

CACHE = ('--cache' in sys.argv) or ('-x' in sys.argv)

#protocol = '../protocols/protocol-Milnes.csv'
protocol = '../protocols/protocol-Milnes.mmt'
times = np.arange(0, 15e3, 10)
# Get time window where V = 0 mV
#p = np.loadtxt(protocol, delimiter=',', skiprows=1)
#win = p[:, 0][np.abs(p[:, 1] - 0) < 1e-5] * 1e3  # s -> ms
#win = (times >= win[0]) & (times < win[-1])
#win = (times >= 1e3) & (times < 10.9e3)
win = (times >= 1.1e3) & (times < 11e3)

compounds = list(parameters._drug_list)
#compounds = ['verapamil']
base_model = 'li'
#base_model = 'lei'
if base_model == 'lei':
    model_names = [f'm{i}' for i in range(1, 14)]
    model_names += [f'm{i}' for i in ['0a', '0b', '2i', '5i']]
else:
    model_names = [f'{base_model}-m{i}' for i in range(1, 14)]
    model_names += [f'{base_model}-m{i}' for i in ['0a', '0b', '2i', '5i']]
exclude_model_list = parameters.exclude_model_list[base_model]

path2cache = os.path.join(results, f'Milnes-{base_model}')
if not CACHE:
    milnes = {}

    # Go through all compounds
    for compound in compounds:
        milnes[compound] = {}

        print(f'Simulating the Milnes protocol for {compound}')
        conc = parameters.drug_concs[compound]
        colors = [f'C{i}' for i in range(len(conc))]

        # Go through all models
        for i_m, model_name in enumerate(model_names):
            milnes[compound][model_name] = {}

            # Model
            model = models.Model(model_name,
                                 protocol,
                                 parameters=['binding'],
                                 analytical=True)
            binding_params = parameters.binding[model_name][compound]

            # Control
            model.set_dose(0)
            before = model.simulate(binding_params, times)

            # Go through all concentrations
            for c, color in zip(conc, colors):
                model_milnes = []

                # Compound
                model.set_dose(c)
                after = model.simulate(binding_params, times)
                after[before < 1e-1] = np.NaN
                after[before >= 1e-1] /= before[before >= 1e-1]
                model_milnes = np.append(model_milnes, after[win])
                for i in range(9):
                    after = model.simulate(binding_params, times, reset=False)
                    after[before < 1e-1] = np.NaN
                    after[before >= 1e-1] /= before[before >= 1e-1]
                    model_milnes = np.append(model_milnes, after[win])

                milnes[compound][model_name][c] = model_milnes

        # Get time from data (do it once only)
        u = np.loadtxt(
            f'../data/Milnes-data/drug-{compound}-conc-{c}-Milnes.csv',
            delimiter=',',
            skiprows=1
        )
        t = u[:, 0]
        milnes[compound]['times'] = t

        with open(f'{path2cache}-{compound}.pkl', 'wb') as f:
            pickle.dump(milnes[compound], f)

else:
    print('Loading the Milnes protocol from cache')
    milnes = {}

    # Go through all compounds
    for compound in compounds:
        with open(f'{path2cache}-{compound}.pkl', 'rb') as f:
            milnes[compound] = pickle.load(f)

# Visualise

t0 = times[win]
t0 -= t0[0]
t_1 = t0[-1]
for compound in compounds:
    conc = parameters.drug_concs[compound]
    colors = [f'C{i}' for i in range(len(conc))]

    # Data
    for c, color in zip(conc, colors):
        u = np.loadtxt(
            f'../data/Milnes-data/drug-{compound}-conc-{c}-Milnes.csv',
            delimiter=',',
            skiprows=1
        )
        t = u[:, 0]
        u = u[:, 1]
        s = np.loadtxt(
            f'../data/Milnes-data/drug-{compound}-conc-{c}-Milnes-SD.csv',
            delimiter=',',
            skiprows=1
        )[:, 1]

        plt.plot(t, u, alpha=0.5, c=color, label=f'{c} nM')
        plt.fill_between(t, u - s, u + s, alpha=0.25, color=color, lw=0)

    # Model
    for i_m, model_name in enumerate(model_names):
        for c, color in zip(conc, colors):
            model_milnes = milnes[compound][model_name][c]

            # Plot
            if model_name in exclude_model_list[compound]:
                ls = '-'
                alpha = 0.5
                color = '#7f7f7f'
            else:
                ls = '-'
                alpha = 0.75
            plt.plot(t, model_milnes, alpha=alpha, c=color, ls=ls)

    xticks = []
    for i in range(10):
        xticks.append((i + 0.5) * t0[-1])
        if i % 2:
            plt.axvspan(i * t0[-1], (i + 1) * t0[-1], color='#7f7f7f',
                        linestyle='', alpha=0.25)
    plt.xticks(xticks, np.arange(1, 11))

    plt.ylim([0, 1.1])
    plt.xlabel('Pulse number', fontsize=14)
    plt.ylabel('Percentage current', fontsize=14)
    plt.title(compound, loc='left', fontsize=14)
    plt.legend(loc='lower right', bbox_to_anchor=(1, 1.01), ncol=4,
               fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(results, f'Milnes-{base_model}-{compound}'),
                dpi=300)
    plt.close()
