#!/usr/bin/env python3
import sys
sys.path.append('..')
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set_theme()

import methods.parameters as parameters
from methods import results

results = os.path.join(results, 'fits-parameters')
if not os.path.isdir(results):
    os.makedirs(results)

parser = argparse.ArgumentParser(description='Plot model parameters.')
parser.add_argument('--show', action='store_true', help='Show plots',
        default=False)
parser.add_argument('-b', '--base_model', type=str, choices=['lei', 'li'],
        default='li', help='Base hERG model')
parser.add_argument('-v', '--validation', action='store_true',
        help='Use validation drugs', default=False)
args = parser.parse_args()

ap_model_name = 'dutta'
dose_name = 'li'
prefix = 'parameters-validation' if args.validation else 'parameters'

if args.validation:
    compounds = list(parameters._drug_validation_list)
    print('Using FDA CiPA validation list')
else:
    compounds = list(parameters._drug_training_list)
    print('Using FDA CiPA training list')
#compounds = ['dofetilide', 'cisapride', 'verapamil', 'mexiletine']

base_model = args.base_model
if base_model == 'lei':
    model_names = [f'm{i}' for i in range(1, 3)]
    model_names += [f'm{i}' for i in ['2i']]
    model_names += [f'm{i}' for i in range(3, 6)]
    model_names += [f'm{i}' for i in ['5i']]
    model_names += [f'm{i}' for i in range(6, 14)]
else:
    model_names = [f'{base_model}-m{i}' for i in range(1, 3)]
    model_names += [f'{base_model}-m{i}' for i in ['2i']]
    model_names += [f'{base_model}-m{i}' for i in range(3, 6)]
    model_names += [f'{base_model}-m{i}' for i in ['5i']]
    model_names += [f'{base_model}-m{i}' for i in range(6, 14)]

model_names_with_default = ['li'] + model_names

exclude_model_list = parameters.exclude_model_list['li']

parameter_names = ['kon', 'koff', 'hill']
# NOTE: ku, kt here refers to m11 (not Li et al. 2017 model)
excluded_parameters = ['k2f', 'ku' ,'kt', 'halfmax', 'vhalf']
# >>> Parameter naming consistency
kon_variants = ['kforward', 'konO', 'konI', 'kon_o', 'kon_i']
kmax_variants = ['kmax', 'Kmax']  # kon = kmax * ku
koff_variants = ['ku', 'Ku', 'koffO', 'koffI', 'koff_o', 'koff_i']
vhalf_variants = ['Vhalf']
kt_variants = ['Kt']
# <<<

xlabel = 'Model'
if base_model == 'li':
    xlabel += ' (base Li et al.)'
elif base_model == 'lei':
    xlabel += ' (base Lei et al.)'
else:
    raise ValueError(f'Unexpected base model {base_model}')
xticks_loc = np.arange(len(model_names_with_default))
xticks_value = ['Li et al.']
if base_model == 'lei':
    xticks_value += [x.strip('m') for x in model_names]
else:
    xticks_value += [x.split('-m')[1] for x in model_names]

for compound in compounds:
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axes[-1].set_xticks(xticks_loc, xticks_value)
    axes[-1].set_xlabel(xlabel)
    axes[0].set_title(f'Compound: {compound}')
    # NOTE: match parameter_names order
    axes[0].set_ylabel(r'$k_{on}$ (ms$^{-1}$)')
    axes[1].set_ylabel(r'$k_{off}$ (ms$^{-1}$)')
    axes[2].set_ylabel(r'hill')
    for i_m, model_name in enumerate(model_names_with_default):
        binding_names = [x.split('.')[-1] for x in
                         parameters._model_binding[model_name]]
        binding_params = parameters.binding[model_name][compound]
        # >>> Goodness of fits of the model
        if model_name in exclude_model_list[compound]:
            alpha = 0.25
            marker = 'x'
        else:
            alpha = 1
            marker = 'o'
        # <<<
        for n, v in zip(binding_names, binding_params):
            # >>> Parameter naming consistency
            if n in kon_variants:
                n = 'kon'
            if n in koff_variants and ('m11' not in model_name):
                n = 'koff'
            if n in kmax_variants:
                n = 'kon'
                if 'ku' in binding_names:
                    v *= binding_params[binding_names.index('ku')]
                elif 'Ku' in binding_names:
                    v *= binding_params[binding_names.index('Ku')]
                else:
                    raise ValueError('Expecting kon = kmax * ku')
            if n in vhalf_variants:
                n = 'vhalf'
            if n in kt_variants:
                n = 'kt'
            # <<<
            if n not in excluded_parameters:
                idx = parameter_names.index(n)
                axes[idx].scatter(i_m, v, alpha=alpha, marker=marker)
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    fig.tight_layout()
    if args.show:
        plt.show()
    else:
        path = os.path.join(results, f'{prefix}-{compound}-{base_model}')
        fig.savefig(path+'.pdf', format='pdf')
    plt.close(fig)
