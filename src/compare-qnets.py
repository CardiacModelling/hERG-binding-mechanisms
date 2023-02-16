#!/usr/bin/env python3
import sys
sys.path.append('..')
import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

import methods.models as models
import methods.parameters as parameters
from methods import results

results = os.path.join(results, 'qnet')
if not os.path.isdir(results):
    os.makedirs(results)
if not os.path.isdir(os.path.join(results, 'aps')):
    os.makedirs(os.path.join(results, 'aps'))

parser = argparse.ArgumentParser(
    description='Simulating qNet.')
parser.add_argument('--show', action='store_true', help='Show plots', \
    default=False)
parser.add_argument('-x', '--cache', action='store_true', help='Use cache', \
    default=False)
parser.add_argument('-m', '--model', type=int, help='Simulate only one binding model; all if None; FDA model if 0', \
    default=None)
parser.add_argument('-b', '--base_model', type=str, choices=['lei', 'li'],
        default='li', help='Base hERG model')
parser.add_argument('-v', '--validation', action='store_true', help='Use validation drugs', \
    default=False)
parser.add_argument('-s', '--save_raw', action='store_true', help='Save raw AP simulation results', \
    default=False)
args = parser.parse_args()

CACHE = args.cache

ap_model_name = 'dutta'
dose_name = 'li'
prefix = 'qnet-validation' if args.validation else 'qnet'

times = np.arange(0, 2000.01, 0.01)
x_cmax = [0, 0.5, 1, 5, 10, 15, 20, 25]

exception = {
    'quinidine': [0, 1]
}

if args.validation:
    compounds = list(parameters._drug_validation_list)
    print('Using FDA CiPA validation list')
else:
    compounds = list(parameters._drug_training_list)
    print('Using FDA CiPA training list')
#compounds = ['dofetilide', 'cisapride', 'verapamil', 'mexiletine']
base_model = args.base_model
if base_model == 'lei':
    model_names = [f'm{i}' for i in range(1, 14)]
    model_names += [f'm{i}' for i in ['2i', '5i', '0a', '0b']]
else:
    model_names = [f'{base_model}-m{i}' for i in range(1, 14)]
    model_names += [f'{base_model}-m{i}' for i in ['2i', '5i', '0a', '0b']]

model_names_with_default = ['li'] + model_names

if args.model is not None:
    model_names_with_default = [model_names_with_default[args.model]]
    if args.model != 0:
        model_names = [model_names[args.model - 1]]
    else:
        model_names = []

exclude_model_list = parameters.exclude_model_list[base_model]

qnet = {}
if not CACHE:
    for model_name in model_names_with_default:

        print(f'Simulating qNet with {model_name}')
        model = models.APModel(ap_model_name, model_name, cl=2000,
                               parameters=['binding'])
        #model.init_state = _model_control_steady_state[ap_model_name] # TODO
        ikr_conductance = parameters._dutta_ikr_conductance[base_model]
        model.set_ikr_conductance(ikr_conductance)
        model.update_init_state_as_steady_state()
        qnet[model_name] = {}

        for compound in compounds:
            print(f'For {compound}')

            qnet[model_name][compound] = {}

            binding_params = parameters.binding[model_name][compound]
            non_hERG_ic50s = parameters.non_hERG_ic50[dose_name][compound]
            non_hERG_hs = parameters.non_hERG_hill[dose_name][compound]
            dose = parameters.cmax[dose_name][compound]
            model.set_non_hERG_parameters(non_hERG_ic50s, non_hERG_hs)

            for i in x_cmax:
                if args.save_raw:
                    raw_path = os.path.join(
                        results,
                        'aps',
                        f'{ap_model_name}-{model_name}-{compound}-{i}_cmax'
                    )
                else:
                    raw_path = None
                model.set_dose(dose * i)
                q = model.qnet(binding_params, times, save=raw_path)
                qnet[model_name][compound][i] = q

        with open(os.path.join(results, f'{prefix}-{ap_model_name}-{model_name}.pkl'), 'wb') as f:
            pickle.dump(qnet[model_name], f)

else:
    print('Loading qNet from cache')
    for model_name in model_names_with_default:
        with open(os.path.join(results, f'{prefix}-{ap_model_name}-{model_name}.pkl'), 'rb') as f:
            qnet[model_name] = pickle.load(f)

# Visualise

for model_name in model_names:
    plt.figure()
    plt.title(f'Model: {model_name}')
    plt.xlabel(r'$\times$ Cmax')
    plt.ylabel('qNet (C/F)')
    for i, compound in enumerate(compounds):
        q0 = []
        q = []
        if compound in exception:
            x_cmax_ = [x_cmax[k] for k in exception[compound]]
        else:
            x_cmax_ = x_cmax
        for x in x_cmax_:
            q0.append(qnet['li'][compound][x])
            q.append(qnet[model_name][compound][x])
        plt.plot(x_cmax_, q0, 'o--', c=f'C{i}', alpha=0.5)
        if model_name in exclude_model_list[compound]:
            plt.plot(x_cmax_, q, 'o:', c=f'C{i}', alpha=0.2, label=compound)
        else:
            plt.plot(x_cmax_, q, 'o-', c=f'C{i}', label=compound)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(results, f'{prefix}-{ap_model_name}-{model_name}'), dpi=300)
    plt.close()

for compound in compounds:
    plt.figure()
    plt.title(f'Compound: {compound}')
    plt.xlabel(r'$\times$ Cmax')
    plt.ylabel('qNet (C/F)')
    if compound in exception:
        x_cmax_ = [x_cmax[k] for k in exception[compound]]
    else:
        x_cmax_ = x_cmax
    for model_name in model_names:
        q = []
        for x in x_cmax_:
            q.append(qnet[model_name][compound][x])
        if model_name in exclude_model_list[compound]:
            plt.plot(x_cmax_, q, 'o:', alpha=0.2, label=model_name)
        else:
            plt.plot(x_cmax_, q, 'o-', label=model_name)
    q0 = []
    for x in x_cmax_:
        q0.append(qnet['li'][compound][x])
    plt.plot(x_cmax_, q0, 'o--', c='k', alpha=0.5, label='li')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(results, f'{prefix}-{ap_model_name}-{base_model}-{compound}'), dpi=300)
    plt.close()
