#!/usr/bin/env python3
import sys
sys.path.append('..')
import os
import argparse
import numpy as np

import methods.parameters as parameters
from methods import results

results = os.path.join(results, 'fits-parameters')
if not os.path.isdir(results):
    os.makedirs(results)

parser = argparse.ArgumentParser(
    description='Plot model parameters.')
parser.add_argument('--show', action='store_true', help='Show plots', \
    default=False)
parser.add_argument('-b', '--base_model', type=str, choices=['lei', 'li'],
        default='li', help='Base hERG model')
parser.add_argument('-v', '--validation', action='store_true', help='Use validation drugs', \
    default=False)
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
    model_names = [f'm{i}' for i in range(1, 14)]
    model_names += [f'm{i}' for i in ['2i', '5i']]
else:
    model_names = [f'{base_model}-m{i}' for i in range(1, 14)]
    model_names += [f'{base_model}-m{i}' for i in ['2i', '5i']]

model_names_with_default = ['li'] + model_names

exclude_model_list = parameters.exclude_model_list['li']

for compound in compounds:
    for model_name in model_names_with_default:
        binding_names = [x.split('.')[-1] for x in
                         parameters._model_binding[model_name]]
        binding_params = parameters.binding[model_name][compound]
        print(binding_names)
        print(binding_params)
        if model_name in exclude_model_list[compound]:
            print('Exclude')
