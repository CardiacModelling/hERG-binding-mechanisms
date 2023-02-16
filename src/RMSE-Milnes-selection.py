#!/usr/bin/env python3
import sys
sys.path.append('..')
import argparse
import numpy as np

path2fits = '../data/Milnes-data-fits'
path2fitsre = '../data/Milnes-data-fits-re'
path2data = '../data/Raw-Milnes-data'

drugs = ['astemizole', 'azimilide', 'bepridil', 'chlorpromazine', 'cisapride', 'clarithromycin', 'clozapine', \
    'diltiazem', 'disopyramide', 'dofetilide', 'domperidone', 'droperidol', 'ibutilide', 'loratadine', \
    'metoprolol', 'mexiletine', 'nifedipine', 'nitrendipine', 'ondansetron', 'pimozide',  'quinidine', \
    'ranolazine', 'risperidone', 'sotalol', 'tamoxifen', 'terfenadine', 'vandetanib', 'verapamil']

# Check input arguments
parser = argparse.ArgumentParser(
    description='Select plausible models with experimental data')
parser.add_argument("--n_samples", type=int, help="Number of Bootstrap samples to use", \
    default=1000)
args = parser.parse_args()


baseline_models_old = ['lei', 'fda']
base_models = ['lei', 'li']
base_models_prefix = ['', 'li-']

models = ['0a', '0b', 1, 2, '2i', 3, 4, 5, '5i', 6, 7, 8, 9, 10, 11, 12, 13]

#func = lambda x, y: x + (x - y) * .7
func = lambda x, y: 1.2 * x
exclude_model_list_template = lambda x, y: [f'{x}m{i}' for i in y]

excluded_list = {}
for b in base_models:
    excluded_list[b] = {}

for drug in drugs:
    RMSEs = {}
    for b in base_models:
        RMSEs[b] = {}

    for i, m in enumerate(models):
        if True and (type(m) is int):
            model_str = 'model' + str(m)
            for a, b in enumerate(baseline_models_old):
                RMSE = np.loadtxt(path2fits + '/Drug-' + drug + '-' + b + '-' + model_str + \
                    '-protocol-Milnes-fit-RMSE-errors.txt')
                if (RMSE.size > 1):
                    RMSE = RMSE[0]
                RMSEs[base_models[a]][m] = RMSE
        else:
            for a, b in enumerate(base_models):
                RMSE = np.loadtxt(path2fitsre + f'/{drug}-{base_models_prefix[a]}m{m}-errors.txt')
                if (RMSE.size > 1):
                    RMSE = RMSE[0]
                RMSEs[base_models[a]][m] = RMSE

    FDA_RMSE = np.loadtxt(path2fits + '/FDA/' + drug + '-RMSE.txt')
    expt_samples = np.loadtxt(path2data + '/' + drug + '-bootstrap-' + str(args.n_samples) + '-samples.txt')

    for a, b in enumerate(base_models):
        excluded = []
        for m in models:
            if FDA_RMSE > np.max(expt_samples):
                if RMSEs[b][m] > func(FDA_RMSE, np.max(expt_samples)):
                    excluded.append(m)
            else:
                if RMSEs[b][m] > np.max(expt_samples):
                    excluded.append(m)
        excluded_list[b][drug] = exclude_model_list_template(base_models_prefix[a], excluded)

# Print it as a dict for methods/parameters.py
print('{')
for b in base_models:
    print(f'    \'{b}\': {{')
    for drug in drugs:
        print(f'        \'{drug}\': {excluded_list[b][drug]},')
    print('    },')
print('}')
