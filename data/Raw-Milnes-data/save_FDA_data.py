#!/usr/bin/env python3
import argparse
import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

drugs = ['astemizole', 'azimilide', 'bepridil', 'chlorpromazine', 'cisapride', 'clarithromycin', 'clozapine', \
    'diltiazem', 'disopyramide', 'dofetilide', 'domperidone', 'droperidol', 'ibutilide', 'loratadine', \
    'metoprolol', 'mexiletine', 'nifedipine', 'nitrendipine', 'ondansetron', 'pimozide',  'quinidine', \
    'ranolazine', 'risperidone', 'sotalol', 'tamoxifen', 'terfenadine', 'vandetanib', 'verapamil']

parser = argparse.ArgumentParser()
parser.add_argument("--drug", type=str, choices=drugs, help="which country to use", default='quinidine')
args = parser.parse_args()

quinidine_concs = [100, 300, 1000, 10000]
bepridil_concs = [10, 30, 100, 300]
dofetilide_concs = [1, 3, 10, 30]
sotalol_concs = [10000, 30000, 100000, 300000]
chlorpromazine_concs = [100, 300, 1000, 3000]
cisapride_concs = [1, 10, 100, 300]
terfenadine_concs = [3, 10, 30, 100]
ondansetron_concs = [300, 1000, 3000, 10000]
diltiazem_concs = [3000, 10000, 30000, 100000]
mexiletine_concs = [10000, 30000, 100000, 300000]
ranolazine_concs = [1000, 10000, 30000, 100000]
verapamil_concs = [30, 100, 300, 1000]

disopyramide_concs = [1000, 3000, 6000, 10000]
ibutilide_concs = [1, 3, 10, 100]
domperidone_concs = [3, 10, 30, 100]
metoprolol_concs = [3000, 10000, 30000, 100000]
loratadine_concs = [250, 2000, 5000, 20000]
tamoxifen_concs = [100, 300, 1000, 3000]
risperidone_concs = [30, 100, 300, 1000]
clozapine_concs = [300, 1000, 3000, 10000]
astemizole_concs = [1, 3, 10, 30]

azimilide_concs = [30, 300, 1000, 3000]
clarithromycin_concs = [3000, 10000, 30000, 100000]
droperidol_concs = [10, 30, 100, 1000]
pimozide_concs = [1, 10, 50, 100]
vandetanib_concs = [30, 100, 300, 1000]

nifedipine_concs = [100000, 300000, 500000]
nitrendipine_concs = [10000, 30000, 100000]

sweeps = np.linspace(1, 10, 10, dtype=int)
concs_dict = {'astemizole': astemizole_concs, 'azimilide': azimilide_concs, 'bepridil': bepridil_concs, \
    'chlorpromazine': chlorpromazine_concs, 'cisapride': cisapride_concs, 'clarithromycin': clarithromycin_concs, \
    'clozapine': clozapine_concs, 'diltiazem': diltiazem_concs, 'disopyramide': disopyramide_concs, \
    'dofetilide': dofetilide_concs, 'domperidone': domperidone_concs, 'droperidol': droperidol_concs, \
    'ibutilide': ibutilide_concs, 'loratadine': loratadine_concs, 'metoprolol': metoprolol_concs, \
    'pimozide': pimozide_concs, 'mexiletine': mexiletine_concs, 'nifedipine': nifedipine_concs, \
    'nitrendipine': nitrendipine_concs, 'ondansetron': ondansetron_concs, 'quinidine': quinidine_concs, \
    'ranolazine': ranolazine_concs, 'risperidone': risperidone_concs, 'sotalol': sotalol_concs, \
    'tamoxifen': tamoxifen_concs, 'terfenadine': terfenadine_concs, 'vandetanib': vandetanib_concs, \
    'verapamil': verapamil_concs}
concs = concs_dict[args.drug]

exp = []
with open(args.drug + '.csv', encoding='utf8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['sweep'] == '1':
            exp.append(int(row['exp']))

N_exps = np.max(exp)

frac = np.zeros((len(sweeps), len(concs), 1000))
frac_SD = np.zeros((len(sweeps), len(concs), 1000))

for i, conc in enumerate(concs):

    for j, sweep in enumerate(sweeps):
        fblock = np.zeros(1000)
        fblock_SD = []
        N_complete_exps = np.copy(N_exps)

        for k in range(N_exps):
            fblock_exp = []

            with open(args.drug + '.csv', encoding='utf8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['conc'] == str('%.5g' % conc) and row['sweep'] == str(sweep) and row['exp'] == str(k+1):
                        fblock_exp.append(float(row['frac']))

            if len(fblock) == 1000 and len(fblock_exp) == 1000:
                fblock += fblock_exp 
                fblock_SD.append(fblock_exp)
            else:
                N_complete_exps = N_complete_exps - 1
                print('Missing entries')

            print('Conc: ' + str('%.5g' % conc) + ', Sweep: ' + str(sweep) + ', Exp: ' + str(k+1))

        print('N complete exps:')
        print(N_complete_exps)
        frac[j, i, :] = fblock / N_complete_exps
        frac_SD[j, i, :] = np.std(fblock_SD, axis=0)
np.save(args.drug + '.npy', frac)
np.save(args.drug + '-SD.npy', frac_SD)

