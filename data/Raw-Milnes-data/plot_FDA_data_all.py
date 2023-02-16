#!/usr/bin/env python3
import numpy as np
import argparse

drugs = ['astemizole', 'azimilide', 'bepridil', 'chlorpromazine', 'cisapride', 'clarithromycin', 'clozapine', \
    'diltiazem', 'disopyramide', 'dofetilide', 'domperidone', 'droperidol', 'ibutilide', 'loratadine', \
    'metoprolol', 'mexiletine', 'nifedipine', 'nitrendipine', 'ondansetron', 'pimozide',  'quinidine', \
    'ranolazine', 'risperidone', 'sotalol', 'tamoxifen', 'terfenadine', 'vandetanib', 'verapamil']

parser = argparse.ArgumentParser()
parser.add_argument("--drug", type=str, choices=drugs, help="which country to use", default='quinidine')
parser.add_argument("--show", action='store_true', help="whether to show plots or not", \
                    default=False)
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

frac_block = np.load(args.drug + "-all.npy")
n_exps = frac_block.shape[-1]
n_exps_concs = n_exps * np.ones(len(concs), dtype=int)

times = np.linspace(0, 249990, 25000)
times_full = np.linspace(0, 98990, 9900)

import matplotlib.pyplot as plt

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
n_sweeps = 10

dblocks = np.zeros((9900, len(concs), n_exps))

for j in range(n_exps):
    for k in range(len(concs)):
        if frac_block[:, k, :, j].all() == 0:
            n_exps_concs[k] = n_exps_concs[k] - 1
    for i in range(n_sweeps):
        for l in range(len(concs)):
            dblocks[i*990:(i+1)*990, l, j] = frac_block[i, l, 10:, j]

fig = plt.figure(figsize=(12, 6), dpi=150)
ax1 = fig.add_subplot(111)
ax1.set_title(args.drug)
ax1.set_ylim([0, 1.25])
ax1.set_xlim([-500, 100500])
for k in range(n_sweeps):
    for l in range(len(concs)):
        if k == 0:
            ax1.plot(times[k*1000:k*1000+990], dblocks[k*990:(k+1)*990, l, 0], linewidth=1, color=colors[l], \
                alpha=0.5, label='Conc: ' + str(concs[l]) + ' nM (N = ' + str(n_exps_concs[l]) + ')')
            ax1.plot(times[k*1000:k*1000+990], dblocks[k*990:(k+1)*990, l, 1:], linewidth=1, color=colors[l], \
                alpha=0.5)
        else:
            ax1.plot(times[k*1000:k*1000+990], dblocks[k*990:(k+1)*990, l, :], linewidth=1, color=colors[l], \
                alpha=0.5)
ax1.legend()
ax1.grid(True)
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Normalised current')
plt.tight_layout()
if args.show:
    plt.show()
else:
    plt.savefig(args.drug + '-individual-experiments.png')

