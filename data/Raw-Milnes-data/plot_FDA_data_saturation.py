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

cmax_dict = {'astemizole': 0.26, 'azimilide': 70, 'bepridil': 33, \
    'chlorpromazine': 38, 'cisapride': 2.6, 'clarithromycin': 1206, \
    'clozapine': 71, 'diltiazem': 122, 'disopyramide': 742, \
    'dofetilide': 2, 'domperidone': 19, 'droperidol': 6.33, \
    'ibutilide': 140, 'loratadine': 0.45, 'metoprolol': 1800, \
    'pimozide': 0.431, 'mexiletine': 4129, 'nifedipine': 7.7, \
    'nitrendipine': 3.02, 'ondansetron': 139, 'quinidine': 3237, \
    'ranolazine': 1948.2, 'risperidone': 1.81, 'sotalol': 14690, \
    'tamoxifen': 21, 'terfenadine': 4, 'vandetanib': 255.4, \
    'verapamil': 81}
cmax = 1e-9 * cmax_dict[args.drug]

frac_block = np.load(args.drug + "-all.npy")
n_exps = frac_block.shape[-1]
n_exps_concs = n_exps * np.ones(len(concs), dtype=int)

times = np.linspace(0, 249990, 25000)
times_full = np.linspace(0, 98990, 9900)

import matplotlib.pyplot as plt
plt.rcParams['axes.axisbelow'] = True

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

means = np.zeros(len(concs))
sems = np.zeros(len(concs))

fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_subplot(111)
ax1.set_title(args.drug)
ax1.set_xlabel('log [drug] (M)')
ax1.set_ylabel('Fractional block')
ax1.grid(True)
for l in range(len(concs)):
    mean = 0
    store = np.zeros(n_exps_concs[l])
    for k in range(n_exps_concs[l]):
        store[k] = np.mean(dblocks[9890:9900, l, k])
        mean += store[k]
    means[l] = 1.0 - (mean / n_exps_concs[l])
    sems[l] = np.std(store) / np.sqrt(n_exps_concs[l])

from scipy.optimize import curve_fit
def fsigmoid(x, k):
    return 1.0 / (1.0 + (k / pow(10, x)))

def fsigmoid_sat(x, k, k2):
    return k2 / (1.0 + (k / pow(10, x)))

def fsigmoid_hill(x, k, k2):
    return 1.0 / (1.0 + pow((k / pow(10, x)), k2))

def fsigmoid_both(x, k, k2, k3):
    return k3 / (1.0 + pow((k / pow(10, x)), k2))

nconcs = [i * 1e-9 for i in concs]
lconcs = np.log10(nconcs)
popt, pcov = curve_fit(fsigmoid, lconcs, means)#, bounds=([0., 600.],[0.01, 1200.]))

popt2, pcov2 = curve_fit(fsigmoid_sat, lconcs, means, bounds=([0, 0],[1, 1]))

popt3, pcov3 = curve_fit(fsigmoid_hill, lconcs, means, bounds=([0, 0.2],[1, 2]))

popt4, pcov4 = curve_fit(fsigmoid_both, lconcs, means, bounds=([0, 0.4, 0.5],[1, 2, 1]))

x = np.linspace(lconcs[0]-2, lconcs[-1]+2, 1000)
y = fsigmoid(x, *popt)
y2 = fsigmoid_sat(x, *popt2)
y3 = fsigmoid_hill(x, *popt3)
y4 = fsigmoid_both(x, *popt4)

ax1.plot(x, y, label=r'IC$_{50}$ = ' + str('%.1f' % (1e9 * popt)) + ' nM')
ax1.plot(x, y2, linestyle='dashed', label=r'IC$_{50}$ = ' + str('%.1f' % (1e9 * popt2[0])) + ' nM\nSaturation = ' + str('%.1f' % (100*popt2[-1])) + '%')
ax1.plot(x, y3, linestyle='dotted', label=r'IC$_{50}$ = ' + str('%.1f' % (1e9 * popt3[0])) + ' nM\nHill coefficient = ' + str('%.2f' % popt3[-1]))
ax1.plot(x, y4, linestyle='dashdot', label=r'IC$_{50}$ = ' + str('%.1f' % (1e9 * popt4[0])) + ' nM\nHill coefficient = ' + str('%.2f' % popt4[1]) + \
    '\nSaturation = ' + str('%.1f' % (100*popt4[-1])) + '%')
ax1.errorbar(lconcs, means, yerr=sems, fmt='o', capsize=2, color=colors[0])
ax1.axvline(np.log10(cmax), color='red', alpha=0.5, label=r'1$\times$Cmax')
ax1.axvline(np.log10(4*cmax), color='blue', alpha=0.5, label=r'4$\times$Cmax')
ax1.legend()
plt.tight_layout()

# np.save(args.drug + '-summary-data.npy', [means, sems])

if args.show:
    plt.show()
else:
    plt.savefig(args.drug + '-IC50.png')
