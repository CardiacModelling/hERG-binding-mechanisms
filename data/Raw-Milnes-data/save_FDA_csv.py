#!/usr/bin/env python3
import numpy as np
import argparse
import csv

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

frac_block = np.load(args.drug + ".npy")
frac_SD = np.load(args.drug + "-SD.npy")

times = np.linspace(0, 249990, 25000)
times_full = np.linspace(0, 98990, 9900)

import matplotlib.pyplot as plt

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
n_sweeps = 10

dblocks = np.zeros((9900, len(concs)))
dblocks_SD = np.zeros((9900, len(concs)))

for i in range(n_sweeps):
    for j in range(len(concs)):
        dblocks[i*990:(i+1)*990, j] = frac_block[i, j, 10:]
        dblocks_SD[i*990:(i+1)*990, j] = frac_SD[i, j, 10:]

plot = False
if plot:
    fig = plt.figure(dpi=150)
    ax1 = fig.add_subplot(111)
    ax1.set_title(args.drug)
    ax1.set_ylim([0, 1])
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Normalised current')
    for k in range(10):
        for j in range(len(concs)):
            ax1.plot(times[k*1000:k*1000+990], dblocks[k*990:(k+1)*990, j], color=colors[j])
            ax1.fill_between(times[k*1000:k*1000+990], dblocks[k*990:(k+1)*990, j] - dblocks_SD[k*990:(k+1)*990, j], \
                dblocks[k*990:(k+1)*990, j] + dblocks_SD[k*990:(k+1)*990, j], color=colors[j], alpha=0.25, linewidth=0)
    ax1.grid(True)
    plt.show()

for i, conc in enumerate(concs):
    filename = 'drug-' + args.drug + '-conc-' + str(conc) + '-Milnes'

    row = [times_full, dblocks[:, i]]
    row = np.column_stack(row)

    row_SD = [times_full, dblocks_SD[:, i]]
    row_SD = np.column_stack(row_SD)

    header = ['time', 'current']

    with open('../Milnes-data/' + filename + '.csv', 'a') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(header)
        writer.writerows(row)

    csvFile.close()

    with open('../Milnes-data/' + filename + '-SD.csv', 'a') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(header)
        writer.writerows(row_SD)

    csvFile.close()
