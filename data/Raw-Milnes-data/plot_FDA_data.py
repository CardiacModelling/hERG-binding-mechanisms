#!/usr/bin/env python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

drugs = ['astemizole', 'azimilide', 'bepridil', 'chlorpromazine', 'cisapride', 'clarithromycin', 'clozapine', \
    'diltiazem', 'disopyramide', 'dofetilide', 'domperidone', 'droperidol', 'ibutilide', 'loratadine', \
    'metoprolol', 'mexiletine', 'nifedipine', 'nitrendipine', 'ondansetron', 'pimozide',  'quinidine', \
    'ranolazine', 'risperidone', 'sotalol', 'tamoxifen', 'terfenadine', 'vandetanib', 'verapamil']

parser = argparse.ArgumentParser()
parser.add_argument("--drug", type=str, choices=drugs, help="which country to use", default='quinidine')
parser.add_argument("--show", action='store_true', help="whether to show plots or not", \
                    default=False)
parser.add_argument("--plot_conc", type=int, help="whether to show plots or not", \
                    default=4)
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

sweeps = np.linspace(1, 10, 10, dtype=int)
concs_dict = {'astemizole': astemizole_concs, 'azimilide': azimilide_concs, 'bepridil': bepridil_concs, \
    'chlorpromazine': chlorpromazine_concs, 'cisapride': cisapride_concs, 'clarithromycin': clarithromycin_concs, \
    'clozapine': clozapine_concs, 'diltiazem': diltiazem_concs, 'disopyramide': disopyramide_concs, \
    'dofetilide': dofetilide_concs, 'domperidone': domperidone_concs, 'droperidol': droperidol_concs, \
    'ibutilide': ibutilide_concs, 'loratadine': loratadine_concs, 'metoprolol': metoprolol_concs, \
    'pimozide': pimozide_concs, 'mexiletine': mexiletine_concs, 'ondansetron': ondansetron_concs, \
    'quinidine': quinidine_concs, 'ranolazine': ranolazine_concs, 'risperidone': risperidone_concs, \
    'sotalol': sotalol_concs, 'tamoxifen': tamoxifen_concs, 'terfenadine': terfenadine_concs, \
    'vandetanib': vandetanib_concs, 'verapamil': verapamil_concs}
concs = concs_dict[args.drug]

frac_block = np.load(args.drug + ".npy")

n_sweeps = 10

# Create colormap for plotting
cmap = matplotlib.cm.get_cmap('viridis')
norm = matplotlib.colors.Normalize(0, n_sweeps)

times = np.linspace(100, 9990, 990)

fig = plt.figure(figsize=(12, 8))
fig.suptitle(args.drug)
ax1 = fig.add_subplot(241)
ax1.set_ylim([0, 1])
ax1.set_title('Conc = ' + str(concs[0]) + ' nM')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Normalised current')
ax1.grid(True)
ax2 = fig.add_subplot(242)
ax2.set_ylim([0, 1])
ax2.set_title('Conc = ' + str(concs[1]) + ' nM')
ax2.set_xlabel('Time (ms)')
[label.set_visible(False) for label in ax2.get_yticklabels()]
ax2.grid(True)
ax3 = fig.add_subplot(243)
ax3.set_ylim([0, 1])
ax3.set_title('Conc = ' + str(concs[2]) + ' nM')
ax3.set_xlabel('Time (ms)')
[label.set_visible(False) for label in ax3.get_yticklabels()]
ax3.grid(True)
ax4 = fig.add_subplot(244)
ax4.set_ylim([0, 1])
ax4.set_title('Conc = ' + str(concs[3]) + ' nM')
ax4.set_xlabel('Time (ms)')
[label.set_visible(False) for label in ax4.get_yticklabels()]
ax4.grid(True)
ax5 = fig.add_subplot(2,1,2)
ax5.set_xlabel('Time (ms)')
ax5.set_ylabel('Normalised current')
ax5.grid(True)

for i in range(n_sweeps):
    ax1.plot(frac_block[i, 0, 5:], color=cmap(norm(i)), label='Sweep ' + str(i+1))
    ax2.plot(frac_block[i, 1, 5:], color=cmap(norm(i)))
    ax3.plot(frac_block[i, 2, 5:], color=cmap(norm(i)))
    ax4.plot(frac_block[i, 3, 5:], color=cmap(norm(i)))
    ax5.plot(times + 10000*i, frac_block[i, args.plot_conc - 1, 10:], color=cmap(norm(i)))
ax1.legend()
plt.tight_layout()
if args.show:
    plt.show()
else:
    plt.savefig(args.drug + '.png')

