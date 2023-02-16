#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

drugs = ['astemizole', 'azimilide', 'bepridil', 'chlorpromazine', 'cisapride', 'clarithromycin', 'clozapine', \
    'diltiazem', 'disopyramide', 'dofetilide', 'domperidone', 'droperidol', 'ibutilide', 'loratadine', \
    'metoprolol', 'mexiletine', 'nifedipine', 'nitrendipine', 'ondansetron', 'pimozide',  'quinidine', \
    'ranolazine', 'risperidone', 'sotalol', 'tamoxifen', 'terfenadine', 'vandetanib', 'verapamil']

training = ['cisapride', 'mexiletine', 'dofetilide', 'sotalol', 'ranolazine', 'quinidine', 'diltiazem', \
    'bepridil', 'chlorpromazine', 'ondansetron', 'terfenadine', 'verapamil']

bootstrap_samples = np.zeros((len(drugs), 1000))

def box_plot(data, position, edge_color, fill_color):
    bp = ax.boxplot(data, positions=[position], patch_artist=True)
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color) 

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)

legend_dict = {'training' : 'red', 'validation' : 'blue'}

patchList = []
for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)

for i, d in enumerate(drugs):
    bootstrap_samples[i][:] = np.loadtxt(d + '-bootstrap-1000-samples.txt')
    if d in training:
        box_plot(bootstrap_samples[i], i, 'red', 'white')
    else:
        box_plot(bootstrap_samples[i], i, 'blue', 'white')

ax.legend(handles=patchList)
ax.set_ylabel('Error')
ax.set_xlabel('Compounds')
ax.set_xticklabels(drugs, rotation=90)
ax.grid(True)
plt.tight_layout()
plt.show()
