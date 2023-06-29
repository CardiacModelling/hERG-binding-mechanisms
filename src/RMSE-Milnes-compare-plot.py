#!/usr/bin/env python3
import sys
sys.path.append('..')
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.axisbelow'] = True

from methods import results

results = os.path.join(results, 'RMSE-Milnes')
if not os.path.isdir(results):
    os.makedirs(results)

path2fits = '../data/Milnes-data-fits'
path2fitsre = '../data/Milnes-data-fits-re'
path2data = '../data/Raw-Milnes-data'

drugs = ['astemizole', 'azimilide', 'bepridil', 'chlorpromazine', 'cisapride', 'clarithromycin', 'clozapine', \
    'diltiazem', 'disopyramide', 'dofetilide', 'domperidone', 'droperidol', 'ibutilide', 'loratadine', \
    'metoprolol', 'mexiletine', 'nifedipine', 'nitrendipine', 'ondansetron', 'pimozide',  'quinidine', \
    'ranolazine', 'risperidone', 'sotalol', 'tamoxifen', 'terfenadine', 'vandetanib', 'verapamil']

# Check input arguments
parser = argparse.ArgumentParser(
    description='Plot model and experimental data')
parser.add_argument("--drug", type=str, choices=drugs, default='quinidine')
parser.add_argument("--all", action='store_true', help="Plot all drugs, this will ignore --drug argument", \
    default=False)
parser.add_argument("--show", action='store_true', help="Show plots", \
    default=False)
parser.add_argument("--n_samples", type=int, help="Number of Bootstrap samples to use", \
    default=1000)
args = parser.parse_args()


baseline_models = ['lei', 'fda']

xticks_val = [r'0$\alpha$', r'0$\beta$', 1, 2, '2i', 3, 4, 5, '5i', 6, 7, 8, 9, 10, 11, 12, 13, '']
for i in range(1, len(xticks_val), 2):
    xticks_val[i] = '\n' + str(xticks_val[i])
nmodels = len(xticks_val) - 1
RMSEs = np.zeros((len(baseline_models), nmodels))
models = np.linspace(1, nmodels + 1, nmodels + 1, dtype=int)

if args.all:
    plot_drugs = drugs
else:
    plot_drugs = [args.drug]

for drug in plot_drugs:

    for i, m in enumerate(models[:13]):
        model_str = 'model' + str(m)

        for a, b in enumerate(baseline_models):
            RMSE = np.loadtxt(path2fits + '/Drug-' + drug + '-' + b + '-' + model_str + \
                '-protocol-Milnes-fit-RMSE-errors.txt')
            if (RMSE.size > 1):
                RMSE = RMSE[0]
            RMSEs[a][i] = RMSE
    # NOTE: Match xticks_val[:-1]
    RMSEs[:, 9:] = RMSEs[:, 5:13]
    RMSEs[:, 5:8] = RMSEs[:, 2:5]
    RMSEs[:, 2:4] = RMSEs[:, :2]
    for i, m in zip([0, 1, 4, 8], ['0a', '0b', '2i', '5i']):
        for a, b in enumerate(baseline_models):
            if b == 'lei':
                RMSE = np.loadtxt(path2fitsre + f'/{drug}-m{m}-errors.txt')
            elif b == 'fda':
                RMSE = np.loadtxt(path2fitsre + f'/{drug}-li-m{m}-errors.txt')
            else:
                raise ValueError(f'Unexpected model {b}')
            if (RMSE.size > 1):
                RMSE = RMSE[0]
            RMSEs[a][i] = RMSE

    FDA_RMSE = np.loadtxt(path2fits + '/FDA/' + drug + '-RMSE.txt')
    expt_samples = np.loadtxt(path2data + '/' + drug + '-bootstrap-' + str(args.n_samples) + '-samples.txt')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

    xs = np.random.normal(1.0, 0.05, len(expt_samples))

    show_all_samples = False

    fig, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 4]}, figsize=(8.5, 4), sharey=True)
    plt.title(drug)
    B = a0.boxplot(expt_samples, sym='', whis=10)
    Q0 = np.min([item.get_ydata()[1] for item in B['whiskers']])
    Q4 = np.max([item.get_ydata()[1] for item in B['whiskers']])
    if show_all_samples:
        for d, e in enumerate(expt_samples):
            if e >= Q0 and e <= Q4:
                a0.scatter(xs[d], e, alpha=0.2, color='dodgerblue')
            else:
                a0.scatter(xs[d], e, alpha=0.2, color='red')
        a0.scatter(xs, expt_samples, alpha=0.2, color='dodgerblue')
    a0.axhline(Q4, color='silver', linestyle='dashed', zorder=-1)
    a0.set_ylabel('RMSD')
    a0.set_xticklabels(['Bootstrap\nsamples'])
    a0.grid(True)
    a1.scatter(models[0], RMSEs[1][0], marker='s', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[0], label='Physiological model A')
    a1.scatter(models[0], RMSEs[0][0], marker='o', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[0], label='Physiological model B')
    #a1.scatter(models[0], RMSEs[2][0], marker='d', s=100, color='none', linewidth=1.5, \
    #    edgecolor='k', label='Whittaker model')
    a1.scatter(models[1], RMSEs[1][1], marker='s', s=100, color='none', linewidth=1.5, \
        edgecolor='none', label='\n')

    #a1.scatter(models[1], RMSEs[0][1], marker='o', s=100, color='none', linewidth=1.5, \
    #    edgecolor='k', label=r'2 params: $k_{\rm{on}}$, $k_{\rm{off}}$')
    a1.scatter(models[1:9], RMSEs[0][1:9], marker='o', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[0], label=r'3 params: $k_{\rm{on}}$, $k_{\rm{off}}$, $n$')
    a1.scatter(models[9], RMSEs[0][9], marker='o', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[1], label=r'4 params: $k_{\rm{on}}$, $k_{\rm{off}}$, $k_{\rm{trap}}$, $n$')
    a1.scatter(models[10], RMSEs[0][10], marker='o', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[2], label=r'5 params: $k_{\rm{on, O}}$, $k_{\rm{off, O}}$, $k_{\rm{on, I}}$, $k_{\rm{off, I}}$, $n$')
    a1.scatter(models[11:13], RMSEs[0][11:13], marker='o', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[3], label=r'4 params: $k_{\rm{on, O}}$, $k_{\rm{off, O}}$, $k_{\rm{on, I}}$, $n$')
    a1.scatter(models[13], RMSEs[0][13], marker='o', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[4], label=r'5 params: $k_{\rm{on, O}}$, $k_{\rm{off, O}}$, $k_{\rm{on, I}}$, $k_{\rm{trap}}$, $n$')
    a1.scatter(models[14], RMSEs[0][14], marker='o', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[5], label=r'5 params: $k_{\rm{on}}$, $k_{\rm{off}}$, $k_{\rm{untrap}}$, $k_{\rm{trap}}$, $n$')
    a1.scatter(models[15], RMSEs[0][15], marker='o', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[6], label=r'5 params: $\hat{k}_{\rm{on}}$, $k_{\rm{off}}$, $\rm{EC}_{50}$, $n$, $V_{\rm{1/2, trap}}$')
    a1.scatter(models[16], RMSEs[0][16], marker='o', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[7], label=r'4 params: $k_{\rm{on}}$, $k_{\rm{off}}$, $n$, $V_{\rm{1/2, trap}}$')

    #a1.scatter(models[1], RMSEs[1][1], marker='s', s=100, color='none', linewidth=1.5, \
    #    edgecolor='k')
    a1.scatter(models[1:9], RMSEs[1][1:9], marker='s', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[0])
    a1.scatter(models[9], RMSEs[1][9], marker='s', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[1])
    a1.scatter(models[10], RMSEs[1][10], marker='s', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[2])
    a1.scatter(models[11:13], RMSEs[1][11:13], marker='s', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[3])
    a1.scatter(models[13], RMSEs[1][13], marker='s', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[4])
    a1.scatter(models[14], RMSEs[1][14], marker='s', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[5])
    a1.scatter(models[15], RMSEs[1][15], marker='s', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[6])
    a1.scatter(models[16], RMSEs[1][16], marker='s', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[7])

    '''
    #a1.scatter(models[1], RMSEs[2][1], marker='d', s=100, color='none', linewidth=1.5, \
    #    edgecolor='k')
    a1.scatter(models[1:9], RMSEs[2][1:9], marker='d', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[0])
    a1.scatter(models[9], RMSEs[2][9], marker='d', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[1])
    a1.scatter(models[10], RMSEs[2][10], marker='d', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[2])
    a1.scatter(models[11:13], RMSEs[2][11:13], marker='d', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[3])
    a1.scatter(models[13], RMSEs[2][13], marker='d', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[4])
    a1.scatter(models[14], RMSEs[2][14], marker='d', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[5])
    a1.scatter(models[15], RMSEs[2][15], marker='d', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[6])
    a1.scatter(models[16], RMSEs[2][16], marker='d', s=100, color='none', linewidth=1.5, \
        edgecolor=colors[7])
    '''

    a1.scatter(models[nmodels], FDA_RMSE, marker='*', s=100, color='none', linewidth=1.5, \
        edgecolor='magenta', label=r'5 params: CiPA v1.0 model')
    a1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=7.5)
    a1.axhline(Q4, color='silver', linestyle='dashed', zorder=-1)
    a1.set_xlabel('Models')
    a1.set_xticks(models, xticks_val)
    a1.grid(True)
    plt.tight_layout()
    if args.show:
        plt.show()
    else:
        plt.savefig(os.path.join(results, 'RMSE-' + drug + '-all-models'),
                    dpi=300)
    plt.close()
