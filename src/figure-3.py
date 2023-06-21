#!/usr/bin/env python3
import sys
sys.path.append('..')
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
sns.set_context('paper')
sns.set_style('ticks')

import methods.models as models
import methods.parameters as parameters

results = 'figures'
prefix = 'figure-3'

results = os.path.join(results, 'figure-3')
if not os.path.isdir(results):
    os.makedirs(results)

CACHE = ('--cache' in sys.argv) or ('-x' in sys.argv)

protocol = '../protocols/protocol-Milnes.mmt'
times = np.arange(0, 15e3, 10)
win = (times >= 1.1e3) & (times < 11e3)

compounds = ['dofetilide', 'terfenadine', 'verapamil']
Compounds = ['Dofetilide', 'Terfenadine', 'Verapamil']

base_model = 'li'
if base_model == 'lei':
    model_names = [f'm{i}' for i in range(1, 14)]
    model_names += [f'm{i}' for i in ['0a', '0b', '2i', '5i']]
else:
    model_names = [f'{base_model}-m{i}' for i in range(1, 14)]
    model_names += [f'{base_model}-m{i}' for i in ['0a', '0b', '2i', '5i']]
exclude_model_list = parameters.exclude_model_list[base_model]

path2cache = os.path.join(results, f'{prefix}-cache')
if not os.path.isdir(path2cache):
    os.makedirs(path2cache)
if not CACHE:
    milnes = {}

    # Go through all compounds
    for compound in compounds:
        milnes[compound] = {}

        print(f'Simulating the Milnes protocol for {compound}')
        conc = parameters.drug_concs[compound]
        colors = [f'C{i}' for i in range(len(conc))]

        # Go through all models
        for i_m, model_name in enumerate(model_names):
            milnes[compound][model_name] = {}

            # Model
            model = models.Model(model_name,
                                 protocol,
                                 parameters=['binding'],
                                 analytical=True)
            binding_params = parameters.binding[model_name][compound]

            # Control
            model.set_dose(0)
            before = model.simulate(binding_params, times)

            # Go through all concentrations
            for c, color in zip(conc, colors):
                model_milnes = []

                # Compound
                model.set_dose(c)
                after = model.simulate(binding_params, times)
                after[before < 1e-1] = np.NaN
                after[before >= 1e-1] /= before[before >= 1e-1]
                model_milnes = np.append(model_milnes, after[win])
                for i in range(9):
                    after = model.simulate(binding_params, times, reset=False)
                    after[before < 1e-1] = np.NaN
                    after[before >= 1e-1] /= before[before >= 1e-1]
                    model_milnes = np.append(model_milnes, after[win])

                milnes[compound][model_name][c] = model_milnes

        # Get time from data (do it once only)
        u = np.loadtxt(
            f'../data/Milnes-data/drug-{compound}-conc-{c}-Milnes.csv',
            delimiter=',',
            skiprows=1
        )
        t = u[:, 0]
        milnes[compound]['times'] = t

        with open(f'{path2cache}/{base_model}-{compound}.pkl', 'wb') as f:
            pickle.dump(milnes[compound], f)

else:
    print('Loading the Milnes protocol from cache')
    milnes = {}

    # Go through all compounds
    for compound in compounds:
        with open(f'{path2cache}/{base_model}-{compound}.pkl', 'rb') as f:
            milnes[compound] = pickle.load(f)

# Visualise

fig, axes = plt.subplots(len(compounds), 2, figsize=(8.5, 6.75))

nmodels = 13 + 2 + 2
models = np.arange(1, nmodels - 3, 1, dtype=int)
baseline_models = ['lei', 'fda']
RMSEs = np.zeros((len(baseline_models), nmodels))
path2fits = '../data/Milnes-data-fits'
path2fitsre = '../data/Milnes-data-fits-re'
path2data = '../data/Raw-Milnes-data'
exclude_1 = parameters.exclude_model_list['lei']
exclude_2 = parameters.exclude_model_list['li']
model_names_tmp = [f'm{i}' for i in ['0a', '0b']]
model_names_tmp += [f'm{i}' for i in range(1, 3)]
model_names_tmp += [f'm{i}' for i in ['2i']]
model_names_tmp += [f'm{i}' for i in range(3, 6)]
model_names_tmp += [f'm{i}' for i in ['5i']]
model_names_tmp += [f'm{i}' for i in range(6, 14)]
xticks_loc = np.arange(nmodels + 2) + 1
xticks_val = ['\nBootstrap', r'0$\alpha$', r'0$\beta$', 1, 2, '2i', 3, 4, 5, '5i', 6, 7, 8, 9,
              10, '\n11', 12, '\n13', '']
#colors = ['#9467bd', '#8c564b', '#d62728']
colors = ['#66c2a5', '#fc8d62', '#d62728']
for i_c, compound in enumerate(compounds):
    ax = axes[i_c, 1]
    #ax.set_title(Compounds[i_c], loc='left')

    for i, m in enumerate(models):
        model_str = 'model' + str(m)

        for a, b in enumerate(baseline_models):
            RMSE = np.loadtxt(path2fits + '/Drug-' + compound + '-' + b + '-' 
                    + model_str + '-protocol-Milnes-fit-RMSE-errors.txt')
            if (RMSE.size > 1):
                RMSE = RMSE[0]
            RMSEs[a][i] = RMSE
    # NOTE: Match model_names_tmp
    RMSEs[:, 9:] = RMSEs[:, 5:len(models)]
    RMSEs[:, 5:8] = RMSEs[:, 2:5]
    RMSEs[:, 2:4] = RMSEs[:, :2]
    for i, m in zip([0, 1, 4, 8], ['0a', '0b', '2i', '5i']):
        for a, b in enumerate(baseline_models):
            if b == 'lei':
                RMSE = np.loadtxt(path2fitsre + f'/{compound}-m{m}-errors.txt')
            elif b == 'fda':
                RMSE = np.loadtxt(path2fitsre +
                                  f'/{compound}-li-m{m}-errors.txt')
            else:
                raise ValueError(f'Unexpected model {b}')
            if (RMSE.size > 1):
                RMSE = RMSE[0]
            RMSEs[a][i] = RMSE

    FDA_RMSE = np.loadtxt(path2fits + '/FDA/' + compound + '-RMSE.txt')
    expt_samples = np.loadtxt(path2data + '/' + compound
                              + '-bootstrap-1000-samples.txt')


    xs = np.random.normal(1.0, 0.05, len(expt_samples))

    B = ax.boxplot(expt_samples, sym='x', widths=(.7))
    Q4 = np.max([item.get_ydata()[1] for item in B['whiskers']])

    for i in range(len(model_names_tmp)):
        # Li et al.
        boo = 'li-' + model_names_tmp[i] in exclude_2[compound]
        c = '#7f7f7f' if boo else colors[0]
        a = 0.25 if boo else 0.9
        ax.scatter(i + 2, RMSEs[1][i], marker='s', color='none', s=40,
                linewidth=1.5, edgecolor=c, alpha=a)
        # Lei et al.
        boo = model_names_tmp[i] in exclude_1[compound] 
        c = '#7f7f7f' if boo else colors[1]
        a = 0.25 if boo else 0.9
        ax.scatter(i + 2, RMSEs[0][i], marker='o', color='none',
                linewidth=1.5, edgecolor=c, alpha=a)

    ax.scatter(len(model_names_tmp) + 2, FDA_RMSE, marker='*', color='none',
            alpha=0.9, linewidth=1.5, edgecolor=colors[-1])

    ax.axhline(Q4, color='C3', linestyle='dashed')
    #ax.set_ylabel('RMSD')
    if i_c == len(compounds) - 1:
        ax.set_xticks(xticks_loc, xticks_val)
        ax.set_xlabel('Binding model', fontsize=11)
    else:
        ax.set_xticks(xticks_loc, ['']*(nmodels + 2))
    ax.grid(axis='x')

# Legend
ax = axes[0, 1]
ax.scatter(1, np.NaN, marker='s', color='none', lw=1.5, ec=colors[0],
           label='Phyiological model A') # Li et al.
ax.scatter(1, np.NaN, marker='o', color='none', lw=1.5, ec=colors[1],
           label='Phyiological model B') # Lei et al.
ax.scatter(1, np.NaN, marker='*', color='none', lw=1.5, ec=colors[-1],
           label='CiPA v1.0 model')
ax.scatter(1, np.NaN, marker='o', color='none', lw=1.5, ec='#7f7f7f',
           label='Implausible model')
ax.scatter(1, np.NaN, marker='s', color='none', lw=1.5, ec='#7f7f7f',
           label='Implausible model')
#ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2,
#          fontsize=7)
ax.legend(loc=1, ncol=2, fontsize=8, columnspacing=1.2)

t0 = times[win]
t0 -= t0[0]
t_1 = t0[-1]
skip = 2
for i_c, compound in enumerate(compounds):
    conc = parameters.drug_concs[compound]
    colors = [f'C{i}' for i in range(len(conc))]
    ax = axes[i_c, 0]

    # Data
    for c, color in zip(conc, colors):
        u = np.loadtxt(
            f'../data/Milnes-data/drug-{compound}-conc-{c}-Milnes.csv',
            delimiter=',',
            skiprows=1
        )
        t = u[::skip, 0]
        u = u[::skip, 1]
        s = np.loadtxt(
            f'../data/Milnes-data/drug-{compound}-conc-{c}-Milnes-SD.csv',
            delimiter=',',
            skiprows=1
        )[::skip, 1]

        ax.plot(t, u, alpha=0.5, c=color, label=f'{c} nM')
        ax.fill_between(t, u - s, u + s, alpha=0.25, color=color, lw=0)

    # Model
    for i_m, model_name in enumerate(model_names):
        for c, color in zip(conc, colors):
            model_milnes = milnes[compound][model_name][c][::skip]

            # Plot
            if model_name in exclude_model_list[compound]:
                ls = '-'
                alpha = 0.5
                color = '#7f7f7f'
            else:
                ls = '-'
                alpha = 0.75
            ax.plot(t, model_milnes, alpha=alpha, c=color, ls=ls)

    xticks = []
    for i in range(10):
        xticks.append((i + 0.5) * t0[-1])
        if i % 2:
            ax.axvspan(i * t0[-1], (i + 1) * t0[-1], color='#CACAD2',
                        linestyle='', alpha=0.5)

    ax.set_ylim([0, 1.1])
    ax.set_xlim([0, t0[-1] * 10])
    ax.set_ylabel(Compounds[i_c] + '\n', fontsize=11)
    #ax.set_title(compound, loc='left')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=4,
              fontsize=8)
    if i_c == len(compounds) - 1:
        ax.set_xlabel('Pulse number', fontsize=11)
        ax.set_xticks(xticks, np.arange(1, 11))
    else:
        ax.set_xticks(xticks, ['']*len(xticks))

fig.align_xlabels(axes[-1, :])

# Titles
axes[0, 0].text(0.5, 1.25, 'Percentage current', ha='center', va='bottom',
                transform=axes[0, 0].transAxes, fontsize=11)
axes[0, 1].text(0.5, 1.25, 'RMSD', ha='center', va='bottom',
                transform=axes[0, 1].transAxes, fontsize=11)

# Despine
sns.despine(fig=fig)
sns.set(rc={'axes.facecolor':'none', 'grid.color':'#CACAD2'})
for i in [0, 1]:
    for j in [0, 1]:
        axes[i, j].tick_params(axis='x', color='none')
        axes[i, j].spines['bottom'].set_visible(False)

fig.tight_layout(h_pad=0.5)
fig.savefig(os.path.join(results, f'{prefix}'), dpi=300)
fig.savefig(os.path.join(results, f'{prefix}.pdf'), format='pdf')
plt.close(fig)

