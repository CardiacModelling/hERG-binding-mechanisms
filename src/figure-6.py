#!/usr/bin/env python3
import sys
sys.path.append('..')
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
import seaborn as sns
sns.set_theme()

import methods.parameters as parameters

from methods import results
cache = os.path.join(results, 'qnet')

results = 'figures'
prefix = 'figure-6'

results = os.path.join(results, 'figure-6')
if not os.path.isdir(results):
    os.makedirs(results)

compounds_1 = ['diltiazem', 'nifedipine', 'verapamil']
Compounds_1 = ['Diltiazem', 'Nifedipine', 'Verapamil']
compounds_2 = ['dofetilide', 'cisapride', 'bepridil']
Compounds_2 = ['Dofetilide', 'Cisapride', 'Bepridil']
all_compounds = [compounds_1, compounds_2]
All_Compounds = [Compounds_1, Compounds_2]

ap_model_name = 'dutta'

x_cmax = [0, 0.5, 1, 5, 10, 15, 20, 25]

exception = {
    'quinidine': [0, 1]
}

model_list = [f'{i}' for i in ['0a', '0b']]
model_list += [f'{i}' for i in range(1, 3)]
model_list += [f'{i}' for i in ['2i']]
model_list += [f'{i}' for i in range(3, 6)]
model_list += [f'{i}' for i in ['5i']]
model_list += [f'{i}' for i in range(6, 14)]

#colours = sns.color_palette('Set3', n_colors=len(model_list))
#colours = sns.husl_palette(n_colors=len(model_list)+2, h=.5, s=.6)
colours = sns.color_palette(cc.glasbey_category10, n_colors=len(model_list))
#colours = sns.color_palette(cc.glasbey_hv, n_colors=len(model_list))

for base_model in ['li', 'lei']:
    if base_model == 'lei':
        model_names = [f'm{m}' for m in model_list]
    else:
        model_names = [f'{base_model}-m{m}' for m in model_list]
    exclude_model_list = parameters.exclude_model_list[base_model]

    model_names_with_default = ['li'] + model_names

    if base_model == 'li':
        #''' # manual patch data threshold
        tms_u = 0.0579
        tms_l = 0.0689
        ''' # auto/manual patch data threshold
        tms_u = 0.0581
        tms_l = 0.0671
        #'''
    elif base_model == 'lei':
        # NOTE: Match output naming in ordinal-logistic-regression.py
        print('Loading thresholds from cache (ordinal-logistic-regression.py)')
        f = os.path.join(cache, '..', 'ordinal-logistic/thresholds')
        f += f'-{base_model}.txt'
        tms_l, tms_u = np.loadtxt(f)

    qnet = {}
    # NOTE: Match output naming in compare-qnets.py
    print('Loading qNet from cache (compare-qnets.py)')
    for model_name in model_names_with_default:
        f = f'qnet-{ap_model_name}-{model_name}.pkl'
        with open(os.path.join(cache, f), 'rb') as f:
            qnet[model_name] = pickle.load(f)
    for model_name in model_names_with_default:
        f = f'qnet-validation-{ap_model_name}-{model_name}.pkl'
        with open(os.path.join(cache, f), 'rb') as f:
            tmp = pickle.load(f)
        for compound in tmp:
            qnet[model_name][compound] = tmp[compound]
        del(tmp)

    # Plot
    sns.set_context('paper')
    sns.set_style('ticks')

    fig, axes = plt.subplots(len(all_compounds), len(compounds_1),
                             figsize=(8.5, 5), sharex=True, sharey=True)

    for i in range(len(all_compounds)):
        axes[i, 0].set_ylabel(r'$q_\mathrm{net}$ ($C/F$)', fontsize=11)
    for j in range(len(compounds_1)):
        axes[-1, j].set_xlabel(r'$\times$ $C_\mathrm{max}$', fontsize=11)

    for ii, compounds in enumerate(all_compounds):
        for jj, compound in enumerate(compounds):
            ax = axes[ii, jj]
            ax.set_title(f'{All_Compounds[ii][jj]}', loc='left', fontsize=11)

            if compound in exception:
                x_cmax_ = [x_cmax[k] for k in exception[compound]]
            else:
                x_cmax_ = x_cmax

            for i_m, model_name in enumerate(model_names):
                q = []
                for x in x_cmax_:
                    q.append(qnet[model_name][compound][x])
                if model_name in exclude_model_list[compound]:
                    ax.plot(x_cmax_, q, 'o:', alpha=0.2, c=colours[i_m],
                            label=model_list[i_m])
                else:
                    ax.plot(x_cmax_, q, 'o-', alpha=0.5, c=colours[i_m],
                            label=model_list[i_m])

            if base_model == 'li':
                q0 = []
                for x in x_cmax_:
                    q0.append(qnet['li'][compound][x])

                ax.plot(x_cmax_, q0, 'o--', c='k', alpha=0.75, lw=1.5,
                        label='CiPA v1.0')

    # Shade the background according to the tms decision boundaries
    ymin, ymax = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(ymin, ymax)
    for ii, compounds in enumerate(all_compounds):
        for jj, compound in enumerate(compounds):
            ax = axes[ii, jj]
            kwargs = dict(ec='none', zorder=-1)
            ax.axhspan(ymin, tms_u, alpha=0.15, color='C3', **kwargs)
            ax.axhspan(tms_u, tms_l, alpha=0.25, color='C0', **kwargs)
            ax.axhspan(tms_l, ymax, alpha=0.15, color='C2', **kwargs)
            kwargs = dict(ls='--', lw=1.5, alpha=0.5, zorder=0)
            ax.axhline(tms_u, color='#7f7f7f', **kwargs)
            ax.axhline(tms_l, color='#7f7f7f', **kwargs)

    fig.tight_layout(rect=(0, 0, 1, 0.925))

    axes[0, 0].legend(loc='lower left', bbox_to_anchor=(-0.02, 1.15), ncol=9)

    sns.despine(fig=fig)
    sns.set(rc={'axes.facecolor':'none'})

    fig.savefig(os.path.join(results, f'{prefix}-{base_model}'), dpi=300)
    fig.savefig(os.path.join(results, f'{prefix}-{base_model}.pdf'),
                format='pdf')
    plt.close(fig)

