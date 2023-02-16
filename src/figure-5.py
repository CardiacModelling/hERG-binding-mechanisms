#!/usr/bin/env python3
import sys
sys.path.append('..')
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
sns.set_context('paper')
sns.set_style('ticks')

import methods.parameters as parameters
from methods import results

results = 'figures'
prefix = 'figure-5'

results = os.path.join(results, 'figure-5')
if not os.path.isdir(results):
    os.makedirs(results)

compounds = ['verapamil', 'metoprolol']#'diltiazem']
Compounds = ['Verapamil', 'Metoprolol']#'Diltiazem']

colors = ['#9467bd', '#8c564b', '#d62728']
markers = ['s', 'o']

model_list = [f'{i}' for i in ['0b']]
model_list += [f'{i}' for i in range(1, 3)]
model_list += [f'{i}' for i in ['2i']]
model_list += [f'{i}' for i in range(3, 6)]
model_list += [f'{i}' for i in ['5i']]
model_list += [f'{i}' for i in range(6, 14)]

parameter_names = ['kon', 'koff', 'hill']
# NOTE: ku, kt here refers to m11 (not Li et al. 2017 model)
excluded_parameters = ['k2f', 'ku' ,'kt', 'halfmax', 'vhalf']
# >>> Parameter naming consistency
kon_variants = ['kforward', 'konO', 'konI', 'kon_o', 'kon_i']
kmax_variants = ['kmax', 'Kmax']  # kon = kmax * ku
koff_variants = ['ku', 'Ku', 'koffO', 'koffI', 'koff_o', 'koff_i']
vhalf_variants = ['Vhalf']
kt_variants = ['Kt']
# <<<

xlabel = 'Binding model'
'''
if base_model == 'li':
    xlabel += ' (base model A)'
elif base_model == 'lei':
    xlabel += ' (base model B)'
else:
    raise ValueError(f'Unexpected base model {base_model}')
'''
xticks_loc = np.arange(len(model_list) + 1)
xticks_value = model_list + ['CiPA v1'] #+ ['Li et al.']
xticks_value[0] = '0a/b'
#for i in [1, 4, 8, -4, -2]:
for i in range(1, len(xticks_value), 2):
    xticks_value[i] = '\n' + xticks_value[i]

fig, axes = plt.subplots(3, 2, figsize=(8.5, 5), sharex=True)

# NOTE: match parameter_names order
axes[0, 0].set_ylabel(r'$k_\mathrm{on}$ (ms$^{-1}$nM$^{-n}$)', fontsize=11)
axes[1, 0].set_ylabel(r'$k_\mathrm{off}$ (ms$^{-1}$)', fontsize=11)
axes[2, 0].set_ylabel(r'$n$', fontsize=11)

for i_c, compound in enumerate(compounds):
    axes[-1, i_c].set_xticks(xticks_loc, xticks_value)
    axes[-1, i_c].set_xlabel(xlabel, fontsize=11)
    axes[0, i_c].set_title(f'{Compounds[i_c]}', loc='left', fontsize=11)

    thetas = [[], [], []]  # Calculate the spread
    for i_b, base_model in enumerate(['li', 'lei']):
        if base_model == 'lei':
            model_names = [f'm{m}' for m in model_list]
        else:
            model_names = [f'{base_model}-m{m}' for m in model_list]
        if i_b == 1: # NOTE
            model_names_with_default = model_names + ['li']
        else:
            model_names_with_default = model_names
        exclude_model_list = parameters.exclude_model_list[base_model]

        for i_m, model_name in enumerate(model_names_with_default):
            binding_names = [x.split('.')[-1] for x in
                             parameters._model_binding[model_name]]
            binding_params = parameters.binding[model_name][compound]
            # >>> Goodness of fits of the model
            marker = markers[i_b]
            if model_name in exclude_model_list[compound]:
                alpha = 0.25
                ec = '#7f7f7f'
            else:
                alpha = 1
                ec = colors[i_b]
            if model_name == 'li':
                alpha = 1
                marker = '*'
                ec = colors[-1]
            # <<<
            for n, v in zip(binding_names, binding_params):
                # >>> Parameter naming consistency
                if n in ['konI', 'koffI', 'kon_i', 'koff_i']:
                    solid = True
                else:
                    solid = False
                if n in kon_variants:
                    n = 'kon'
                if n in koff_variants and ('m11' not in model_name):
                    n = 'koff'
                if n in kmax_variants:
                    n = 'kon'
                    if 'ku' in binding_names:
                        v *= binding_params[binding_names.index('ku')]
                    elif 'Ku' in binding_names:
                        v *= binding_params[binding_names.index('Ku')]
                    else:
                        raise ValueError('Expecting kon = kmax * ku')
                if n in vhalf_variants:
                    n = 'vhalf'
                if n in kt_variants:
                    n = 'kt'
                # <<<
                if 'm0a' in model_name:
                    if n == 'kon': continue
                    if n == 'koff': continue
                if 'm12' in model_name or model_name == 'li':
                    if n == 'kon': continue
                if n not in excluded_parameters:
                    idx = parameter_names.index(n)
                    if i_m == 2 and base_model == 'li':  # Not m0a,b
                        b, l = False, 'Physiological model A'
                    elif i_m == 2 and base_model == 'lei':  # Not m0a,b
                        b, l = False, 'Physiological model B'
                    elif model_name == 'li':
                        b, l = False, 'Li et al.'
                    else:
                        b = True
                    color = ec if solid else 'none'
                    alpha = alpha * 0.65 if solid else alpha
                    axes[idx, i_c].scatter(i_m, v, alpha=alpha, marker=marker,
                            color=color, linewidth=1.5, edgecolor=ec,
                            label='_' if b else l)
                    # Keep parameters to calculate the spread
                    if model_name not in exclude_model_list[compound]:
                        if ('m12' in model_name) or (model_name == 'li'):
                            pass
                        else:
                            thetas[idx].append(v)
    if '-v' in sys.argv or '--verbose' in sys.argv:
        tmp = ['kon', 'koff', 'n']
        print(compound)
        for i, theta in enumerate(thetas):
            print(tmp[i], 'coef of var', np.std(theta)/np.mean(theta),
                  f'(mean {np.mean(theta)}, std {np.std(theta)})')
    axes[0, i_c].set_yscale('log')
    axes[1, i_c].set_yscale('log')

axes[0, 1].legend(loc='lower right', bbox_to_anchor=(1.015, 1.01), ncol=1,
                  fontsize=8)

# Despine and add grid
sns.despine(fig=fig)
for i in [0, 1]:
    for j in [0, 1]:
        axes[i, j].tick_params(axis='x', color='none')
        axes[i, j].spines['bottom'].set_visible(False)
for iy, ix in np.ndindex(axes.shape):
    axes[iy, ix].grid(axis='x')
sns.set(rc={'axes.facecolor':'none', 'grid.color':'#CACAD2'})

fig.tight_layout()
path = os.path.join(results, f'{prefix}')
fig.savefig(path, dpi=300)
fig.savefig(path+'.pdf', format='pdf')
plt.close(fig)
