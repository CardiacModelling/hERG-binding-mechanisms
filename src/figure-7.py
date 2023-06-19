#!/usr/bin/env python3
import sys
sys.path.append('..')
import os
import pickle
import numpy as np
import myokit
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
sns.set_context('paper')
sns.set_style('ticks')

import methods.parameters as parameters

from methods import results
cache = os.path.join(results, 'torsade-metric')

results = 'figures'
prefix = 'figure-7'

results = os.path.join(results, 'figure-7')
if not os.path.isdir(results):
    os.makedirs(results)

compounds_t = list(parameters._drug_training_list)
classes_t = dict(parameters._drug_training_classes)
compounds_v = list(parameters._drug_validation_list)
classes_v = dict(parameters._drug_validation_classes)
compounds_selected_t = ['dofetilide', 'terfenadine', 'diltiazem']
Compounds_selected_t = ['Dofetilide', 'Terfenadine', 'Diltiazem']
compounds_selected_v = ['vandetanib', 'domperidone', 'pimozide', 'nifedipine']
Compounds_selected_v = ['Vandetanib', 'Domperidone', 'Pimozide', 'Nifedipine']

ap_model_name = 'dutta'

model_list = [f'{i}' for i in ['0a', '0b']]
model_list += [f'{i}' for i in range(1, 3)]
model_list += [f'{i}' for i in ['2i']]
model_list += [f'{i}' for i in range(3, 6)]
model_list += [f'{i}' for i in ['5i']]
model_list += [f'{i}' for i in range(6, 14)]

base_model = 'li'
if base_model == 'lei':
    model_names = [f'm{m}' for m in model_list]
else:
    model_names = [f'{base_model}-m{m}' for m in model_list]
exclude_model_list = parameters.exclude_model_list[base_model]

model_names_with_default = ['li'] + model_names

times = np.arange(0, 2000.01, 0.01)
x_cmax = [1, 2, 3, 4]

#''' # manual patch data threshold
tms_u = 0.0579
tms_l = 0.0689
''' # auto/manual patch data threshold
tms_u = 0.0581
tms_l = 0.0671
#'''

exception = {
    'quinidine': [0, 1]
}

qnet_t = {}
qnet_v = {}
# NOTE: Match output naming in compare-torsade-metric-scores.py
print('Loading qNet from cache (compare-torsade-metric-scores.py)')
for model_name in model_names_with_default:
    f = f'torsade-metric-{ap_model_name}-{model_name}.pkl'
    with open(os.path.join(cache, f), 'rb') as f:
        qnet_t[model_name] = pickle.load(f)
    f = f'torsade-metric-validation-{ap_model_name}-{model_name}.pkl'
    with open(os.path.join(cache, f), 'rb') as f:
        qnet_v[model_name] = pickle.load(f)

def tms_color(x):
    if x < tms_u:
        return '#d95f02'
    elif x >= tms_u and x < tms_l:
        return '#7570b3'
    else:
        return '#1b9e77'

def get_cl(c):
    if c == 0:
        l0 = 'Low'
        color0 = '#1b9e77'
    elif c == 1:
        l0 = 'Intermediate'
        color0 = '#7570b3'
    elif c == 2:
        l0 = 'High'
        color0 = '#d95f02'
    return l0, color0

# Plot
fig, axes = plt.subplot_mosaic('AC;AD;AE;BF;BG;BH;BI', figsize=(8.5, 9.5))

xmin, xmax = -0.018, 0.098
for ax, compounds, classes, qnet in [('A', compounds_t, classes_t, qnet_t),
                                     ('B', compounds_v, classes_v, qnet_v)]:
    for i, compound in enumerate(compounds[::-1]):
        if compound in exception:
            x_cmax_ = [x_cmax[k] for k in exception[compound]]
        else:
            x_cmax_ = x_cmax
        for j, model_name in enumerate(model_names):
            q = []
            for x in x_cmax_:
                q.append(qnet[model_name][compound][x])
            tms = np.mean(q)
            color = tms_color(tms)
            alpha = 0.3 if model_name in exclude_model_list[compound] else 0.75
            if 'm0a' in model_name:
                axes[ax].scatter(tms, i, marker='s', color='none', lw=1.5,
                                 ec=color, s=60, alpha=alpha)
            else:
                axes[ax].plot(tms, i, 'x', c=color, alpha=alpha)
        q0 = []
        for x in x_cmax_:
            q0.append(qnet['li'][compound][x])
        tms0 = np.mean(q0)
        #_, color0 = get_cl(classes[compound])
        color0 = tms_color(tms0)
        #axes[ax].plot(tms0, i, 'o', c=color0, alpha=0.75)
        axes[ax].scatter(tms0, i, marker='o', color='none', lw=1.5, ec=color0,
                         s=50, alpha=0.75)
    axes[ax].axvline(tms_u, ls='--', c='#7f7f7f', zorder=0)
    axes[ax].axvline(tms_l, ls='--', c='#7f7f7f', zorder=0)
    axes[ax].set_xlim([xmin, xmax])

    # Legend
    l0, color0 = get_cl(0)
    axes[ax].plot(-np.inf, 1, 'o', c=color0, alpha=0.75, label=l0)
    axes[ax].scatter(-np.inf, 1, marker='s', color='none', lw=1.5,
                     ec='#7f7f7f', alpha=0.75, label='0a')
    l0, color0 = get_cl(1)
    axes[ax].plot(-np.inf, 1, 'o', c=color0, alpha=0.75, label=l0)
    axes[ax].plot(-np.inf, 1, 'x', c='#7f7f7f', alpha=0.75, label='Model')
    l0, color0 = get_cl(2)
    axes[ax].plot(-np.inf, 1, 'o', c=color0, alpha=0.75, label=l0)
    axes[ax].scatter(-np.inf, 1, marker='o', color='none', lw=1.5,
                     ec='#7f7f7f', alpha=0.75, label='CiPA v1.0')

    # Shade the background
    kwargs = dict(alpha=0.085, ec='none', zorder=-1)
    cls = list(classes.values())
    n_high = 1 - cls.count(2) / len(cls)
    n_low = cls.count(0) / len(cls)
    axes[ax].axvspan(xmin, tms_u, ymin=n_high, color='#fc8d62', **kwargs)
    axes[ax].axvspan(tms_u, tms_l, ymin=n_low, ymax=n_high, color='#8da0cb',
                     **kwargs)
    axes[ax].axvspan(tms_l, xmax, ymax=n_low, color='#66c2a5', **kwargs)

ax_t = ['C', 'D', 'E']
ax_v = ['F', 'G', 'H', 'I']

# >>> get drug free/control AP
# NOTE: Match output naming in compare-qnets.py
f = os.path.join(cache, '..', 'qnet', 'aps', f'{ap_model_name}-li')
f += f'-dofetilide-0_cmax'  # NOTE: drug free (doesn't matter which drug)
v_ctrl = myokit.DataLog().load(f)['membrane.V']
# <<<

for axs, compounds, Compounds, qnet in [
    (ax_t, compounds_selected_t, Compounds_selected_t, qnet_t),
    (ax_v, compounds_selected_v, Compounds_selected_v, qnet_v),
    ]:
    for ax, compound, Compound in zip(axs, compounds, Compounds):
        if compound in exception:
            x_cmax_ = [x_cmax[k] for k in exception[compound]]
        else:
            x_cmax_ = x_cmax
        for j, model_name in enumerate(model_names):
            # >>> get AP
            # NOTE: Match output naming in compare-torsade-metric-scores.py
            f = os.path.join(cache, 'aps', f'{ap_model_name}-{model_name}')
            f += f'-{compound}-4_cmax'  # NOTE: 4 x Cmax
            v = myokit.DataLog().load(f)['membrane.V']
            # <<<
            # >>> fancy colouring
            q = []
            for x in x_cmax_:
                q.append(qnet[model_name][compound][x])
            tms = np.mean(q)
            color = tms_color(tms)
            # <<<
            if model_name in exclude_model_list[compound]:
                axes[ax].plot(times, v, c=color, ls=':', alpha=0.15)
            else:
                axes[ax].plot(times, v, c=color, alpha=0.5)
        # >>> get AP
        # NOTE: Match output naming in compare-torsade-metric-scores.py
        f = os.path.join(cache, 'aps', f'{ap_model_name}-li')
        f += f'-{compound}-4_cmax'  # NOTE: 4 x Cmax
        v = myokit.DataLog().load(f)['membrane.V']
        # <<<
        axes[ax].plot(times, v, c='k', alpha=0.75, lw=1.5, ls='--',
                      label='CiPA v1.0')
        axes[ax].plot(times, v_ctrl, c='#7f7f7f', alpha=0.75, lw=1.5, ls=':',
                      label='Drug free')
        axes[ax].set_xlim([0, 650])
        axes[ax].text(0.95, 0.95, Compound, ha='right', va='top',
                      transform=axes[ax].transAxes)

axes['A'].set_yticks(range(len(compounds_t)), compounds_t[::-1])
axes['B'].set_yticks(range(len(compounds_v)), compounds_v[::-1])
for i in ['A', 'C', 'D', 'E', 'F', 'G', 'H']:
    axes[i].set_xticklabels([])
xlabel = r'Torsade metric score (C/F, $1-4$ $\times$ $C_\mathrm{max}$)'
axes['B'].set_xlabel(xlabel, fontsize=11)
axes['A'].set_ylabel('Training compounds', fontsize=11)
axes['B'].set_ylabel('Validation compounds', fontsize=11)
for i in ['C', 'D', 'E', 'F', 'G', 'H', 'I']:
    axes[i].set_ylabel('Voltage (mV)')
axes['I'].set_xlabel('Time (ms)', fontsize=11)

axes['A'].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3,
                 fontsize=8)
axes['C'].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2,
                 fontsize=8)

# Despine and add grid
sns.despine(fig=fig)
axes['A'].tick_params(axis='x', color='none')
axes['A'].spines['bottom'].set_visible(False)
for i in ['A', 'B']:
    axes[i].grid()
sns.set(rc={'axes.facecolor':'none', 'grid.color':'#CACAD2'})

fig.tight_layout()
fig.savefig(os.path.join(results, f'{prefix}'), dpi=300)
fig.savefig(os.path.join(results, f'{prefix}.pdf'), format='pdf')
plt.close(fig)
