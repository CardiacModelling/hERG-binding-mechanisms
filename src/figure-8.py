#!/usr/bin/env python3
import sys
sys.path.append('..')
import os
import pickle
import numpy as np
import myokit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
import colorcet as cc
import seaborn as sns
sns.set_theme()
sns.set_context('paper')
#sns.set_style('whitegrid')
sns.set_style('ticks')

import methods.models as models
import methods.parameters as parameters
from methods.prepare import _model_voltage, change_component_name

from methods import results
cache = os.path.join(results, 'torsade-metric')

results = 'figures'
prefix = 'figure-8'

results = os.path.join(results, 'figure-8')
if not os.path.isdir(results):
    os.makedirs(results)

compounds_t = list(parameters._drug_training_list)
classes_t = dict(parameters._drug_training_classes)
compounds_v = list(parameters._drug_validation_list)
classes_v = dict(parameters._drug_validation_classes)
compounds_selected_t = ['dofetilide', 'terfenadine', 'diltiazem']
Compounds_selected_t = ['Dofetilide', 'Terfenadine', 'Diltiazem']

ap_model_name = 'dutta'

model_list = [f'{i}' for i in ['0a', '0b']]
model_list += [f'{i}' for i in range(1, 3)]
model_list += [f'{i}' for i in ['2i']]
model_list += [f'{i}' for i in range(3, 6)]
model_list += [f'{i}' for i in ['5i']]
model_list += [f'{i}' for i in range(6, 14)]

base_model = 'lei'
if base_model == 'lei':
    model_names = [f'm{m}' for m in model_list]
else:
    model_names = [f'{base_model}-m{m}' for m in model_list]
exclude_model_list = parameters.exclude_model_list[base_model]

x_cmax = [1, 2, 3, 4]
x_cmax_qnet = [0, 0.5, 1, 5, 10, 15, 20, 25]

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

exception = {
    'quinidine': [0, 1]
}

qnet_s = {}
# NOTE: Match output naming in compare-qnets.py
print('Loading qNet from cache (compare-qnets.py)')
for model_name in model_names:
    f = f'qnet-{ap_model_name}-{model_name}.pkl'
    with open(os.path.join(cache, '..', 'qnet', f), 'rb') as f:
        qnet_s[model_name] = pickle.load(f)

qnet_t = {}
qnet_v = {}
# NOTE: Match output naming in compare-torsade-metric-scores.py
print('Loading qNet from cache (compare-torsade-metric-scores.py)')
for model_name in model_names:
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
#fig, axes = plt.subplot_mosaic('CA;DA;EA;FB;GB;HB;IB', figsize=(9, 11))
fig = plt.figure(figsize=(8.5, 9.5))
gs = gridspec.GridSpec(7, 2)
axes = {
    # LEFT
    'C': fig.add_subplot(gs[0, 0]),
    'D': fig.add_subplot(gs[1, 0]),
    'E': fig.add_subplot(gs[2, 0]),
    # Skip one
    'G': fig.add_subplot(gs[4, 0]),
    'H': fig.add_subplot(gs[5, 0]),
    'I': fig.add_subplot(gs[6, 0]),
    # RIGHT
    'A': fig.add_subplot(gs[:3, 1]),
    'B': fig.add_subplot(gs[3:, 1]),
}

# APs@control
ax_a = ['C', 'D', 'E']
n_prepace = 5 if ('-q' in sys.argv) or ('--quick' in sys.argv) else 1000
print('Initialising AP models...')
m0 = models.APModel('dutta', 'li', cl=2000, n_prepace=n_prepace,
                    parameters=['conductance'])
m1 = models.APModel('dutta', 'm1', cl=2000, n_prepace=n_prepace,
                    parameters=['conductance'])
#dt = 0.01
dt = 0.5
t = np.arange(0, 2000+dt, dt)
g0 = parameters._dutta_ikr_conductance['li']
g1 = parameters._dutta_ikr_conductance['lei']
# NOTE: just so the model works; we are still using the control model
compound = 'dofetilide'
non_hERG_ic50s = parameters.non_hERG_ic50['li'][compound]
non_hERG_hs = parameters.non_hERG_hill['li'][compound]
m0.set_non_hERG_parameters(non_hERG_ic50s, non_hERG_hs)
m1.set_non_hERG_parameters(non_hERG_ic50s, non_hERG_hs)
m0.set_dose(0)
m1.set_dose(0)
# Variables and simulate
#Red: '#fb9a99', '#e78ac3', '#e31a1c'
#Blue: '#a6cee3', '#8da0cb', '#1f78b4'
#Orange: '#fdbf6f', '#fc8d62', '#ff7f00'
o0 = ['ikr.IKr', 'ikr.IC1', 'ikr.IC2', 'ikr.O', 'ikr.IO', 'ikr.C1', 'ikr.C2']
c0 = ['#fb9a99', '#e78ac3', '#8da0cb', '#fdbf6f', '#fc8d62', '#ff7f00']
#l0 = ['IC1', 'IC2', 'O', 'I', 'C1', 'C2']  # original model naming
l0 = ['C1', 'C2', 'O', 'I', 'IC1', 'IC2']
o1 = ['ikr.IKr', 'ikr.C', 'ikr.O', 'ikr.I', 'ikr.CI']
c1 = ['#fb9a99', '#8da0cb', '#fdbf6f', '#fc8d62']
l1 = ['C', 'O', 'I', 'IC']
for i in range(len(o0)): o0[i] = change_component_name(o0[i])
for i in range(len(o1)): o1[i] = change_component_name(o1[i])
print(f'Simulating APs with {n_prepace} prepacing...')
d0 = m0.simulate([g0], t, extra_log=o0)
d1 = m1.simulate([g1], t, extra_log=o1)
print('Plotting...')
# 1.
ax = axes[ax_a[0]]
ax2 = ax.twinx()
ax.plot(t, d0[_model_voltage[ap_model_name]], c='#7570b3', ls='--',
        label='hERG physiological model A')
ax.plot(t, d1[_model_voltage[ap_model_name]], c='#7570b3', ls='-',
        label='hERG physiological model B')
ax.set_ylabel('Voltage (mV)', fontsize=11)
ax2.plot(t, d0[o0[0]], c='#d95f02', ls='--')
ax2.plot(t, d1[o1[0]], c='#d95f02', ls='-')
ax2.set_ylabel(r'$I_\mathrm{Kr}$ (A/F)', color='#d95f02', fontsize=11)
ax2.tick_params(axis='y', color='#d95f02', labelcolor='#d95f02')
for ax in ax_a:
    axes[ax].set_xlim([0, 500])
if '-v' in sys.argv or '--verbose' in sys.argv:
    q0 = np.trapz(d0[o0[0]], x=t) * 1e-3  # pA/pF*ms -> pA/pF*s
    q1 = np.trapz(d1[o1[0]], x=t) * 1e-3  # pA/pF*ms -> pA/pF*s
    print('q(IKr_A) =', q0)
    print('q(IKr_B) =', q1)
# 2.
ax = axes[ax_a[1]]
ax.set_ylabel('State\noccupancy', fontsize=11)
ax.stackplot(t, [d0[i] for i in o0[1:]], colors=c0, alpha=0.5, labels=l0,
             linestyle='--', ec='#7f7f7f')
ax.text(0.05, 1.05, 'hERG physiological model A', ha='left', va='bottom',
        transform=ax.transAxes)
ax.set_ylim([0, 1])
# 3.
ax = axes[ax_a[2]]
ax.set_ylabel('State\noccupancy', fontsize=11)
ax.stackplot(t, [d1[i] for i in o1[1:]], colors=c1, alpha=0.5, labels=l1,
             ec='#7f7f7f')
ax.text(0.05, 1.05, 'hERG physiological model B', ha='left', va='bottom',
        transform=ax.transAxes)
ax.set_ylim([0, 1])

# qNet
ax_b = ['G', 'H', 'I']
#colours = sns.color_palette('Set3', n_colors=len(model_list))
#colours = sns.husl_palette(n_colors=len(model_list)+2, h=.5, s=.6)
colours = sns.color_palette(cc.glasbey_category10, n_colors=len(model_list))
#colours = sns.color_palette(cc.glasbey_hv, n_colors=len(model_list))
for ax, compound, Compound in zip(ax_b,
                                  compounds_selected_t,
                                  Compounds_selected_t):
    ax = axes[ax]
    ax.text(0.05, 0.1, Compound, ha='left', va='bottom', #fontsize=10,
            transform=ax.transAxes)

    if compound in exception:
        x_cmax_ = [x_cmax_qnet[k] for k in exception[compound]]
    else:
        x_cmax_ = list(x_cmax_qnet)

    for i_m, model_name in enumerate(model_names):
        q = []
        for x in x_cmax_:
            q.append(qnet_s[model_name][compound][x])
        if model_name in exclude_model_list[compound]:
            ax.plot(x_cmax_, q, 'o:', alpha=0.2, c=colours[i_m],
                    label=model_list[i_m])
        else:
            ax.plot(x_cmax_, q, 'o-', alpha=0.5, c=colours[i_m],
                    label=model_list[i_m])
# Shade the background according to the tms decision boundaries
ymin, ymax = np.inf, -np.inf
for ax in ax_b:
    ax = axes[ax]
    ymin_i, ymax_i = ax.get_ylim()
    ymin = min(ymin, ymin_i)
    ymax = max(ymax, ymax_i)
    yrange = ymax - ymin
ymin -= .15 * yrange
ymax += .15 * yrange
for ax in ax_b:
    ax = axes[ax]
    ax.set_ylim(ymin, ymax)
    kwargs = dict(ec='none', zorder=-1)
    ax.axhspan(ymin, tms_u, alpha=0.085, color='#fc8d62', **kwargs)
    ax.axhspan(tms_u, tms_l, alpha=0.085, color='#8da0cb', **kwargs)
    ax.axhspan(tms_l, ymax, alpha=0.085, color='#66c2a5', **kwargs)
    kwargs = dict(ls='--', lw=1.5, alpha=0.5, zorder=0)
    ax.axhline(tms_u, color='#7f7f7f', **kwargs)
    ax.axhline(tms_l, color='#7f7f7f', **kwargs)

# Torsade metric
xmin, xmax = -0.025, 0.09
for ax, compounds, classes, qnet in [('A', compounds_t, classes_t, qnet_t),
                                     ('B', compounds_v, classes_v, qnet_v)]:
    for i, compound in enumerate(compounds[::-1]):
        if compound in exception:
            x_cmax_ = [x_cmax[k] for k in exception[compound]]
        else:
            x_cmax_ = x_cmax
        c = classes[compound]
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
    axes[ax].axvline(tms_u, ls='--', c='#7f7f7f', zorder=0)
    axes[ax].axvline(tms_l, ls='--', c='#7f7f7f', zorder=0)
    axes[ax].set_xlim([xmin, xmax])

    # Legend
    l0, color0 = get_cl(0)
    axes[ax].plot(-np.inf, 1, 'o', c=color0, alpha=0.75, label=l0)
    #axes[ax].plot(-np.inf, 1, 's', c='#7f7f7f', alpha=0.75, label='0a')
    axes[ax].scatter(-np.inf, 1, marker='s', color='none', lw=1.5,
                     ec='#7f7f7f', alpha=0.75, label='0a')
    l0, color0 = get_cl(1)
    axes[ax].plot(-np.inf, 1, 'o', c=color0, alpha=0.75, label=l0)
    axes[ax].plot(-np.inf, 1, 'x', c='#7f7f7f', alpha=0.75, label='Model')
    l0, color0 = get_cl(2)
    axes[ax].plot(-np.inf, 1, 'o', c=color0, alpha=0.75, label=l0)

    # Shade the background
    kwargs = dict(alpha=0.085, ec='none', zorder=-1)
    cls = list(classes.values())
    n_high = 1 - cls.count(2) / len(cls)
    n_low = cls.count(0) / len(cls)
    axes[ax].axvspan(xmin, tms_u, ymin=n_high, color='#fc8d62', **kwargs)
    axes[ax].axvspan(tms_u, tms_l, ymin=n_low, ymax=n_high, color='#8da0cb',
                     **kwargs)
    axes[ax].axvspan(tms_l, xmax, ymax=n_low, color='#66c2a5', **kwargs)

axes['A'].set_yticks(range(len(compounds_t)), compounds_t[::-1])
axes['B'].set_yticks(range(len(compounds_v)), compounds_v[::-1])
for i in ['A', 'C', 'D', 'G', 'H']:
    axes[i].set_xticklabels([])
xlabel = r'Torsade metric score (C/F, $1-4$ $\times$ $C_\mathrm{max}$)'
axes['B'].set_xlabel(xlabel, fontsize=11)
axes['A'].set_ylabel('Training compounds', fontsize=11)
axes['B'].set_ylabel('Validation compounds', fontsize=11)
for i in ['G', 'H', 'I']:
    axes[i].set_ylabel(r'$q_\mathrm{net}$ (C/F)', fontsize=11)
axes['I'].set_xlabel(r'$\times$ $C_\mathrm{max}$', fontsize=11)
axes['E'].set_xlabel('Time (ms)', fontsize=11)
fig.align_ylabels([axes[ax] for ax in ['C', 'D', 'E', 'G', 'H', 'I']])

# Legends
axes['C'].legend(loc='lower left', bbox_to_anchor=(-0.015, 1))
axes['D'].legend(loc='lower right', bbox_to_anchor=(1, -0.03), fontsize=7)
axes['E'].legend(loc='lower right', bbox_to_anchor=(1, -0.03), fontsize=7)
axes['G'].legend(loc='lower left', bbox_to_anchor=(-0.015, 1.13), ncol=5,
                 columnspacing=1.25)
axes['A'].legend(loc='lower right', bbox_to_anchor=(1.015, 1), ncol=3)

# Labelling subplots
labels = [('A', 'C'), ('B', 'G'), ('C', 'A')]  # Match weird labellings
for label, ax in labels:
    ax = axes[ax]
    trans = mtransforms.ScaledTranslation(-25/72, 20/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=11, va='bottom', fontfamily='serif')

# Despine and add grid
sns.despine(fig=fig)
#sns.despine(fig=fig, offset=10, trim=True)
axes['A'].tick_params(axis='x', color='none')
#axes['A'].set_xticks([])
axes['A'].spines['bottom'].set_visible(False)
ax2.axvline(500.5, lw=3.5, c='#d95f02')
for i in ['A', 'B']:
    axes[i].grid()
sns.set(rc={'axes.facecolor':'none', 'grid.color':'#CACAD2'})

fig.tight_layout(h_pad=-3.5, w_pad=-1)
fig.savefig(os.path.join(results, f'{prefix}'), dpi=300)
fig.savefig(os.path.join(results, f'{prefix}.pdf'), format='pdf')
plt.close(fig)
