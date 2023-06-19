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
sns.set(rc={'axes.facecolor':'none', 'grid.color':'none'})
#cmap = sns.color_palette('blend:#7AB,#EDA', as_cmap=True)
#cmap = sns.color_palette('dark:#5B9_r', as_cmap=True)
cmap = sns.color_palette('dark:#66c2a4_r', as_cmap=True)

import methods.parameters as parameters
import methods.heatmap as heatmap

only_bootstrap = '--only_bootstrap' in sys.argv
mark_bootstrap = '--mark_bootstrap' in sys.argv

results = 'figures'
if only_bootstrap:
    prefix = 'figure-4-only_bootstrap'
if mark_bootstrap:
    prefix = 'figure-4-mark_bootstrap'
else:
    prefix = 'figure-4'

results = os.path.join(results, 'figure-4')
if not os.path.isdir(results):
    os.makedirs(results)

mark_asts = {
    'li': ['bepridil', 'cisapride', 'chlorpromazine', 'ibutilide', 'pimozide',
           'clozapine', 'clarithromycin', 'loratadine'],
    'lei': ['bepridil', 'cisapride', 'chlorpromazine', 'ibutilide', 'pimozide',
            'clozapine', 'clarithromycin', 'tamoxifen', 'loratadine'],
}

compounds_1 = list(parameters._drug_training_list)
classes_1 = dict(parameters._drug_training_classes)
compounds_2 = list(parameters._drug_validation_list)
classes_2 = dict(parameters._drug_validation_classes)
shift = len(compounds_1) + 1

model_list = [f'{i}' for i in ['0a', '0b']]
model_list += [f'{i}' for i in range(1, 3)]
model_list += [f'{i}' for i in ['2i']]
model_list += [f'{i}' for i in range(3, 6)]
model_list += [f'{i}' for i in ['5i']]
model_list += [f'{i}' for i in range(6, 14)]

for base_model in ['li', 'lei']:
    if base_model == 'lei':
        model_names = [f'm{m}' for m in model_list]
    else:
        model_names = [f'{base_model}-m{m}' for m in model_list]
        
    if only_bootstrap:
        exclude_model_list = parameters.exclude_model_list_only_bootstrap[base_model]
    elif mark_bootstrap:
        exclude_model_list = parameters.exclude_model_list[base_model]
        exclude_model_list_tmp = parameters.exclude_model_list_only_bootstrap[base_model]
    else:
        exclude_model_list = parameters.exclude_model_list[base_model]

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 9.5))

    # >>> Get matrix
    exclude = np.zeros((len(compounds_1) + len(compounds_2) + 1,
                        len(model_list)))  # +1 for an empty row

    for i, c in enumerate(compounds_1):
        for j, m in enumerate(model_names):
            if mark_bootstrap:
                if m in exclude_model_list_tmp[c]:
                    exclude[i, j] = 0.4
            if m in exclude_model_list[c]:
                exclude[i, j] = 1

    exclude[shift - 1, :] = np.NaN  # empty row

    for i, c in enumerate(compounds_2):
        for j, m in enumerate(model_names):
            if mark_bootstrap:
                if m in exclude_model_list_tmp[c]:
                    exclude[i + shift, j] = 0.4
            if m in exclude_model_list[c]:
                exclude[i + shift, j] = 1
    # <<<

    # >>> Annotate ast
    if not (only_bootstrap or mark_bootstrap):
        mark_ast = mark_asts[base_model]
        compounds_1_ast = [c if c not in mark_ast else r'$^*$ ' + c for c in compounds_1]
        compounds_2_ast = [c if c not in mark_ast else r'$^*$ ' + c for c in compounds_2]
    else:
        compounds_1_ast = compounds_1
        compounds_2_ast = compounds_2

    # <<<

    heatmap.heatmap(exclude, compounds_1_ast + [''] + compounds_2_ast, model_list,
                    cmap=cmap, rotation=0, ha='center', cbarlabel=None,
                    alpha=0.95, ax=ax)
    ax.axhline(shift - 1, c='#7f7f7f')

    # >>> Classes
    colors = ['C3', 'C0', 'C2']
    cl_name = ['High', 'Intermediate', 'Low']
    cl_v = []
    i_i, cl_i = 0, classes_1[compounds_1[0]]
    for i, c in enumerate(compounds_1):
        if classes_1[c] != cl_i:
            cl_v.append((i_i, i - 1))
            i_i, cl_i = i, classes_1[c]
    cl_v.append((i_i, i))

    for i, (v, n) in enumerate(zip(cl_v, cl_name)):
        ax.plot([len(model_list)]*2, v, c=colors[i], ls='-', marker='')
        ax.text(len(model_list) + 0.5, np.mean(v), n, c=colors[i],
                ha='center', va='center', rotation=90)

    cl_v = []
    i_i, cl_i = 0, classes_2[compounds_2[0]]
    for i, c in enumerate(compounds_2):
        if classes_2[c] != cl_i:
            cl_v.append((i_i + shift, i + shift - 1))
            i_i, cl_i = i, classes_2[c]
    cl_v.append((i_i + shift, i + shift))

    for i, (v, n) in enumerate(zip(cl_v, cl_name)):
        ax.plot([len(model_list)]*2, v, c=colors[i], ls='-', marker='')
        ax.text(len(model_list) + 0.5, np.mean(v), n, c=colors[i],
                ha='center', va='center', rotation=90)
    # <<<

    # >>> Labels
    ax.text(-0.285,
            1 - 0.5 * len(compounds_1) / (len(compounds_1 + compounds_2) + 1.),
            'Training compounds',
            ha='center', va='center', rotation=90,
            transform=ax.transAxes)

    ax.text(-0.285,
            0.5 * len(compounds_2) / (len(compounds_1 + compounds_2) + 1.),
            'Validation compounds',
            ha='center', va='center', rotation=90,
            transform=ax.transAxes)

    if base_model == 'li':
        title = 'Binding model (physiological model A)'
    elif base_model == 'lei':
        title = 'Binding model (physiological model B)'
    else:
        raise ValueError(f'Unexpected base model {base_model}')
    ax.set_title(title + '\n')
    # <<<

    # >>> Legend
    ax.scatter(np.NaN, np.NaN, marker='s', color=cmap(-np.inf), alpha=0.95,
               label='Plausible')
    if mark_bootstrap:
        ax.scatter(np.NaN, np.NaN, marker='s', color=cmap(0.4), alpha=0.95,
                   label='As plausible as CiPA v1.0')
    ax.scatter(np.NaN, np.NaN, marker='s', color=cmap(np.inf), alpha=0.95,
               label='Implausible')
    if mark_bootstrap:
        ax.legend(loc='lower left', bbox_to_anchor=(-0.35, 1.035), ncol=2,
                  columnspacing=-5.5)
    else:
        ax.legend(loc='lower left', bbox_to_anchor=(-0.35, 1.045), ncol=2,
                  columnspacing=1.25)
    # <<<

    fig.tight_layout(rect=(0, 0, 1.1, 1))
    fig.savefig(os.path.join(results, f'{prefix}-{base_model}'), dpi=300)
    fig.savefig(os.path.join(results, f'{prefix}-{base_model}.pdf'),
                format='pdf')
    plt.close(fig)
