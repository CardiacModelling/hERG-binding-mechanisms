#!/usr/bin/env python3
import sys
sys.path.append('..')
import os
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler
from mord import LogisticAT

import methods.parameters as parameters

from methods import results
cache = os.path.join(results, 'torsade-metric')

np.random.seed(1)

results = os.path.join(results, 'ordinal-logistic')
if not os.path.isdir(results):
    os.makedirs(results)

VERBOSE = '-v' in sys.argv or '--verbose' in sys.argv

compounds_t = list(parameters._drug_training_list)
classes_t = dict(parameters._drug_training_classes)
compounds_v = list(parameters._drug_validation_list)
classes_v = dict(parameters._drug_validation_classes)

ap_model_name = 'dutta'

model_list = [f'{i}' for i in range(1, 3)]
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

if base_model == 'li':
    model_names = ['li'] + model_names

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
for model_name in model_names:
    f = f'torsade-metric-{ap_model_name}-{model_name}.pkl'
    with open(os.path.join(cache, f), 'rb') as f:
        qnet_t[model_name] = pickle.load(f)
    f = f'torsade-metric-validation-{ap_model_name}-{model_name}.pkl'
    with open(os.path.join(cache, f), 'rb') as f:
        qnet_v[model_name] = pickle.load(f)

# Get data
X = []
y = []
X_excluded = []
y_excluded = []
for compounds, classes, qnet in [(compounds_t, classes_t, qnet_t),]:
    #                             (compounds_v, classes_v, qnet_v)]:
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

            if model_name in exclude_model_list[compound]:
                X_excluded.append(tms)
                y_excluded.append(c)
            else:
                X.append(tms)
                y.append(c)
X = np.array(X).reshape(-1, 1)
y = np.array(y)
X_excluded = np.array(X_excluded).reshape(-1, 1)
y_excluded = np.array(y_excluded)

#print(X.shape, y.shape)
#print(X_excluded.shape, y_excluded.shape)

# Feature scaling on the input data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Multinomial ordinal logistic regression model
alpha = 1
clf = LogisticAT(alpha=alpha, verbose=0)
clf.fit(X, y)

# Model parameters/coefficients
# NOTE: Checking the model's convention: mord/threshold_based.py:L158
#       beta in Li et al. 2019 supplementary #2 is negative of coef_
beta = -clf.coef_[0]
l2, l1 = clf.theta_  # This is the only way rough to give sensible numbers

# Score of the model
score = clf.score(X, y)
print('Mean absolute error:', -score)
print('========')

def sigmoid(x, l, b):
    return 1. / (1. + np.exp(-(l + b * x)))

if VERBOSE:
    print('True | Predict | P(model) | P(risk < low) | P(risk < intermediate)')
    print('--------')
    y_hat = clf.predict(X)
    p_y = clf.predict_proba(X)
    for yi, yi_hat, pi_y, x in zip(y, y_hat, p_y, X.reshape(-1)):
        print(yi, '|', yi_hat, '|', pi_y, '|', sigmoid(x, l1, beta), '|',
              sigmoid(x, l2, beta))
    print('========')

print('Coefficient (beta):', beta)
print('Intercepts (l1 & l2):', l1, '&', l2)
print('========')

# Follows https://ascpt.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Fcpt.1184&file=cpt1184-sup-0002-SupInfoS2.pdf#page=20
thres_1 = np.log(np.exp(-l2) - 2 * np.exp(-l1)) / beta
thres_2 = np.log(np.exp(-l1) * np.exp(-l2) / (-(2. * np.exp(-l1) - np.exp(-l2)))) / beta

tms_thres_1 = scaler.inverse_transform([[thres_1]])[0][0]
tms_thres_2 = scaler.inverse_transform([[thres_2]])[0][0]

print('Threshold 1:', tms_thres_1)  # high/intermediate | low
print('Threshold 2:', tms_thres_2)  # high | intermediate/low

np.savetxt(os.path.join(results, f'thresholds-{base_model}.txt'),
           [tms_thres_1, tms_thres_2])

