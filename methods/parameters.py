import os
import numpy as np
from . import DIR_METHOD

_drug_list = ['astemizole', 'azimilide', 'bepridil', 'chlorpromazine',
              'cisapride', 'clarithromycin', 'clozapine', 'diltiazem',
              'disopyramide', 'dofetilide', 'domperidone', 'droperidol',
              'ibutilide', 'loratadine', 'metoprolol', 'mexiletine',
              'nifedipine', 'nitrendipine', 'ondansetron', 'pimozide',
              'quinidine', 'ranolazine', 'risperidone', 'sotalol', 'tamoxifen',
              'terfenadine', 'vandetanib', 'verapamil']

_drug_training_list = [
    # High
    'quinidine', 'bepridil', 'dofetilide', 'sotalol',
    # Intermediate
    'cisapride', 'terfenadine', 'ondansetron', 'chlorpromazine',
    # Low
    'verapamil', 'ranolazine', 'mexiletine', 'diltiazem',
]

_drug_validation_list = [
    # High
    'ibutilide', 'vandetanib', 'azimilide', 'disopyramide',
    # Intermediate
    'domperidone', 'droperidol', 'pimozide', 'astemizole', 'clozapine',
    'clarithromycin', 'risperidone',
    # Low
    'metoprolol', 'tamoxifen', 'loratadine', 'nitrendipine', 'nifedipine',
]

assert(len(_drug_training_list + _drug_validation_list) == len(_drug_list))

_drug_training_classes = {
    # High
    'quinidine': 2,
    'bepridil': 2,
    'dofetilide': 2,
    'sotalol': 2,
    # Intermediate
    'cisapride': 1,
    'terfenadine': 1,
    'ondansetron': 1,
    'chlorpromazine': 1,
    # Low
    'verapamil': 0,
    'ranolazine': 0,
    'mexiletine': 0,
    'diltiazem': 0,
}

assert(len(_drug_training_list) == len(_drug_training_classes))

_drug_validation_classes = {
    # High
    'ibutilide': 2,
    'vandetanib': 2,
    'azimilide': 2,
    'disopyramide': 2,
    # Intermediate
    'domperidone': 1,
    'droperidol': 1,
    'pimozide': 1,
    'astemizole': 1,
    'clozapine': 1,
    'clarithromycin': 1,
    'risperidone': 1,
    # Low
    'metoprolol': 0,
    'tamoxifen': 0,
    'loratadine': 0,
    'nitrendipine': 0,
    'nifedipine': 0,
}

assert(len(_drug_validation_list) == len(_drug_validation_classes))

#
# Model parameters
#

# voltage: mV
# current: pA
# time: ms
# drug concentration: nM

# Compound parameters

# Literature parameters
binding_li = { # Kt, Kmax, Ku, EC50^n, n, Vhalf
    # https://www.ahajournals.org/action/downloadSupplement?doi=10.1161%2FCIRCEP.116.004628&file=circae_circae-2016-004628_supp1.pdf#page=11
    'quinidine': [3.5e-5, 5770, 0.01, 1e6, 0.8311, -64.87],
    'bepridil': [3.5e-5, 37350000, 0.0001765, 1e9, 0.9365, -54.93],
    'dofetilide': [3.5e-5, 1e8, 1.79e-5, 5.483e8, 0.9999, -1.147],
    'sotalol': [3.5e-5, 2403, 0.01985, 9619000, 0.7516, -55],
    'chlorpromazine': [3.5e-5, 206000, 0.03866, 56770000, 0.8871, -14.57],
    'cisapride': [3.5e-5, 9.997, 4.161e-4, 42.06, 0.9728, -199.5],
    'terfenadine': [3.5e-5, 9884, 8.18e-5, 41380, 0.65, -77.49],
    'ondansetron': [3.5e-5, 33540, 0.02325, 9950000, 0.8874, -82.11],
    'diltiazem': [3.5e-5, 251, 0.2816, 1e6, 0.9485, -90.89],
    'mexiletine': [3.5e-5, 9.996, 0.09967, 2308000, 1.304, -86.26],
    'ranolazine': [3.5e-5, 55.84, 0.01929, 147200, 0.95, -94.87],
    'verapamil': [3.5e-5, 46460, 7.927e-4, 9.184e6, 1.043, -100.0],
    # https://ascpt.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Fcpt.1184&file=cpt1184-sup-0002-SupInfoS2.pdf#page=13
    'ibutilide': [3.5e-5, 14.57, 6.078e-05, 38.63, 0.9231, -9.771],
    'vandetanib': [3.5e-5, 36.28, 0.01974, 2223, 0.7126, -48.55],
    'azimilide': [3.5e-5, 6.543e+05, 0.00825, 1.413e+07, 0.6028, -8.821],
    'disopyramide': [3.5e-5, 3.685, 0.1216, 2*4.473e+02, 0.7894, -78.11], # EC50^n = 4.473e4?
    'domperidone': [3.5e-5, 3.339, 0.0003558, 16.09, 0.7026, -65.65],
    'droperidol': [3.5e-5, 14.21, 0.001256, 116.5, 0.578, -78.68],
    'pimozide': [3.5e-5, 10.07, 4.576e-05, 5.601, 0.8714, -158.5],
    'astemizole': [3.5e-5, 2.42, 3.306e-05, 4.883, 1.449, -6.11],
    'clozapine': [3.5e-5, 7.486, 0.02989, 9.048e+04, 1.367, -8.81],
    'clarithromycin': [3.5e-5, 92.89, 0.01216, 2.229e+05, 0.7867, -102.8],
    'risperidone': [3.5e-5, 3.934, 0.001151, 752.8, 1.122, -80.43],
    'metoprolol': [3.5e-5, 3.179e+04, 0.8508, 4.223e+07, 0.811, -89.35],
    'tamoxifen': [3.5e-5, 3.9, 0.01175, 4.067e+05, 2, -2.036],
    'loratadine': [3.5e-5, 3.765e+05, 0.009642, 4.77e+08, 0.8368, -1],
    'nitrendipine': [3.5e-5, 1.713, 0.9819, 5.669e+08, 1.923, -61.58],
    'nifedipine': [3.5e-5, 4.748, 0.9916, 9.752e+08, 1.235, -87.37],
}

# Fitted parameters
Milnes = [os.path.join(DIR_METHOD, '..', 'data', 'Milnes-data-fits', 'Drug'),
          'protocol-Milnes-fit-RMSE-parameters-1.txt']

def binding_mx(m, x):
    o = {}
    if x in [12, 13]:
        for drug in _drug_list:
            o[drug] = np.append(
                3.5e-5,
                np.loadtxt(f'{Milnes[0]}-{drug}-{m}-model{x}-{Milnes[1]}')
            )
    else:
        for drug in _drug_list:
            o[drug] = np.loadtxt(f'{Milnes[0]}-{drug}-{m}-model{x}-{Milnes[1]}')
    return o

Milnes_re = os.path.join(DIR_METHOD, '..', 'data', 'Milnes-data-fits-re')

def binding_mx_re(m):
    o = {}
    for drug in _drug_list:
        f = os.path.join(Milnes_re, f'{drug}-{m}.txt')
        o[drug] = np.loadtxt(f)
    return o

binding = {
    #'m0a': binding_mx_re('m0a'),
    'm0a': binding_mx_re('m0b'),  # NOTE: Use results from m0b steady-state
    'm0b': binding_mx_re('m0b'),
    'm1': binding_mx('lei', 1),
    'm2': binding_mx('lei', 2),
    'm2i': binding_mx_re('m2i'),
    'm3': binding_mx('lei', 3),
    'm4': binding_mx('lei', 4),
    'm5': binding_mx('lei', 5),
    'm5i': binding_mx_re('m5i'),
    'm6': binding_mx('lei', 6),
    'm7': binding_mx('lei', 7),
    'm8': binding_mx('lei', 8),
    'm9': binding_mx('lei', 9),
    'm10': binding_mx('lei', 10),
    'm11': binding_mx('lei', 11),
    'm12': binding_mx('lei', 12),
    'm13': binding_mx('lei', 13),
    'li': binding_li,
    #'li-m0a': binding_mx_re('li-m0a'),
    'li-m0a': binding_mx_re('li-m0b'),  # NOTE: Use m0b steady-state
    'li-m0b': binding_mx_re('li-m0b'),
    'li-m1': binding_mx('fda', 1),
    'li-m2': binding_mx('fda', 2),
    'li-m2i': binding_mx_re('li-m2i'),
    'li-m3': binding_mx('fda', 3),
    'li-m4': binding_mx('fda', 4),
    'li-m5': binding_mx('fda', 5),
    'li-m5i': binding_mx_re('li-m5i'),
    'li-m6': binding_mx('fda', 6),
    'li-m7': binding_mx('fda', 7),
    'li-m8': binding_mx('fda', 8),
    'li-m9': binding_mx('fda', 9),
    'li-m10': binding_mx('fda', 10),
    'li-m11': binding_mx('fda', 11),
    'li-m12': binding_mx('fda', 12),
    'li-m13': binding_mx('fda', 13),
}

# Fixed hill=1
Milnes_fh = os.path.join(DIR_METHOD, '..', 'data', 'Milnes-data-fits-fix_hill')

def binding_mx_fh(m):
    o = {}
    if m in ['m12', 'm13', 'li-m12', 'li-m13']:
        for drug in _drug_list:
            f = os.path.join(Milnes_fh, f'{drug}-{m}.txt')
            o[drug] = np.append(3.5e-5, np.loadtxt(f))
    else:
        for drug in _drug_list:
            f = os.path.join(Milnes_fh, f'{drug}-{m}.txt')
            o[drug] = np.loadtxt(f)
    return o

binding_fix_hill = {
    #'m0a': binding_mx_fh('m0a'),
    'm0a': binding_mx_re('m0b'),  # NOTE: Use m0b steady-state
    'm0b': binding_mx_fh('m0b'),
    'm1': binding_mx_fh('m1'),
    'm2': binding_mx_fh('m2'),
    'm2i': binding_mx_fh('m2i'),
    'm3': binding_mx_fh('m3'),
    'm4': binding_mx_fh('m4'),
    'm5': binding_mx_fh('m5'),
    'm5i': binding_mx_fh('m5i'),
    'm6': binding_mx_fh('m6'),
    'm7': binding_mx_fh('m7'),
    'm8': binding_mx_fh('m8'),
    'm9': binding_mx_fh('m9'),
    'm10': binding_mx_fh('m10'),
    'm11': binding_mx_fh('m11'),
    'm12': binding_mx_fh('m12'),
    'm13': binding_mx_fh('m13'),
    'li': binding_li,  # NOTE: not fixed hill
    #'li-m0a': binding_mx_fh('li-m0a'),
    'li-m0a': binding_mx_fh('li-m0b'),  # NOTE: Use m0b steady-state
    'li-m0b': binding_mx_fh('li-m0b'),
    'li-m1': binding_mx_fh('li-m1'),
    'li-m2': binding_mx_fh('li-m2'),
    'li-m2i': binding_mx_fh('li-m2i'),
    'li-m3': binding_mx_fh('li-m3'),
    'li-m4': binding_mx_fh('li-m4'),
    'li-m5': binding_mx_fh('li-m5'),
    'li-m5i': binding_mx_fh('li-m5i'),
    'li-m6': binding_mx_fh('li-m6'),
    'li-m7': binding_mx_fh('li-m7'),
    'li-m8': binding_mx_fh('li-m8'),
    'li-m9': binding_mx_fh('li-m9'),
    'li-m10': binding_mx_fh('li-m10'),
    'li-m11': binding_mx_fh('li-m11'),
    'li-m12': binding_mx_fh('li-m12'),
    'li-m13': binding_mx_fh('li-m13'),
}

# K = k_off / k_on
K_m1 = lambda x: x[1] / x[0]

K = { # k_off / k_on (= IC_50 in the case of all state blocker)
    'm1': K_m1,
    'm2': K_m1,
    'm3': K_m1,
    'm4': K_m1,
    'm5': K_m1,
    'm6': K_m1,
}

# IC50
ic50_m12 = {
    'dofetilide': 60.,
    'cisapride': 100.,
}

ic50_li = {
    'quinidine': 992.,
    'bepridil': 50.,
    'dofetilide': 4.9,
    'sotalol': 110600.,
    'chlorpromazine': 929.2,
    'cisapride': 10.1,
    'terfenadine': 23.,
    'ondansetron': 1320.,
    'diltiazem': 13150.,
    'mexiletine': 28880.,
    'ranolazine': 8270.,
    'verapamil': 288.,

    'risperidone': 298,
    'metoprolol': 29570,
    'ibutilide': 35.5,
    'domperidone': 23.8,
    'clarithromycin': 14910,
    'astemizole': 33.3,
    'azimilide': 233,
    'disopyramide': 6653,
    'droperidol': 71.319,
    'pimozide': 1.912,
    'vandetanib': 280.9,
    'tamoxifen': 609,
    'loratadine': 4388.5,
    'clozapine': 1820,
    'nifedipine': 141260,
    'nitrendipine': 19744,
}

ic50 = {
    'li': ic50_li,
}

# Cmax
cmax_li = {
    'quinidine': 3237.,
    'bepridil': 33.0,
    'dofetilide': 2.0,
    'sotalol': 14690.,
    'chlorpromazine': 38.0,
    'cisapride': 2.6,
    'terfenadine': 4.0,
    'ondansetron': 139.,
    'diltiazem': 122.,
    'mexiletine': 4129.,
    'ranolazine': 1948.2,
    'verapamil': 81.0,

    'risperidone': 1.81,
    'metoprolol': 1800,
    'ibutilide': 100,
    'domperidone': 19,
    'clarithromycin': 1206,
    'astemizole': 0.26,
    'azimilide': 70,
    'disopyramide': 742,
    'droperidol': 6.33,
    'pimozide': 0.431,
    'vandetanib': 255.4,
    'tamoxifen': 21,
    'loratadine': 0.45,
    'clozapine': 71,
    'nifedipine': 7.7,
    'nitrendipine': 3.02,
}

cmax = {
    'li': cmax_li,
}

# Non-hERG current compound IC50
non_hERG_ic50_li = { # INaL, ICaL, INa, Ito, IK1, IKs
    # https://doi.org/10.3389/fphys.2017.00616
    'quinidine': [9417., 51592.3, 12329., 3487.4, 39589919., 4898.9],
    'bepridil': [1813.9, 2808.1, 2929.3, 8594, None, 28628.3],
    'dofetilide': [753160.4, 260.3, 380.5, 18.8, 394.3, None],
    'sotalol': [None, 7061527, 1.14e9, 43143455, 3050260, 4221856],
    'chlorpromazine': [4559.6, 8191.9, 4535.6, 17616711, 9269.9, None],
    'cisapride': [None, 9258076., None, 219112.4, 29498., 81192862.],
    'terfenadine': [20056, 700.4, 4803.2, 239960.8, None, 399754],
    'ondansetron': [19180.8, 22551.4, 57666.4, 1023378, None, 569807],
    'diltiazem': [21868.5, 112.1, 110859, 2.82e9, None, None],
    'mexiletine': [8956.8, 38243.6, None, None, None, None],
    'ranolazine': [7884.5, None, 68774, None, None, 36155020],
    'verapamil': [7028, 201.8, None, 13429.2, 3.49e8, None],
    ## https://github.com/FDA/CiPA/blob/Model-Validation-2018/AP_simulation/data/newCiPA.csv
    ## https://github.com/FDA/CiPA/blob/ed7c2bc46d7dfa151823ec58ebe4eebf628175c6/AP_simulation/data/newCiPA.csv
    #'risperidone': [None,1465,24079303,None,None,None],
    #'metoprolol': [627433,3260877,30103,None,None,None],
    #'ibutilide': [None,38663,35213,None,None,None],
    #'domperidone': [1162308,74.11,69487,None,None,None],
    #'clarithromycin': [3001951,38012,1089035,None,None,None],
    #'astemizole': [None,557.2,5498,None,None,None],
    #'azimilide': [3536723,12966,608961,None,None,None],
    #'disopyramide': [366539,32717,191760,None,None,None],
    #'droperidol': [None,3320,None,None,None,None],
    #'pimozide': [7473,64.35,10440,None,None,None],
    #'vandetanib': [215377568,6058,175346,None,None,None],
    #'tamoxifen': [None,5704,122208,None,None,None],
    #'loratadine': [None,703,520029,None,None,None],
    #'clozapine': [76920,5465,254597,None,None,None],
    #'nifedipine': [69958,11.49,28441,None,None,None],
    #'nitrendipine': [None,38.37,21939,None,None,None],
    # https://ascpt.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Fcpt.1184&file=cpt1184-sup-0002-SupInfoS2.pdf#page=14
    'risperidone': [1.14e+04*1e3,1.47*1e3,534*1e3,None,None,None],
    'metoprolol': [630*1e3,3.28e+03*1e3,30.3*1e3,None,None,None],
    'ibutilide': [287*1e3,37*1e3,24.1*1e3,None,None,None],
    'domperidone': [225*1e3,0.0736*1e3,41.9*1e3,None,None,None],
    'clarithromycin': [1.81e+03*1e3,38.1*1e3,1.09e+03*1e3,None,None,None],
    'astemizole': [10.3*1e3,0.553*1e3,5.41*1e3,None,None,None],
    'azimilide': [2.94e+03*1e3,13.2*1e3,363*1e3,None,None,None],
    'disopyramide': [377*1e3,32.9*1e3,192*1e3,None,None,None],
    'droperidol': [33.8*1e3,3.23*1e3,36.6*1e3,None,None,None],
    'pimozide': [1.91*1e3,0.0645*1e3,10.2*1e3,None,None,None],
    'vandetanib': [3.79e+03*1e3,6.06*1e3,80.9*1e3,None,None,None],
    'tamoxifen': [3.64e+03*1e3,5.72*1e3,84*1e3,None,None,None],
    'loratadine': [192*1e3,0.703*1e3,113*1e3,None,None,None],
    'clozapine': [73.6*1e3,5.49*1e3,257*1e3,None,None,None],
    'nifedipine': [45.6*1e3,0.0114*1e3,27.6*1e3,None,None,None],
    'nitrendipine': [70.7*1e3,0.0357*1e3,22.4*1e3,None,None,None],
}

non_hERG_ic50 = {
    'li': non_hERG_ic50_li,
}

non_hERG_hill_li = { # INaL, ICaL, INa, Ito, IK1, IKs
    # https://doi.org/10.3389/fphys.2017.00616
    'quinidine': [1.3, 0.6, 1.5, 1.3, 0.4, 1.4],
    'bepridil': [1.4, 0.6, 1.2, 3.5, None, 0.7],
    'dofetilide': [0.3, 1.2, 0.9, 0.8, 0.8, None],
    'sotalol': [None, 0.9, 0.5, 0.7, 1.2, 1.2],
    'chlorpromazine': [0.9, 0.8, 2, 0.4, 0.7, None],
    'cisapride': [None, 0.4, None, 0.2, 0.5, 0.3],
    'terfenadine': [0.6, 0.7, 1, 0.3, None, 0.5],
    'ondansetron': [1, 0.8, 1, 1, None, 0.7],
    'diltiazem': [0.7, 0.7, 0.7, 0.2, None, None],
    'mexiletine': [1.4, 1, None, None, None, None],
    'ranolazine': [0.9, None, 1.4, None, None, 0.5],
    'verapamil': [1, 1.1, None, 0.8, 0.3, None],
    ## https://github.com/FDA/CiPA/blob/Model-Validation-2018/AP_simulation/data/newCiPA.csv
    ## https://github.com/FDA/CiPA/blob/ed7c2bc46d7dfa151823ec58ebe4eebf628175c6/AP_simulation/data/newCiPA.csv
    #'risperidone': [None,0.585,0.371,None,None,None],
    #'metoprolol': [0.66,0.533,0.6,None,None,None],
    #'ibutilide': [None,0.829,1.6,None,None,None],
    #'domperidone': [0.57,0.486,0.88,None,None,None],
    #'clarithromycin': [1.309,0.87,0.8864,None,None,None],
    #'astemizole': [None,1.22,0.735,None,None,None],
    #'azimilide': [0.445,0.701,0.586,None,None,None],
    #'disopyramide': [2.017,0.677,1.293,None,None,None],
    #'droperidol': [None,1.147,None,None,None,None],
    #'pimozide': [0.909,0.449,0.451,None,None,None],
    #'vandetanib': [0.294,0.716,1.048,None,None,None],
    #'tamoxifen': [None,0.757,0.658,None,None,None],
    #'loratadine': [None,0.56,0.536,None,None,None],
    #'clozapine': [1.337,0.924,0.578,None,None,None],
    #'nifedipine': [1.595,0.666,1.113,None,None,None],
    #'nitrendipine': [None,0.4757,0.5766,None,None,None],
    # https://ascpt.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Fcpt.1184&file=cpt1184-sup-0002-SupInfoS2.pdf#page=14
    'risperidone': [5.8,0.59,0.71,None,None,None],
    'metoprolol': [0.66,0.54,0.61,None,None,None],
    'ibutilide': [2.4,0.86,2.3,None,None,None],
    'domperidone': [2.1,0.49,1.5,None,None,None],
    'clarithromycin': [3,0.88,0.89,None,None,None],
    'astemizole': [2.3,1.2,0.76,None,None,None],
    'azimilide': [0.47,0.71,0.72,None,None,None],
    'disopyramide': [2.1,0.69,1.3,None,None,None],
    'droperidol': [2.8,1.2,2.5,None,None,None],
    'pimozide': [1.9,0.45,0.46,None,None,None],
    'vandetanib': [0.82,0.72,1.9,None,None,None],
    'tamoxifen': [4,0.76,0.82,None,None,None],
    'loratadine': [2.1,0.56,1.4,None,None,None],
    'clozapine': [2,0.94,0.59,None,None,None],
    'nifedipine': [4.5,0.67,1.1,None,None,None],
    'nitrendipine': [3.2,0.5,0.58,None,None,None],
}

non_hERG_hill = {
    'li': non_hERG_hill_li,
}


#
# Model variables
#

_dutta_ikr_conductance = {
    # IKr conductance that match to original Dutta el al. 2017 APD90
    #'lei': 0.08795742,  # fitted with match_ap.py n_prepace=10
    #'lei': 0.09017992,  # fitted with match_ap.py n_prepace=100
    'lei': 0.09118699,  # fitted with match_ap.py n_prepace=1000
    'li': 0.046585,  # original model
}

_model_current = {
    'm0a': 'ikr.IKr',
    'm0b': 'ikr.IKr',
    'm1': 'ikr.IKr',
    'm2': 'ikr.IKr',
    'm2i': 'ikr.IKr',
    'm3': 'ikr.IKr',
    'm4': 'ikr.IKr',
    'm5': 'ikr.IKr',
    'm5i': 'ikr.IKr',
    'm6': 'ikr.IKr',
    'm7': 'ikr.IKr',
    'm8': 'ikr.IKr',
    'm9': 'ikr.IKr',
    'm10': 'ikr.IKr',
    'm11': 'ikr.IKr',
    'm12': 'ikr.IKr',  # Li et al. binding model
    'm13': 'ikr.IKr',
    'li': 'ikr.IKr',
    'li-m0a': 'ikr.IKr',
    'li-m0b': 'ikr.IKr',
    'li-m1': 'ikr.IKr',
    'li-m2': 'ikr.IKr',
    'li-m2i': 'ikr.IKr',
    'li-m3': 'ikr.IKr',
    'li-m4': 'ikr.IKr',
    'li-m5': 'ikr.IKr',
    'li-m5i': 'ikr.IKr',
    'li-m6': 'ikr.IKr',
    'li-m7': 'ikr.IKr',
    'li-m8': 'ikr.IKr',
    'li-m9': 'ikr.IKr',
    'li-m10': 'ikr.IKr',
    'li-m11': 'ikr.IKr',
    'li-m12': 'ikr.IKr',  # Same as 'li'
    'li-m13': 'ikr.IKr',
}

_model_dose = {
    'm0a': 'ikr.D',
    'm0b': 'ikr.D',
    'm1': 'ikr.D',
    'm2': 'ikr.D',
    'm2i': 'ikr.D',
    'm3': 'ikr.D',
    'm4': 'ikr.D',
    'm5': 'ikr.D',
    'm5i': 'ikr.D',
    'm6': 'ikr.D',
    'm7': 'ikr.D',
    'm8': 'ikr.D',
    'm9': 'ikr.D',
    'm10': 'ikr.D',
    'm11': 'ikr.D',
    'm12': 'ikr.D',  # Li et al. binding model
    'm13': 'ikr.D',
    'li': 'ikr.D',
    'li-m0a': 'ikr.D',
    'li-m0b': 'ikr.D',
    'li-m1': 'ikr.D',
    'li-m2': 'ikr.D',
    'li-m2i': 'ikr.D',
    'li-m3': 'ikr.D',
    'li-m4': 'ikr.D',
    'li-m5': 'ikr.D',
    'li-m5i': 'ikr.D',
    'li-m6': 'ikr.D',
    'li-m7': 'ikr.D',
    'li-m8': 'ikr.D',
    'li-m9': 'ikr.D',
    'li-m10': 'ikr.D',
    'li-m11': 'ikr.D',
    'li-m12': 'ikr.D',  # Same as 'li'
    'li-m13': 'ikr.D',
}

_model_conductance = {
    'm0a': ['ikr.p9'],
    'm0b': ['ikr.p9'],
    'm1': ['ikr.p9'],
    'm2': ['ikr.p9'],
    'm2i': ['ikr.p9'],
    'm3': ['ikr.p9'],
    'm4': ['ikr.p9'],
    'm5': ['ikr.p9'],
    'm5i': ['ikr.p9'],
    'm6': ['ikr.p9'],
    'm7': ['ikr.p9'],
    'm8': ['ikr.p9'],
    'm9': ['ikr.p9'],
    'm10': ['ikr.p9'],
    'm11': ['ikr.p9'],
    'm12': ['ikr.p9'],  # Li et al. binding model
    'm13': ['ikr.p9'],
    'li': ['ikr.GKr'],
    'li-m0a': ['ikr.g'],
    'li-m0b': ['ikr.g'],
    'li-m1': ['ikr.g'],
    'li-m2': ['ikr.g'],
    'li-m2i': ['ikr.g'],
    'li-m3': ['ikr.g'],
    'li-m4': ['ikr.g'],
    'li-m5': ['ikr.g'],
    'li-m5i': ['ikr.g'],
    'li-m6': ['ikr.g'],
    'li-m7': ['ikr.g'],
    'li-m8': ['ikr.g'],
    'li-m9': ['ikr.g'],
    'li-m10': ['ikr.g'],
    'li-m11': ['ikr.g'],
    'li-m12': ['ikr.g'],  # Same as 'li'
    'li-m13': ['ikr.g'],
}

_model_non_hERG_conductance = {  # INaL, ICaL, INa, Ito, IK1, IKs
    'dutta': ['inal.gNaL', 'ical.base', 'ina.gNa', 'ito.gto', 'ik1.gK1',
              'iks.gKs'],
}

_model_control_steady_state = {
    'dutta': [
        -88.0145,  # v
        6.46961,  # nai
        6.46967,  # nass
        145.501,  # ki
        145.501,  # kss
        7.45E-05,  # cai
        7.30E-05,  # cass
        1.37897,  # cansr
        1.37944,  # cajsr
        0.007335,  # m
        0.698542,  # hf
        0.698542,  # hs
        0.698542,  # j
        0.455526,  # hsp
        0.698541,  # jp
        0.000188,  # mL
        0.513396,  # hL
        0.307923,  # hLp
        0.001,  # a
        0.999555,  # iF
        0.871715,  # iS
        0.00051,  # ap
        0.999555,  # iFp
        0.905466,  # iSp
        2.33E-09,  # d
        1,  # ff
        0.971897,  # fs
        1,  # fcaf
        1,  # fcas
        1,  # jca
        0.001539,  # nca
        1,  # ffp
        1,  # fcafp
        0.130941,  # xs1
        0.000193,  # xs2
        0.996756,  # xk1
        1.47E-07,  # Jrelnp
        1.83E-07,  # Jrelp
        0.003252,  # CaMKt
        0.999637,  # IC1
        6.83E-05,  # IC2
        1.80E-08,  # C1
        8.27E-05,  # C2
        0.000156,  # O
        5.68E-05,  # IO
        0,  # IObound
        0,  # Obound
        0,  # Cbound
        0,  # Drug
    ],
}


kinetics_lei = ['ikr.p1', 'ikr.p2', 'ikr.p3', 'ikr.p4', 'ikr.p5', 'ikr.p6',
                'ikr.p7', 'ikr.p8']

kinetics_li = []

_model_kinetics = {
    'm0a': kinetics_lei,
    'm0b': kinetics_lei,
    'm1': kinetics_lei,
    'm2': kinetics_lei,
    'm2i': kinetics_lei,
    'm3': kinetics_lei,
    'm4': kinetics_lei,
    'm5': kinetics_lei,
    'm5i': kinetics_lei,
    'm6': kinetics_lei,
    'm7': kinetics_lei,
    'm8': kinetics_lei,
    'm9': kinetics_lei,
    'm10': kinetics_lei,
    'm11': kinetics_lei,
    'm12': kinetics_lei,  # Li et al. binding model
    'm13': kinetics_lei,
    'li': [],
    'li-m0a': kinetics_li,
    'li-m0b': kinetics_li,
    'li-m1': kinetics_li,
    'li-m2': kinetics_li,
    'li-m2i': kinetics_li,
    'li-m3': kinetics_li,
    'li-m4': kinetics_li,
    'li-m5': kinetics_li,
    'li-m5i': kinetics_li,
    'li-m6': kinetics_li,
    'li-m7': kinetics_li,
    'li-m8': kinetics_li,
    'li-m9': kinetics_li,
    'li-m10': kinetics_li,
    'li-m11': kinetics_li,
    'li-m12': kinetics_li,
    'li-m13': kinetics_li,
}

_model_binding = {
    'm0a': ['ikr.kon', 'ikr.koff', 'ikr.hill'],
    'm0b': ['ikr.kon', 'ikr.koff', 'ikr.hill'],
    'm1': ['ikr.kon', 'ikr.koff', 'ikr.hill'],
    'm2': ['ikr.kon', 'ikr.koff', 'ikr.hill'],
    'm2i': ['ikr.kon', 'ikr.koff', 'ikr.hill'],
    'm3': ['ikr.kon', 'ikr.koff', 'ikr.hill'],
    'm4': ['ikr.kon', 'ikr.koff', 'ikr.hill'],
    'm5': ['ikr.kon', 'ikr.koff', 'ikr.hill'],
    'm5i': ['ikr.kon', 'ikr.koff', 'ikr.hill'],
    'm6': ['ikr.kon', 'ikr.koff', 'ikr.k2f', 'ikr.hill'],
    'm7': ['ikr.kon_o', 'ikr.koff_o', 'ikr.kon_i', 'ikr.koff_i', 'ikr.hill'],
    'm8': ['ikr.kon_o', 'ikr.koff_o', 'ikr.kon_i', 'ikr.hill'],
    'm9': ['ikr.kon_o', 'ikr.koff_o', 'ikr.kon_i', 'ikr.hill'],
    'm10': ['ikr.kon_o', 'ikr.koff_o', 'ikr.kon_i', 'ikr.k2f', 'ikr.hill'],
    'm11': ['ikr.kon', 'ikr.koff', 'ikr.ku', 'ikr.kt', 'ikr.hill'],
    'm12': ['ikr.kt', 'ikr.kmax', 'ikr.ku', 'ikr.halfmax', 'ikr.hill',
            'ikr.vhalf'],
    'm13': ['ikr.kt', 'ikr.kforward', 'ikr.ku', 'ikr.hill', 'ikr.vhalf'],
    'li': ['ikr.Kt', 'ikr.Kmax', 'ikr.Ku', 'ikr.halfmax', 'ikr.hill',
           'ikr.Vhalf'],
    'li-m0a': ['ikr.kon', 'ikr.koff', 'ikr.hill'],
    'li-m0b': ['ikr.kon', 'ikr.koff', 'ikr.hill'],
    'li-m1': ['ikr.kon', 'ikr.koff', 'ikr.hill'],
    'li-m2': ['ikr.kon', 'ikr.koff', 'ikr.hill'],
    'li-m2i': ['ikr.kon', 'ikr.koff', 'ikr.hill'],
    'li-m3': ['ikr.kon', 'ikr.koff', 'ikr.hill'],
    'li-m4': ['ikr.kon', 'ikr.koff', 'ikr.hill'],
    'li-m5': ['ikr.kon', 'ikr.koff', 'ikr.hill'],
    'li-m5i': ['ikr.kon', 'ikr.koff', 'ikr.hill'],
    'li-m6': ['ikr.kon', 'ikr.koff', 'ikr.k2f', 'ikr.hill'],
    'li-m7': ['ikr.konO', 'ikr.koffO', 'ikr.konI', 'ikr.koffI', 'ikr.hill'],
    'li-m8': ['ikr.konO', 'ikr.koffO', 'ikr.konI', 'ikr.hill'],
    'li-m9': ['ikr.konO', 'ikr.koffO', 'ikr.konI', 'ikr.hill'],
    'li-m10': ['ikr.konO', 'ikr.koffO', 'ikr.konI', 'ikr.k2f', 'ikr.hill'],
    'li-m11': ['ikr.kon', 'ikr.koff', 'ikr.ku', 'ikr.kt', 'ikr.hill'],
    'li-m12': ['ikr.kt', 'ikr.kmax', 'ikr.ku', 'ikr.halfmax', 'ikr.hill',
            'ikr.vhalf'],
    'li-m13': ['ikr.kt', 'ikr.kforward', 'ikr.ku', 'ikr.hill', 'ikr.vhalf'],
}


#
# Milnes' protocol data parameters
#
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

nifedipine_concs = [100000, 300000, 500000]
nitrendipine_concs = [10000, 30000, 100000]

drug_concs = {
    'astemizole': astemizole_concs,
    'azimilide': azimilide_concs,
    'bepridil': bepridil_concs,
    'chlorpromazine': chlorpromazine_concs,
    'cisapride': cisapride_concs,
    'clarithromycin': clarithromycin_concs,
    'clozapine': clozapine_concs,
    'diltiazem': diltiazem_concs,
    'disopyramide': disopyramide_concs,
    'dofetilide': dofetilide_concs,
    'domperidone': domperidone_concs,
    'droperidol': droperidol_concs,
    'ibutilide': ibutilide_concs,
    'loratadine': loratadine_concs,
    'metoprolol': metoprolol_concs,
    'mexiletine': mexiletine_concs,
    'nifedipine': nifedipine_concs,
    'nitrendipine': nitrendipine_concs,
    'ondansetron': ondansetron_concs,
    'pimozide': pimozide_concs,
    'quinidine': quinidine_concs,
    'ranolazine': ranolazine_concs,
    'risperidone': risperidone_concs,
    'sotalol': sotalol_concs,
    'tamoxifen': tamoxifen_concs,
    'terfenadine': terfenadine_concs,
    'vandetanib': vandetanib_concs,
    'verapamil': verapamil_concs
}
assert(len(drug_concs) == len(_drug_list))

''' # Manual selection
# Chosen by better than RMSE of the original FDA model OR better than the 1000
# data bootstrapping RMSE (max. except 'outliers'), using (in `../sandbox`)
# $ python RMSE-Milnes-compare-plot.py --all
# Not chosen by straight cut off but 'similar to' if there is a distinct group.
_exclude_model_list_template = lambda x, y: [f'{x}m{i}' for i in y]

def _exclude_model_list_chlorpromazine(x):
    if x == '':
        return _exclude_model_list_template(x, ['0a', 1, 2, '2i', 3, 4, 5, '5i', 6, 8, 9, 10])
    elif x == 'li-':
        return _exclude_model_list_template(x, ['0a', 1, 2, '2i', 3, 4, 5, '5i', 6, 7, 8, 9, 10, 11])

def _exclude_model_list_loratadine(x):
    if x == '':
        return _exclude_model_list_template(x, ['0a', '0b', 1, 2, '2i', 3, 4, 5, '5i', 6, 8, 9, 10, 11])
    elif x == 'li-':
        return _exclude_model_list_template(x, ['0a', '0b', 1, 2, '2i', 3, 4, 5, '5i', 6, 7, 8, 9, 10, 11])

_exclude_model_list = lambda x: {
    'astemizole': _exclude_model_list_template(x, ['0a']),
    'azimilide': _exclude_model_list_template(x, ['0a', 1, 2, '2i', 3, 4, 5, '5i', 6, 8, 9, 10]),
    'bepridil': _exclude_model_list_template(x, ['0a', '0b', 1, 2, '2i', 3, 4, 5, '5i', 8, 9]),
    'chlorpromazine': _exclude_model_list_chlorpromazine(x),
    'cisapride': _exclude_model_list_template(x, ['0a', '0b', 4, 5, '5i', 9]),
    'clarithromycin': _exclude_model_list_template(x, []),
    'clozapine': _exclude_model_list_template(x, []),
    'diltiazem': _exclude_model_list_template(x, []),
    'disopyramide': _exclude_model_list_template(x, []),
    'dofetilide': _exclude_model_list_template(x, ['0a']),
    'domperidone': _exclude_model_list_template(x, ['0a']),
    'droperidol': _exclude_model_list_template(x, ['0a', '0b', 4, 5, '5i', 9]),
    'ibutilide': _exclude_model_list_template(x, ['0a']), # TODO: Check
    'loratadine': _exclude_model_list_loratadine(x),
    'metoprolol': _exclude_model_list_template(x, []),
    #'mexiletine': _exclude_model_list_template(x,  # TODO: Check
    #                  ['0a', '0b', 1, 2, '2i', 3, 4, 5, '5i', 6, 7, 8, 9, 10, 11, 13]),
    'mexiletine': _exclude_model_list_template(x, [13]),
    'nifedipine': _exclude_model_list_template(x, []),
    'nitrendipine': _exclude_model_list_template(x,
                      ['0a', '0b', 1, 2, '2i', 3, 4, 5, '5i', 6, 7, 8, 9, 10, 11, 13]),
    'ondansetron': _exclude_model_list_template(x, []),
    'pimozide': _exclude_model_list_template(x, []),
    'quinidine': _exclude_model_list_template(x, []),
    'ranolazine': _exclude_model_list_template(x, []),
    'risperidone': _exclude_model_list_template(x, ['0a', '0b', 4, 5, '5i', 9]),
    'sotalol': _exclude_model_list_template(x, []),
    'tamoxifen': _exclude_model_list_template(x,
                      ['0a', '0b', 1, 2, '2i', 3, 4, 5, '5i', 6, 7, 8, 9, 10, 11, 13]),
    'terfenadine': _exclude_model_list_template(x, ['0a', '0b', 1, 2, '2i', 3, 4, 5, '5i', 9]),
    'vandetanib': _exclude_model_list_template(x, []),
    'verapamil': _exclude_model_list_template(x, ['0a', '0b', 4, 5, '5i', 9]),
}

exclude_model_list = {
    'lei': _exclude_model_list(''),
    'li': _exclude_model_list('li-'),
}
'''
# Results from running
# $ python RMSE-Milnes-selection.py
exclude_model_list = {
    'lei': {
        'astemizole': ['m0a'],
        'azimilide': ['m0a', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm8', 'm9', 'm10'],
        'bepridil': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm8', 'm9'],
        'chlorpromazine': [],
        'cisapride': ['m0a', 'm0b', 'm4', 'm5', 'm5i', 'm9', 'm13'],
        'clarithromycin': [],
        'clozapine': [],
        'diltiazem': [],
        'disopyramide': [],
        'dofetilide': ['m0a'],
        'domperidone': [],
        'droperidol': ['m0a', 'm0b', 'm4', 'm5', 'm5i', 'm9'],
        'ibutilide': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm9'],
        'loratadine': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm8', 'm9', 'm10', 'm11'],
        'metoprolol': [],
        'mexiletine': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm13'],
        'nifedipine': [],
        'nitrendipine': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm13'],
        'ondansetron': [],
        'pimozide': [],
        'quinidine': [],
        'ranolazine': [],
        'risperidone': [],
        'sotalol': [],
        'tamoxifen': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm13'],
        'terfenadine': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm9'],
        'vandetanib': [],
        'verapamil': ['m0a', 'm0b', 'm4', 'm5', 'm5i', 'm9'],
    },
    'li': {
        'astemizole': ['li-m0a'],
        'azimilide': ['li-m0a', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m8', 'li-m9', 'li-m10'],
        'bepridil': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m8', 'li-m9'],
        'chlorpromazine': [],
        'cisapride': ['li-m0a', 'li-m0b', 'li-m4', 'li-m5', 'li-m5i', 'li-m9'],
        'clarithromycin': [],
        'clozapine': [],
        'diltiazem': [],
        'disopyramide': [],
        'dofetilide': ['li-m0a'],
        'domperidone': [],
        'droperidol': ['li-m0a', 'li-m0b', 'li-m4', 'li-m5', 'li-m5i', 'li-m9'],
        'ibutilide': ['li-m0a', 'li-m0b', 'li-m4', 'li-m5', 'li-m5i', 'li-m9'],
        'loratadine': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11'],
        'metoprolol': [],
        'mexiletine': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m13'],
        'nifedipine': [],
        'nitrendipine': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m13'],
        'ondansetron': [],
        'pimozide': [],
        'quinidine': [],
        'ranolazine': [],
        'risperidone': [],
        'sotalol': [],
        'tamoxifen': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m13'],
        'terfenadine': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m9'],
        'vandetanib': [],
        'verapamil': ['li-m0a', 'li-m0b', 'li-m4', 'li-m5', 'li-m5i', 'li-m9'],
    },
}
#'''
for k in exclude_model_list:
    assert(len(exclude_model_list[k]) == len(_drug_list))

''' # Manual selection
_all_models = ['0a', '0b', 1, 2, '2i', 3, 4, 5, '5i', 6, 7, 8, 9, 10, 11, 12, 13]

_exclude_model_list_n1_template = lambda x, y: [f'{x}m{i}' for i in _all_models if i not in y]

_exclude_model_list_n1 = lambda x: {
    'astemizole': _exclude_model_list_n1_template(x, [12]),
    'azimilide': _exclude_model_list_n1_template(x, []),
    'bepridil': _exclude_model_list_n1_template(x, [6, 7, 10, 11, 12, 13]),
    'chlorpromazine': _exclude_model_list_n1_template(x, [12]),
    'cisapride': _exclude_model_list_n1_template(x, [12]),
    'clarithromycin': _exclude_model_list_n1_template(x, [12]),
    'clozapine': _exclude_model_list_n1_template(x, [12]),
    'diltiazem': _exclude_model_list_n1_template(x, _all_models),
    'disopyramide': _exclude_model_list_n1_template(x, [12]),
    'dofetilide': _exclude_model_list_n1_template(x, ['0b', 1, 2, '2i', 3, 4, 5, '5i', 6, 7, 8, 9, 10, 11, 12, 13]),
    'domperidone': _exclude_model_list_n1_template(x, [12]),
    'droperidol': _exclude_model_list_n1_template(x, []),
    'ibutilide': _exclude_model_list_n1_template(x, [12]),
    'loratadine': _exclude_model_list_n1_template(x, [11, 12, 13]),
    'metoprolol': _exclude_model_list_n1_template(x, [12]),
    'mexiletine': _exclude_model_list_n1_template(x, ['0a', '0b', 1, 2, '2i', 3, 4, 5, '5i', 6, 7, 8, 9, 10, 11, 12]),
    #'mexiletine': _exclude_model_list_n1_template(x, [12]), # TODO: Check
    'nifedipine': _exclude_model_list_n1_template(x, _all_models),
    'nitrendipine': _exclude_model_list_n1_template(x, [12]),
    'ondansetron': _exclude_model_list_n1_template(x, ['0a', '0b', 1, 2, '2i', 3, 4, 5, '5i', 6, 7, 8, 9, 10, 11, 12]),
    'pimozide': _exclude_model_list_n1_template(x, [12]),
    'quinidine': _exclude_model_list_n1_template(x, _all_models),
    'ranolazine': _exclude_model_list_n1_template(x, _all_models),
    'risperidone': _exclude_model_list_n1_template(x, [12]),
    'sotalol': _exclude_model_list_n1_template(x, [12]),
    'tamoxifen': _exclude_model_list_n1_template(x, []),
    'terfenadine': _exclude_model_list_n1_template(x, [12]),
    'vandetanib': _exclude_model_list_n1_template(x, [12]),
    'verapamil': _exclude_model_list_n1_template(x, [1, 2, '2i', 3, 6, 7, 8, 10, 11, 12, 13]),
}

exclude_model_list_n1 = {
    'lei': _exclude_model_list_n1(''),
    'li': _exclude_model_list_n1('li-'),
}
'''
# Results from running
# $ python RMSE-Milnes-selection-n1.py
exclude_model_list_n1 = {
    'lei': {
        'astemizole': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm13'],
        'azimilide': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13'],
        'bepridil': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm8', 'm9'],
        'chlorpromazine': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm8', 'm9', 'm10'],
        'cisapride': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm13'],
        'clarithromycin': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm13'],
        'clozapine': [],
        'diltiazem': [],
        'disopyramide': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm13'],
        'dofetilide': ['m0a'],
        'domperidone': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm13'],
        'droperidol': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13'],
        'ibutilide': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm13'],
        'loratadine': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10'],
        'metoprolol': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm13'],
        'mexiletine': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm13'],
        'nifedipine': [],
        'nitrendipine': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13'],
        'ondansetron': ['m0a', 'm0b', 'm13'],
        'pimozide': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm13'],
        'quinidine': ['m13'],
        'ranolazine': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm8', 'm9', 'm10', 'm13'],
        'risperidone': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm13'],
        'sotalol': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm13'],
        'tamoxifen': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13'],
        'terfenadine': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13'],
        'vandetanib': ['m0a', 'm0b', 'm1', 'm2', 'm2i', 'm3', 'm4', 'm5', 'm5i', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm13'],
        'verapamil': ['m0a', 'm0b', 'm4', 'm5', 'm5i', 'm9'],
    },
    'li': {
        'astemizole': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m13'],
        'azimilide': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m12', 'li-m13'],
        'bepridil': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m8', 'li-m9'],
        'chlorpromazine': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10'],
        'cisapride': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m13'],
        'clarithromycin': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m13'],
        'clozapine': [],
        'diltiazem': [],
        'disopyramide': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m13'],
        'dofetilide': ['li-m0a'],
        'domperidone': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m13'],
        'droperidol': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m12', 'li-m13'],
        'ibutilide': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m13'],
        'loratadine': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10'],
        'metoprolol': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m13'],
        'mexiletine': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m13'],
        'nifedipine': [],
        'nitrendipine': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m12', 'li-m13'],
        'ondansetron': ['li-m0a', 'li-m0b', 'li-m4', 'li-m5', 'li-m5i', 'li-m9', 'li-m13'],
        'pimozide': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m13'],
        'quinidine': [],
        'ranolazine': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m8', 'li-m9', 'li-m10', 'li-m13'],
        'risperidone': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m13'],
        'sotalol': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m13'],
        'tamoxifen': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m12', 'li-m13'],
        'terfenadine': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m12', 'li-m13'],
        'vandetanib': ['li-m0a', 'li-m0b', 'li-m1', 'li-m2', 'li-m2i', 'li-m3', 'li-m4', 'li-m5', 'li-m5i', 'li-m6', 'li-m7', 'li-m8', 'li-m9', 'li-m10', 'li-m11', 'li-m13'],
        'verapamil': ['li-m0a', 'li-m0b', 'li-m4', 'li-m5', 'li-m5i', 'li-m9'],
    },
}
#'''
for k in exclude_model_list_n1:
    assert(len(exclude_model_list_n1[k]) == len(_drug_list))


#
# FDA classes
#
fda_class = {
    'quinidine': 2,
    'bepridil': 2,
    'dofetilide': 2,
    'sotalol': 2,
    'chlorpromazine': 1,
    'cisapride': 1,
    'terfenadine': 1,
    'ondansetron': 1,
    'diltiazem': 0,
    'flecainide': 0,
    'mexiletine': 0,
    'ranolazine': 0,
    'verapamil': 0,
    'risperidone': 1,
    'metoprolol': 0,
    'ibutilide': 2,
    'domperidone': 1,
    'clarithromycin': 1,
    'astemizole': 1,
    'azimilide': 2,
    'disopyramide': 2,
    'droperidol': 1,
    'pimozide': 1,
    'vandetanib': 2,
    'tamoxifen': 0,
    'loratadine': 0,
    'clozapine': 1,
    'nifedipine': 0,
    'nitrendipine': 0,
}
