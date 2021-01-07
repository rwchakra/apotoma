import re

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
auc_rocs = []
files_dsa = os.listdir('./mnist')
files_dsa = [f for f in files_dsa if 'dsa' in f]
dsa_dons = {}
dsa_nods = {}
dsa_rans = {}

for file_dsa in files_dsa:

    if 'rand' not in file_dsa:
        type_result = file_dsa.split('_')[1]
        param = file_dsa.split('_')[2]
    else:
        type_result = 'rand'
        param = re.findall('\d+', file_dsa.split('_')[1])[0]
    scores = 0
    times = 0
    pickle_models = os.listdir('./mnist/'+file_dsa)
    for pickle_model in pickle_models:
        with open('./mnist/'+file_dsa+'/'+pickle_model, 'rb') as f:
            data_dsa = pickle.load(f)
        scores += data_dsa.evals['adv_fga_0.5'].ood_auc_roc
        times += data_dsa.evals['adv_fga_0.5'].eval_time

    if type_result == 'don':
        dsa_dons[param] = (scores/len(pickle_models), times/len(pickle_models))

    elif type_result == 'nod':
        dsa_nods[param] = (scores/len(pickle_models), times/len(pickle_models))

    else:
        dsa_rans[param] = (scores/len(pickle_models), times/len(pickle_models))

#dons plots
sorted_dons = sorted(dsa_dons.items(), key=lambda item: float(item[0]), reverse=True)
sorted_dons_thresholds = [float(item[0]) for item in sorted_dons]
scores = [item[1][0] for item in sorted_dons]

plt.plot(sorted_dons_thresholds, scores)
plt.xlabel('Norm threshold')
plt.ylabel('AUC score')
plt.title('D-O-N sampling')
plt.savefig('./dsa_plots_mnist/difference_of_norms.png')
plt.clf()
#rans plots
sorted_rans = sorted(dsa_rans.items(), key=lambda item: float(item[0]), reverse=True)
sorted_rans_thresholds = [float(item[0]) for item in sorted_rans]
scores_rans = [item[1][0] for item in sorted_rans]

plt.plot(sorted_rans_thresholds, scores_rans)
plt.xlabel('%age points sampled')
plt.ylabel('AUC score')
plt.title('Random Sampling')
plt.savefig('./dsa_plots_mnist/random_sampling.png')
