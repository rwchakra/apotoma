
import re

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# files_dsa = os.listdir('./mnist')
# files_dsa = [f for f in files_dsa if 'dsa' in f]
# dsa_dons = {}
# dsa_nods = {}
dsa_rans = {}

# for file_dsa in files_dsa:
#
#     if 'rand' not in file_dsa:
#         type_result = file_dsa.split('_')[1]
#         #param = file_dsa.split('_')[2]
#
#     else:
#         type_result = 'rand'
#         #param = re.findall('\d+', file_dsa.split('_')[1])[0]
#     scores = 0
#     times = 0
#     pickle_models = os.listdir('./mnist/'+file_dsa)
#     for pickle_model in pickle_models:
#         with open('./mnist/'+file_dsa+'/'+pickle_model, 'rb') as f:
#             data_dsa = pickle.load(f)
#         scores += data_dsa.evals['adv_fga_0.5'].ood_auc_roc
#         times += data_dsa.evals['adv_fga_0.5'].eval_time
#
#     if type_result == 'don':
#         param = data_dsa.approach_custom_info['num_samples']
#         dsa_dons[param] = (scores/len(pickle_models), times/len(pickle_models))
#
#     elif type_result == 'nod':
#         param = data_dsa.approach_custom_info['num_samples']
#         dsa_nods[param] = (scores/len(pickle_models), times/len(pickle_models))
#
#     else:
#         param = data_dsa.approach_custom_info['sum_samples']
#         dsa_rans[param] = (re.findall('\d+', file_dsa.split('_')[1])[0], scores/len(pickle_models), times/len(pickle_models))
#
# #dons plots
# sorted_dons = sorted(dsa_dons.items(), key=lambda item: float(item[0]), reverse=True)
# sorted_dons_thresholds = [int(item[0]) for item in sorted_dons]
# scores = [item[1][0] for item in sorted_dons]
#
# plt.plot(sorted_dons_thresholds, scores)
# plt.xlabel('#Points sampled')
# plt.ylabel('AUC score')
# plt.title('D-O-N sampling')
# #plt.savefig('./dsa_plots_mnist/difference_of_norms.png')
# #plt.clf()
# #rans plots
# sorted_rans = sorted(dsa_rans.items(), key=lambda item: float(item[0]), reverse=True)
# sorted_rans_thresholds = [float(item[0]) for item in sorted_rans]
# scores_rans = [item[1][1] for item in sorted_rans]
#
# plt.plot(sorted_rans_thresholds, scores_rans)
# plt.xlabel('#points sampled')
# plt.ylabel('AUC score')
# plt.title('Random Sampling')
# plt.savefig('./dsa_plots_mnist/all_plots.png')
# #plt.savefig('./dsa_plots_mnist/random_sampling.png')
# plt.clf()
#
# #Bar plot for random
# root = '/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma/experiments/experiments/mnist/'
# files_dsa = os.listdir(root)
# files_dsa = [f for f in files_dsa if 'dsa' in f and 'rand' in f]
# for f in files_dsa:
#     times = 0
#     pickle_models = os.listdir(root+f)
#     for pickle_model in pickle_models:
#         with open(root+f+'/'+pickle_model, 'rb') as fb:
#             data_dsa = pickle.load(fb)
#         times += data_dsa.evals['adv_fga_0.5'].eval_time
#
#         param = data_dsa.approach_custom_info['sum_samples']
#     dsa_rans[param] = (re.findall('\d+', f.split('_')[1])[0], times/len(pickle_models))
#
# rans = {val[0]:val[1] for item, val in dsa_rans.items()}
# sorted_rans = sorted(rans.items(), key=lambda item: float(item[0]), reverse=True)
# sorted_rans_thresholds = [float(item[0]) for item in sorted_rans]
# scores_rans = [item[1] for item in sorted_rans]
#
# plt.bar(sorted_rans_thresholds, scores_rans, color='maroon')
#
# plt.xlabel("%age Points Sampled")
# plt.ylabel("Time(s)")
# plt.title("Eval Times for random sampling")
# plt.savefig('./dsa_plots_mnist/random_sampling_times_local.png')

#Box plots for lsa and dsa: rand100

# dsa_ran100 = []
# lsa_ran100 = []
#
# lsa_files_ran100 = os.listdir('./mnist/lsa_rand100_perc')
# dsa_files_ran100 = os.listdir('./mnist/dsa_rand100_perc')
#
# for lsa_file in lsa_files_ran100:
#     with open('./mnist/lsa_rand100_perc/'+lsa_file, 'rb') as f:
#         data_lsa = pickle.load(f)
#
#     lsa_ran100.append(data_lsa.evals['adv_fga_0.5'].ood_auc_roc)
#
# for dsa_file in dsa_files_ran100:
#     with open('./mnist/dsa_rand100_perc/' + dsa_file, 'rb') as f:
#         data_dsa = pickle.load(f)
#
#     dsa_ran100.append(data_dsa.evals['adv_fga_0.5'].ood_auc_roc)
# print(dsa_ran100)
# data = [lsa_ran100, dsa_ran100]
# fig = plt.figure(1, figsize=(9, 6))
#
# # Create an axes instance
# ax = fig.add_subplot(111)
#
# bp = ax.boxplot(data, patch_artist=True)
#
# ## change outline color, fill color and linewidth of the boxes
# for box in bp['boxes']:
#     # change outline color
#     box.set( color='#7570b3', linewidth=2)
#     # change fill color
#     box.set( facecolor = '#1b9e77' )
#
# ## change color and linewidth of the whiskers
# for whisker in bp['whiskers']:
#     whisker.set(color='#7570b3', linewidth=2)
#
# ## change color and linewidth of the caps
# for cap in bp['caps']:
#     cap.set(color='#7570b3', linewidth=2)
#
# ## change color and linewidth of the medians
# for median in bp['medians']:
#     median.set(color='#b2df8a', linewidth=2)
#
# ## change the style of fliers and their fill
# for flier in bp['fliers']:
#     flier.set(marker='o', color='#e7298a', alpha=0.5)
#
# ax.set_xticklabels(['LSA', 'DSA'])
# ax.set_title('Stability (LSA vs DSA)')
# ax.set_ylabel('AUC')
# fig.savefig('lsa_vs_dsa_100.png', bbox_inches='tight')

lsa_ran100 = []
root = '/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma/lsa_models/results/mnist/'
lsa_kdes = os.listdir(root)
to_check = ['kde0.1', 'kde0.4', 'kdescott', 'kdesilverman', 'kde1.0', 'kde2.0', 'kde3.0', 'kde5.0']
lsa_kdes = [f for f in lsa_kdes if f.split('_')[2] in to_check]
scott = os.listdir(root+'lsa_rand_kdescott_perc')
silverman = os.listdir(root+'lsa_rand_kdesilverman_perc')
all_data = {}
scott_total = 0
silverman_total = 0
for f in scott:
    with open(root + 'lsa_rand_kdescott_perc/'+f, 'rb') as fb:
        data = pickle.load(fb)
        avg = data.evals['adv_fga_0.5'].avg_bw
        avg = re.findall('\d+.\d+', avg)[0]
        scott_total += float(avg)

for f in silverman:
    with open(root + 'lsa_rand_kdesilverman_perc/'+f, 'rb') as fb:
        data = pickle.load(fb)
        avg = data.evals['adv_fga_0.5'].avg_bw
        avg = re.findall('\d+.\d+', avg)[0]
        silverman_total += float(avg)

scott_avg = scott_total/len(scott)
silverman_avg = silverman_total/len(silverman)
for f in lsa_kdes:
    data_kde = []
    models = os.listdir(root+f)
    for model in models:
        with open(root+f+'/'+model, 'rb') as fb:
            data = pickle.load(fb)
            data_kde.append(data.evals['adv_fga_0.5'].ood_auc_roc)

    if 'scott' in f:
        all_data['scott'+str(scott_avg)] = data_kde
    elif 'silverman' in f:
        all_data['silverman'+str(silverman_avg)] = data_kde

    else:
        param = re.findall('\d+\.\d+', f.split('_')[2])
        all_data[param[0]] = data_kde

print('blafla')
#
# for f in lsa_kdes:
#     data_kde = []
#     lsa_files = os.listdir(root+f)
#     for lsa_file in lsa_files:
#         with open(root + f+'/'+lsa_file, 'rb') as fb:
#             data_dsa = pickle.load(fb)
#
#         data_kde.append(data_dsa.evals['adv_fga_0.5'].ood_auc_roc)
#
#     param = re.findall('\d+\.\d+',f.split('_')[2])
#     if len(param) == 0:
#         if f.split('_')[2] == 'kdescott':
#             all_data[0.53] = data_kde
#
#         else:
#             all_data[0.49] = data_kde
#     else:
#         all_data[param[0]] = data_kde

sorted_rans = sorted(all_data.items(), key=lambda item: float(re.findall('\d+.\d+', item[0])[0]), reverse=True)
#sorted_rans_thresholds = [str(re.findall('\d+.\d+', item[0])[0]) for item in sorted_rans]
sorted_rans_thresholds = []
for item in sorted_rans:
    if 'scott' in item[0]:
        val = np.round(float(re.findall('\d+.\d+', item[0])[0]), 2)
        xtick = str(val) + "\n" + "Scott"
        sorted_rans_thresholds.append(xtick)

    elif 'silverman' in item[0]:
        val = np.round(float(re.findall('\d+.\d+', item[0])[0]), 2)
        xtick = str(val) + "\n" + "Silverman"
        sorted_rans_thresholds.append(xtick)

    else:
        xtick = str(re.findall('\d+.\d+', item[0])[0])
        sorted_rans_thresholds.append(xtick)
scores_rans = [item[1] for item in sorted_rans]

fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

bp = ax.boxplot(scores_rans, patch_artist=True)

## change outline color, fill color and linewidth of the boxes
for box in bp['boxes']:
    # change outline color
    box.set( color='#7570b3', linewidth=2)
    # change fill color
    box.set( facecolor = '#1b9e77' )

## change color and linewidth of the whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)

## change color and linewidth of the caps
for cap in bp['caps']:
    cap.set(color='#7570b3', linewidth=2)

## change color and linewidth of the medians
for median in bp['medians']:
    median.set(color='#b2df8a', linewidth=2)

## change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o', color='#e7298a', alpha=0.5)

ax.set_xticklabels(sorted_rans_thresholds)
ax.set_xlabel('Bandwidth factor')
ax.set_ylabel('AUC-ROC')
#ax.set_title('LSA stability over different kernel bandwidths')
fig.savefig('lsa_thresholds_sorted_proper.png', bbox_inches='tight')
