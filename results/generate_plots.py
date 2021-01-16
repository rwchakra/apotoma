
import re

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

files = os.listdir('./mnist_new')
files_dsa = [f for f in files if 'lsa' in f]
dsa_lsa = {}
dsa_nods = {}
dsa_rans = {}
param = 0

for file_dsa in files_dsa:

    if 'nod' in file_dsa:
        type_result = file_dsa.split('_')[1]
        #param = file_dsa.split('_')[2]

    elif 'rand' in file_dsa:
        type_result = 'rand'
        #param = re.findall('\d+', file_dsa.split('_')[1])[0]
    else:
        type_result = 'by_lsa'
    scores = 0
    times = 0
    pickle_models = os.listdir('./mnist_new/'+file_dsa)
    for pickle_model in pickle_models:
        with open('./mnist_new/'+file_dsa+'/'+pickle_model, 'rb') as f:
            data_dsa = pickle.load(f)
        scores += data_dsa.evals['adv_fga_0.5'].ood_auc_roc
        times += data_dsa.evals['adv_fga_0.5'].eval_time
        if 'nod' in file_dsa or 'by_lsa' in file_dsa:
            param += data_dsa.approach_custom_info['num_samples']

    if type_result == 'by_lsa':
        #param += data_dsa.approach_custom_info['num_samples']
        dsa_lsa[int(param/len(pickle_models))] = (scores/len(pickle_models), times/len(pickle_models))


    elif type_result == 'nod':
        #param += data_dsa.approach_custom_info['num_samples']
        dsa_nods[int(param/len(pickle_models))] = (scores/len(pickle_models), times/len(pickle_models))


    else:
        param = data_dsa.approach_custom_info['num_samples']
        dsa_rans[param] = (re.findall('\d+', file_dsa.split('_')[1])[0], scores/len(pickle_models), times/len(pickle_models))

    param = 0
# dons plots
sorted_dsa_lsa = sorted(dsa_lsa.items(), key=lambda item: float(item[0]), reverse=True)
sorted_dsa_lsa_thresholds = [float(item[0]) for item in sorted_dsa_lsa]
scores = [item[1][0] for item in sorted_dsa_lsa]

sorted_rans = sorted(dsa_rans.items(), key=lambda item: float(item[0]), reverse=True)
sorted_rans_thresholds = [float(item[0]) for item in sorted_rans]
scores_rans = [item[1][1] for item in sorted_rans]


sorted_nods = sorted(dsa_nods.items(), key=lambda item: float(item[0]), reverse=True)
sorted_nods_thresholds = [float(item[0]) for item in sorted_nods]
scores_nods = [item[1][0] for item in sorted_nods]

plt.plot(sorted_dsa_lsa_thresholds, scores, color='maroon')
plt.plot(sorted_rans_thresholds, scores_rans)
plt.plot(sorted_nods_thresholds, scores_nods, color='green')

plt.xlabel('#Points sampled')
plt.ylabel('AUC score')
plt.title('Three sampling methods [Adversarial]')
plt.legend(['BY_LSA', 'RANDOM', 'N-O-D'])
plt.savefig('./lsa_plots_mnist/lsa_auc_adversarial_all.png')
#
# # #rans plots
# sorted_rans = sorted(dsa_rans.items(), key=lambda item: float(item[0]), reverse=True)
# sorted_rans_thresholds = [float(item[0]) for item in sorted_rans]
# scores_rans = [item[1][1] for item in sorted_rans]
# lsa_rans = {}
# files_lsa = [f for f in files if 'lsa' in f and 'rand' in f]
# for file_lsa in files_lsa:
#     scores = 0
#     times = 0
#     pickle_models = os.listdir('./mnist/' + file_lsa)
#     for pickle_model in pickle_models:
#         with open('./mnist/' + file_lsa + '/' + pickle_model, 'rb') as f:
#             data_lsa = pickle.load(f)
#         scores += data_lsa.evals['adv_fga_0.5'].ood_auc_roc
#         times += data_lsa.evals['adv_fga_0.5'].eval_time
#
#
#     param = data_lsa.approach_custom_info['num_samples']
#     lsa_rans[param] = (
#     re.findall('\d+', file_lsa.split('_')[1])[0], scores / len(pickle_models), times / len(pickle_models))
#
# sorted_rans_lsa = sorted(lsa_rans.items(), key=lambda item: float(item[0]), reverse=True)
# sorted_rans_thresholds_lsa = [float(item[0]) for item in sorted_rans_lsa]
# scores_rans_lsa = [item[1][1] for item in sorted_rans_lsa]
#
# plt.plot(sorted_rans_thresholds, scores_rans)
# plt.plot(sorted_rans_thresholds_lsa, scores_rans_lsa)
# plt.xlabel('#points sampled')
# plt.ylabel('AUC score')
# plt.title('DSA vs LSA random sampling [Corrupt]')
# plt.legend(['DSA', 'LSA'])
# plt.savefig('dsa_lsa_auc_plots_corrupted.png')
# #plt.savefig('./dsa_plots_mnist/random_sampling.png')
# plt.clf()
#
# #Bar plot for random
# rans = {val[0]:val[2] for item, val in dsa_rans.items()}
# sorted_rans = sorted(rans.items(), key=lambda item: float(item[0]), reverse=True)
# sorted_rans_thresholds = [float(item[0]) for item in sorted_rans]
# scores_rans = [item[1] for item in sorted_rans]
#
# plt.bar(sorted_rans_thresholds, scores_rans, color='maroon')
#
# plt.xlabel("%age Points Sampled")
# plt.ylabel("Time(s)")
# plt.title("Eval Times for random sampling")
# plt.savefig('./dsa_plots_mnist/random_sampling_times.png')

#Box plots for lsa and dsa: rand100

dsa_ran100 = []
lsa_ran100 = []

lsa_files_ran100 = os.listdir('./mnist_new/lsa_rand100_perc')
dsa_files_ran100 = os.listdir('./mnist_new/dsa_rand100_perc')

for lsa_file in lsa_files_ran100:
    with open('./mnist_new/lsa_rand100_perc/'+lsa_file, 'rb') as f:
        data_lsa = pickle.load(f)

    lsa_ran100.append(data_lsa.evals['corrupted'].ood_auc_roc)

for dsa_file in dsa_files_ran100:
    with open('./mnist_new/dsa_rand100_perc/' + dsa_file, 'rb') as f:
        data_dsa = pickle.load(f)

    dsa_ran100.append(data_dsa.evals['corrupted'].ood_auc_roc)
print(np.mean(dsa_ran100), np.std(dsa_ran100), np.min(dsa_ran100), np.max(dsa_ran100), np.max(dsa_ran100)-np.min(dsa_ran100))
print(np.mean(lsa_ran100), np.std(lsa_ran100), np.min(lsa_ran100), np.max(lsa_ran100), np.max(lsa_ran100)-np.min(lsa_ran100))
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
# fig.savefig('lsa_vs_dsa.png', bbox_inches='tight')

# dsa_ran20 = []
# dsa_files_ran20 = os.listdir('./mnist/dsa_rand20_perc')
#
# for dsa_file in dsa_files_ran20:
#     with open('./mnist/dsa_rand20_perc/' + dsa_file, 'rb') as f:
#         data_dsa = pickle.load(f)
#
#     dsa_ran20.append(data_dsa.evals['adv_fga_0.5'].ood_auc_roc)
#
# # print(dsa_ran50)
# #
# # print(np.mean(dsa_ran50), np.mean(dsa_ran100))
# # print(min(dsa_ran50), min(dsa_ran100))
# # print(max(dsa_ran50), max(dsa_ran100))
# data = [dsa_ran20]
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
#
# fig.savefig('dsa_ran20.png', bbox_inches='tight')

# #Plot all times [only on random]
#
# lsa_files = os.listdir('./mnist')
# lsa_files = [f for f in lsa_files if 'rand' in f and 'lsa' in f]
#
# to_plot = ['10', '40', '70', '100']
#
# lsa_files = [f for f in lsa_files if re.findall('\d+', f.split('_')[1])[0] in to_plot]
# lsa_random_adv = {}
# lsa_random_corr = {}
# for lsa_file in lsa_files:
#     f = os.listdir('./mnist/'+lsa_file)
#     with open('./mnist/'+lsa_file+'/'+f, 'rb') as fb:
#         data=pickle.load(fb)
#     lsa_random_adv[re.findall('\d+', lsa_file.split('_')[1])[0]] = data.evals['adv_fga_0.5'].eval_time
#     lsa_random_corr[re.findall('\d+', lsa_file.split('_')[1])[0]] = data.evals['corrupted'].eval_time
#
# #rans = {val[0]:val[1] for item, val in dsa_random_adv.items()}
# sorted_rans_adv = sorted(lsa_random_adv.items(), key=lambda item: float(item[0]), reverse=True)
# sorted_rans_thresholds = [float(item[0]) for item in sorted_rans_adv]
# scores_rans_adv = [item[1] for item in sorted_rans_adv]
#
# sorted_rans_corr = sorted(lsa_random_corr.items(), key=lambda item: float(item[0]), reverse=True)
# sorted_rans_thresholds_corr = [float(item[0]) for item in sorted_rans_corr]
# scores_rans_corr = [item[1] for item in sorted_rans_corr]
#
#
# plt.bar(sorted_rans_thresholds, scores_rans_adv, color='maroon')
# plt.bar(np.array(sorted_rans_thresholds)+1, scores_rans_corr, color='blue')
# plt.xlabel('%age points sampled')
# plt.ylabel('Time(s)')
# plt.title('LSA run times for random sampling')
# plt.legend(['Adversarial', 'Corrupted'])
# plt.savefig('./lsa_plots_mnist/lsa_corrupted_adversarial_times.png')
# print('bablabb')

# #Plot all AUC-ROC

# files = [f for f in files if 'rand' in f and 'dsa' in f]
root = '/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma/results/mnist/'
# scores_cifar_dsa = {}
#
# for cifar_dsa in files:
#     cf = os.listdir(root+cifar_dsa)
#     scores = 0
#     for pickle_model in cf:
#         with open(root+cifar_dsa+'/'+pickle_model, 'rb') as fb:
#             data = pickle.load(fb)
#         scores += data.evals['corrupted_sev_5'].ood_auc_roc
#     param = data.approach_custom_info['sum_samples']
#
#     scores_cifar_dsa[param] = scores/len(cf)
#
# sorted_rans_dsa_cifar = sorted(scores_cifar_dsa.items(), key=lambda item: float(item[0]), reverse=True)
# sorted_rans_thresholds_dsa_cifar = [float(item[0]) for item in sorted_rans_dsa_cifar]
# scores_rans_dsa = [item[1] for item in sorted_rans_dsa_cifar]
#
# plt.plot(sorted_rans_thresholds_dsa_cifar, scores_rans_dsa)
# plt.xlabel('#points sampled')
# plt.ylabel('AUC score')
# plt.title('DSA random sampling on CIFAR-10 [corrupted]')
# plt.savefig('./dsa_plots_cifar10/dsa_auc_random_corrupted.png')

# cifar_all = os.listdir('./cifar10/dsa_rand100_perc')
# all_scores = []
# for f in cifar_all:
#     with open(root+'dsa_rand100_perc/'+f, 'rb') as fb:
#         data = pickle.load(fb)
#
#
#     all_scores.append(data.evals['corrupted_sev_5'].ood_auc_roc)
#
# print(all_scores, np.mean(all_scores))
# data = [all_scores]
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
# ax.set_xticklabels(['DSA(all points)'])
# ax.set_ylabel('AUC score')
# ax.set_title('Stability of DSA over CIFAR10[corrupted]')
# fig.savefig('./dsa_plots_cifar10/dsa_cifar10_stability_corrupted.png', bbox_inches='tight')

# for f in files:
#     all_f = os.listdir(root+f)
#     all_scores = []
#     for c in all_f:
#         with open(root+f+'/'+c, 'rb') as fb:
#             data = pickle.load(fb)
#
#         all_scores.append(data.evals['corrupted_sev_5'].ood_auc_roc)
#
#     print(f, max(all_scores), min(all_scores), np.mean(all_scores))

# files = os.listdir('./mnist_new/lsa_rand100_perc')
# scores = 0
# for f in files:
#     with open('./mnist_new/lsa_rand100_perc/'+f, 'rb') as fb:
#         data = pickle.load(fb)
#
#     scores += data.evals['adv_fga_0.5'].ood_auc_roc
#
# print(scores/len(files))