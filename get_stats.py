import pandas as pd
from scipy.stats import wilcoxon, normaltest, shapiro
import numpy as np
import researchpy as rp

approaches = ['oe', 'sm', 'diss', 'dsa', 'lsa']
# # for ap in approaches:
# #     data = pd.read_csv('mnist_'+ap+'_auc.csv')
# #     data.columns = ['fmnist', 'mnist-c', 'mnist-fgsm']
# #     data.to_csv('mnist_'+ap+'_auc.csv')
#
# data_oe = pd.read_csv('mnist_oe_auc.csv')
# data_oe_fmnist = data_oe['fmnist']
# data_oe_mnistc = data_oe['mnist-c']
# data_oe_mnistfgsm = data_oe['mnist-fgsm']
#
# for ap in approaches[1:]:
#     data_ap = pd.read_csv('mnist_'+ap+'_auc.csv')
#     data_ap_fmnist = data_ap['fmnist']
#     data_ap_mnistc = data_ap['mnist-c']
#     data_ap_mnistfgsm = data_ap['mnist-fgsm']
#
#     print(ap+" FMNIST : ", wilcoxon(data_oe_fmnist, data_ap_fmnist))
#     print(ap + " MNIST-C : ", wilcoxon(data_oe_mnistc, data_ap_mnistc))
#     if ap == 'sm':
#         continue
#     else:
#         print(ap + " MNIST-FGSM: ", wilcoxon(data_oe_mnistfgsm, data_ap_mnistfgsm))
# # for ap in approaches:
# #     data = pd.read_csv('cifar_'+ap+'_auc.csv')
# #     data.columns = ['cifar100', 'cifar10-c', 'cifar10-fgsm']
# #     data.to_csv('cifar_'+ap+'_auc.csv')
#
# data_oe = pd.read_csv('cifar_oe_auc.csv')
# data_oe_cifar100 = data_oe['cifar100']
# data_oe_cifarc = data_oe['cifar10-c']
# data_oe_cifarfgsm = data_oe['cifar10-fgsm']
#
# for ap in approaches[1:]:
#     data_ap = pd.read_csv('cifar_'+ap+'_auc.csv')
#     data_ap_cifar100 = data_ap['cifar100']
#     if ap == 'lsa' or ap == 'dsa':
#         data_ap_cifarc = data_ap['cifar10-c']/5
#     else:
#         data_ap_cifarc = data_ap['cifar10-c']
#     data_ap_cifarfgsm = data_ap['cifar10-fgsm']
#
#     print(ap+" CIFAR100 : ", wilcoxon(data_oe_cifar100, data_ap_cifar100))
#     print(ap + " CIFAR10-C : ", wilcoxon(data_oe_cifarc, data_ap_cifarc))
#     print(ap + " CIFAR10-FGSM: ", wilcoxon(data_oe_cifarfgsm, data_ap_cifarfgsm))
#
#
# #model_times_cifar.npy, model_times_mnist.npy, dsa_cifar_times.npy, dsa_mnist_times.npy, lsa_cifar_times.npy, lsa_mnist_times.npy
#
# # times_cifar = np.load('model_times_cifar.npy', allow_pickle=True)
# # #print([(a + b) / 2 for a, b in zip(times_cifar.item()['oe'][::2], times_cifar.item()['oe'][1::2])])
# # oe_times_cifar =  [(a + b) / 2 for a, b in zip(times_cifar.item()['oe'][::2], times_cifar.item()['oe'][1::2])]
# #
# # for ap in approaches[1:]:
# #     if ap == 'lsa':
# #         times_lsa = np.load('lsa_cifar_times.npy', allow_pickle=True)
# #         ap_times = [(a + b) / 2 for a, b in zip(times_lsa[::2], times_lsa[1::2])]
# #         print(ap + " CIFAR10", wilcoxon(oe_times_cifar, ap_times))
# #
# #     elif ap == 'dsa':
# #         times_dsa = np.load('dsa_cifar_times.npy', allow_pickle=True)
# #         ap_times = [(a + b) / 2 for a, b in zip(times_dsa[::2], times_dsa[1::2])]
# #         print(ap + " CIFAR10", wilcoxon(oe_times_cifar, ap_times))
# #
# #     else:
# #         ap_times = [(a + b) / 2 for a, b in zip(times_cifar.item()[ap][::2], times_cifar.item()[ap][1::2])]
# #         print(ap + " CIFAR10", wilcoxon(oe_times_cifar, ap_times))
# #
# times_mnist = np.load('model_times_mnist.npy', allow_pickle=True)
# #print([(a + b) / 2 for a, b in zip(times_cifar.item()['oe'][::2], times_cifar.item()['oe'][1::2])])
# oe_times_mnist =  [(a + b) / 3 for a, b in zip(times_mnist.item()['oe'][::3], times_mnist.item()['oe'][1::3])]
#
# for ap in approaches[1:]:
#     if ap == 'lsa':
#         times_lsa = np.load('lsa_mnist_times.npy', allow_pickle=True)
#         ap_times = [(a + b) / 3 for a, b in zip(times_lsa[::3], times_lsa[1::3])]
#         print(ap + " MNIST", wilcoxon(oe_times_mnist, ap_times))
#
#     elif ap == 'dsa':
#         times_dsa = np.load('dsa_mnist_times.npy', allow_pickle=True)
#         ap_times = [(a + b) / 3 for a, b in zip(times_dsa[::3], times_dsa[1::3])]
#         print(ap + " MNIST", wilcoxon(oe_times_mnist, ap_times))
#
#     else:
#         ap_times = [(a + b) / 3 for a, b in zip(times_mnist.item()[ap][::3], times_mnist.item()[ap][1::3])]
#         print(ap + " MNIST", wilcoxon(oe_times_mnist, ap_times))
#
#
# data_oe = pd.read_csv('mnist_oe_auc.csv')
# data_oe_fmnist = data_oe['fmnist']
# data_oe_mnistc = data_oe['mnist-c']
# data_oe_mnistfgsm = data_oe['mnist-fgsm']
#
# for ap in approaches[1:]:
#     data_ap = pd.read_csv('mnist_'+ap+'_auc.csv')
#     data_ap_fmnist = data_ap['fmnist']
#     data_ap_mnistc = data_ap['mnist-c']
#     data_ap_mnistfgsm = data_ap['mnist-fgsm']
#
#     print(ap+" FMNIST : ", wilcoxon(data_oe_fmnist, data_ap_fmnist))
#     print(ap + " MNIST-C : ", wilcoxon(data_oe_mnistc, data_ap_mnistc))
#     if ap == 'sm':
#         continue
#     else:
#         print(ap + " MNIST-FGSM: ", wilcoxon(data_oe_mnistfgsm, data_ap_mnistfgsm))
# # for ap in approaches:
# #     data = pd.read_csv('cifar_'+ap+'_auc.csv')
# #     data.columns = ['cifar100', 'cifar10-c', 'cifar10-fgsm']
# #     data.to_csv('cifar_'+ap+'_auc.csv')

# CIFAR10 COHEN's d

# data_oe = pd.read_csv('cifar_oe_auc.csv')
# data_oe_cifar100 = data_oe['cifar100']
# data_oe_cifarc = data_oe['cifar10-c']
# data_oe_cifarfgsm = data_oe['cifar10-fgsm']
#
# for ap in approaches[1:]:
#     data_ap = pd.read_csv('cifar_'+ap+'_auc.csv')
#     data_ap_cifar100 = data_ap['cifar100']
#     if ap == 'lsa' or ap == 'dsa':
#      data_ap_cifarc = data_ap['cifar10-c']/5
#     else:
#         data_ap_cifarc = data_ap['cifar10-c']
#     data_ap_cifarfgsm = data_ap['cifar10-fgsm']
#
#     print(rp.ttest(data_oe_cifar100, data_ap_cifar100, group1_name='cifar100_oe', group2_name='cifar100_'+ap, equal_variances=False))
#     print(rp.ttest(data_oe_cifarc, data_ap_cifarc, group1_name='cifarc_oe', group2_name='cifarc_'+ap, equal_variances=False))
#     print(rp.ttest(data_oe_cifarfgsm, data_ap_cifarfgsm, group1_name='cifarfgsm_oe', group2_name='cifarfgsm_'+ap, equal_variances=False))

#MNIST COHEN'S d

# data_oe = pd.read_csv('mnist_oe_auc.csv')
# data_oe_fmnist = data_oe['fmnist']
# data_oe_mnistc = data_oe['mnist-c']
# data_oe_mnistfgsm = data_oe['mnist-fgsm']
#
# for ap in approaches[1:]:
#     data_ap = pd.read_csv('mnist_'+ap+'_auc.csv')
#     data_ap_fmnist = data_ap['fmnist']
#     data_ap_mnistc = data_ap['mnist-c']
#     data_ap_mnistfgsm = data_ap['mnist-fgsm']
#
#     print(rp.ttest(data_oe_fmnist, data_ap_fmnist, group1_name='fmnist_oe', group2_name='fmnist_'+ap, equal_variances=False))
#     print(rp.ttest(data_oe_mnistc, data_ap_mnistc, group1_name='mnistc', group2_name='mnistc_'+ap, equal_variances=False))
#     print(rp.ttest(data_oe_mnistfgsm, data_ap_mnistfgsm, group1_name='mnistfgsm_oe', group2_name='mnistfgsm_'+ap, equal_variances=False))

#MNIST TIMES COHEN'S d

# times_mnist = np.load('model_times_mnist.npy', allow_pickle=True)
# #print([(a + b) / 2 for a, b in zip(times_cifar.item()['oe'][::2], times_cifar.item()['oe'][1::2])])
# oe_times_mnist =  [(a + b) / 3 for a, b in zip(times_mnist.item()['oe'][::3], times_mnist.item()['oe'][1::3])]
#
# for ap in approaches[1:]:
#     if ap == 'lsa':
#         times_lsa = np.load('lsa_mnist_times.npy', allow_pickle=True)
#         ap_times = [(a + b) / 3 for a, b in zip(times_lsa[::3], times_lsa[1::3])]
#         print(rp.ttest(pd.Series(ap_times), pd.Series(oe_times_mnist), group1_name='oe_times_mnist', group2_name='_times_mnist'+ap, equal_variances=False))
#         #print(ap + " MNIST", wilcoxon(oe_times_mnist, ap_times))
#
#     elif ap == 'dsa':
#         times_dsa = np.load('dsa_mnist_times.npy', allow_pickle=True)
#         ap_times = [(a + b) / 3 for a, b in zip(times_dsa[::3], times_dsa[1::3])]
#         print(rp.ttest(pd.Series(ap_times), pd.Series(oe_times_mnist), group1_name='oe_times_mnist', group2_name='_times_mnist'+ap, equal_variances=False))
#         #print(ap + " MNIST", wilcoxon(oe_times_mnist, ap_times))
#
#     else:
#         ap_times = [(a + b) / 3 for a, b in zip(times_mnist.item()[ap][::3], times_mnist.item()[ap][1::3])]
#         print(rp.ttest(pd.Series(ap_times), pd.Series(oe_times_mnist), group1_name='oe_times_mnist', group2_name='_times_mnist'+ap, equal_variances=False))
#         #print(ap + " MNIST", wilcoxon(oe_times_mnist, ap_times))
#
# # CIFAR TIMES COHEN's d
#
# times_cifar = np.load('model_times_cifar.npy', allow_pickle=True)
# #print([(a + b) / 2 for a, b in zip(times_cifar.item()['oe'][::2], times_cifar.item()['oe'][1::2])])
# oe_times_cifar =  [(a + b) / 2 for a, b in zip(times_cifar.item()['oe'][::2], times_cifar.item()['oe'][1::2])]
#
# for ap in approaches[1:]:
#     if ap == 'lsa':
#         times_lsa = np.load('lsa_cifar_times.npy', allow_pickle=True)
#         ap_times = [(a + b) / 2 for a, b in zip(times_lsa[::2], times_lsa[1::2])]
#         print(rp.ttest(pd.Series(ap_times), pd.Series(oe_times_cifar), group1_name='oe_times_cifar',
#                        group2_name='_times_cifar' + ap, equal_variances=False))
#         #print(ap + " CIFAR10", wilcoxon(oe_times_cifar, ap_times))
#
#     elif ap == 'dsa':
#         times_dsa = np.load('dsa_cifar_times.npy', allow_pickle=True)
#         ap_times = [(a + b) / 2 for a, b in zip(times_dsa[::2], times_dsa[1::2])]
#         print(rp.ttest(pd.Series(ap_times), pd.Series(oe_times_cifar), group1_name='oe_times_cifar',
#                        group2_name='_times_cifar' + ap, equal_variances=False))
#         #print(ap + " CIFAR10", wilcoxon(oe_times_cifar, ap_times))
#
#     else:
#         ap_times = [(a + b) / 2 for a, b in zip(times_cifar.item()[ap][::2], times_cifar.item()[ap][1::2])]
#         print(rp.ttest(pd.Series(ap_times), pd.Series(oe_times_cifar), group1_name='oe_times_cifar',
#                        group2_name='_times_cifar' + ap, equal_variances=False))
#         #print(ap + " CIFAR10", wilcoxon(oe_times_cifar, ap_times))

#CHECK RQ1: MNIST

check_val = 0.5
print("Checking RQ1")
for ap in approaches:
    data_ap = pd.read_csv('mnist_'+ap+'_auc.csv')
    data_ap_fmnist = data_ap['fmnist']
    data_ap_mnistc = data_ap['mnist-c']
    data_ap_mnistfgsm = data_ap['mnist-fgsm']

    print("FMNIST: ", shapiro(data_ap_fmnist))
    print("MNIST-C: ", shapiro(data_ap_mnistc))
    print("MNIST-FGSM: ", shapiro(data_ap_mnistfgsm))
    #print(ap+" FMNIST : ", wilcoxon(data_ap_fmnist - 0.5))
    #print(ap + " MNIST-C : ", wilcoxon(data_ap_mnistc - 0.5))
    # if ap == 'sm':
    #     continue
    # else:
    #print(ap + " MNIST-FGSM: ", wilcoxon(data_ap_mnistfgsm - 0.5))

#Check RQ2: CIFAR10
for ap in approaches:
    data_ap = pd.read_csv('cifar_'+ap+'_auc.csv')
    data_ap_cifar100 = data_ap['cifar100']
    data_ap_cifar10c = data_ap['cifar10-c']
    data_ap_cifar10fgsm = data_ap['cifar10-fgsm']

    print("CIFAR100: ", shapiro(data_ap_cifar100))
    print("CIFAR10-C: ", shapiro(data_ap_cifar10c))
    print("CIFAR10-FGSM: ", shapiro(data_ap_cifar10fgsm))
    #print(ap+" CIFAR100 : ", wilcoxon(data_ap_cifar100 - 0.5))
    #print(ap + " CIFAR10-C : ", wilcoxon(data_ap_cifar10c - 0.5))
    #print(ap + " CIFAR-FGSM: ", wilcoxon(data_ap_cifar10fgsm - 0.5))
