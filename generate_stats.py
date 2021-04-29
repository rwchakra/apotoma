import pandas as pd
import numpy as np

# data = pd.read_csv('./csv_files_mnist/lsa_scores.csv')
#
# print(np.mean(data['0']))
# print(np.std(data['0']))
# print(np.mean(data['1']))
# print(np.std(data['1']))
# print(np.mean(data['2']))
# print(np.std(data['2']))

# sm_fmnist = []
# sm_mnistc = []
# sm_mnistfgsm = []
# diss_fmnist = []
# diss_mnistc = []
# diss_mnistfgsm = []
# oe_fmnist = []
# oe_mnistc = []
# oe_mnistfgsm = []
#
# for m in range(1, 11):
#
#     f = pd.read_csv('./csv_files_mnist/model_'+str(m)+'.csv')
#
#     sm_fmnist.append(f.loc[0]['0'])
#     sm_mnistc.append(f.loc[0]['1'])
#     sm_mnistfgsm.append(f.loc[0]['2'])
#     diss_fmnist.append(f.loc[1]['0'])
#     diss_mnistc.append(f.loc[1]['1'])
#     diss_mnistfgsm.append(f.loc[1]['2'])
#     oe_fmnist.append(f.loc[2]['0'])
#     oe_mnistc.append(f.loc[2]['1'])
#     oe_mnistfgsm.append(f.loc[2]['2'])
#
#
# print(np.mean(sm_fmnist), np.std(sm_fmnist))
# print(np.mean(sm_mnistc), np.std(sm_mnistc))
# print(np.mean(sm_mnistfgsm), np.std(sm_mnistfgsm))
# print(np.mean(diss_fmnist), np.std(diss_fmnist))
# print(np.mean(diss_mnistc), np.std(diss_mnistc))
# print(np.mean(diss_mnistfgsm), np.std(diss_mnistfgsm))
# print(np.mean(oe_fmnist), np.std(oe_fmnist))
# print(np.mean(oe_mnistc), np.std(oe_mnistc))
# print(np.mean(oe_mnistfgsm), np.std(oe_mnistfgsm))

sm_cifar100 = []
sm_cifarc = []
sm_cifarfgsm = []
diss_cifar100 = []
diss_cifarc = []
diss_cifarfgsm = []
oe_cifar100 = []
oe_cifarc = []
oe_cifarfgsm = []

for m in range(1, 11):

    f = pd.read_csv('./csv_files_cifar/model_'+str(m)+'.csv')

    sm_cifar100.append(f.loc[0]['0'])
    sm_cifarc.append(f.loc[0]['1'])
    sm_cifarfgsm.append(f.loc[0]['2'])
    diss_cifar100.append(f.loc[1]['0'])
    diss_cifarc.append(f.loc[1]['1'])
    diss_cifarfgsm.append(f.loc[1]['2'])
    oe_cifar100.append(f.loc[2]['0'])
    oe_cifarc.append(f.loc[2]['1'])
    oe_cifarfgsm.append(f.loc[2]['2'])


print(np.mean(sm_cifar100), np.std(sm_cifar100))
print(np.mean(sm_cifarc), np.std(sm_cifarc))
print(np.mean(sm_cifarfgsm), np.std(sm_cifarfgsm))
print(np.mean(diss_cifar100), np.std(diss_cifar100))
print(np.mean(diss_cifarc), np.std(diss_cifarc))
print(np.mean(diss_cifarfgsm), np.std(diss_cifarfgsm))
print(np.mean(oe_cifar100), np.std(oe_cifar100))
print(np.mean(oe_cifarc), np.std(oe_cifarc))
print(np.mean(oe_cifarfgsm), np.std(oe_cifarfgsm))