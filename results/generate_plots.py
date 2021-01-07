import numpy as np
import os
import pickle

df = os.listdir('./mnist')[1]

with open('./mnist/'+df, 'rb') as f:
    data = pickle.load(f)

auc_score = data.evals['adv_fga_0.5'].ood_auc_roc
time = data.evals['adv_fga_0.5'].eval_time
print(auc_score, time)
