import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys

sys.path.append('/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma/results/mnist')

file_dsa = os.listdir('./mnist')[1]

with open('./mnist/'+file_dsa, 'rb') as f:
    data_dsa = pickle.load(f)

print(data_dsa)