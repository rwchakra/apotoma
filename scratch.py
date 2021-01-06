import pickle

with open('results/mnist/dsa_rand5_perc.pickle', "rb") as f:
    obj = pickle.load(f)
    print(obj)