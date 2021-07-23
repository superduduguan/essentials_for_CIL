import os
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np

I = [0, 50, 60, 70, 80, 90]
P = []
for i in I:
    with open(str(i) + 'X.data', 'rb') as ff:
        try:
            P.append(pickle.load(ff).cpu().numpy())
        except Exception as e:
            print(e)
for p in P:
    psum = np.linalg.norm(p, axis=1)
    plist = psum.tolist()
    plist = [abs(i) for i in plist]
    index = [i for i in range(len(plist))]
    plt.bar(index, plist)
    plt.show()
