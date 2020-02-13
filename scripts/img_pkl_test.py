import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

file = open('data/im1_1500.pkl', 'rb')
image = pickle.load(file)
file.close()
tr = scipy.ndimage.rotate(image,-90)
tr = np.flip(tr, 1)
print(tr.shape)
b, g, r = np.split(tr, 3, axis=2)
tr2 = np.concatenate((r,b,g), axis=2)
plt.imsave('im1_1500.png',tr2)
