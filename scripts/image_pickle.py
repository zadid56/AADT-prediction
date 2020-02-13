import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

for i in range(1, 3960):
    file = open('data/im3_'+str(i)+'.pkl', 'rb')
    image = pickle.load(file)
    file.close()
    tr = scipy.ndimage.rotate(image,-90)
    tr = np.flip(tr, 1)
    plt.imsave('/home/mdzadik/Yolo_mark/x64/Release/data/img/im3_'+str(i)+'.png',tr)
    



