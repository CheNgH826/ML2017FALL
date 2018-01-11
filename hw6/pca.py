from skimage.io import imread, imsave
import os
import numpy as np
import copy
import sys
import time

start = time.time()

def img_norm(A):
    M = copy.deepcopy(A)
    M -= np.min(M)
    M /= np.max(M)
    M = (M*255).astype(np.uint8)
    return M

IMAGE_PATH = os.path.abspath('.')+'/'+sys.argv[1]
filelist = os.listdir(IMAGE_PATH)
img_shape = imread(IMAGE_PATH+filelist[0]).shape
img_npy_name = 'img_data.npy'

if os.path.isfile(img_npy_name):
    img_data = np.load(img_npy_name)
else:
    img_data = []
    for i, filename in enumerate(filelist):
        img_data.append(imread(IMAGE_PATH+filename).reshape(-1,))
    img_data = np.array(img_data)

mean = np.mean(img_data, axis=0)
x = (img_data-mean).T
u, s, v = np.linalg.svd(x, full_matrices=False)
eigonvector = u.T

picked_img = imread(IMAGE_PATH+sys.argv[2]).reshape(-1,)
weight = np.array([np.dot(picked_img-mean, eigonvector[i]) for i in range(415)])
reconstruct = img_norm(np.dot(eigonvector[:4].T, weight[:4])+mean.astype(np.uint8))
imsave('reconstruction.jpg', reconstruct.reshape(img_shape))
#myreconstruct = img_norm(np.dot(eigonvector.T, weight)+mean.astype(np.uint8))
#imsave('myreconstruction.jpg', myreconstruct.reshape(img_shape))

print(time.time()-start)
