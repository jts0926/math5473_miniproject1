import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import admm

data = scio.loadmat('./data/shoppingmall.mat')
m1 = data['frame_m'][0][0].astype(int)
m2 = data['frame_n'][0][0].astype(int)
X = data['shoppingmall']
imgs = X.reshape((X.shape[0],m2,m1)).transpose((0,2,1))

result = admm.rpcaADMM(X,0.1,0.1)

imgs_E = result['X1_admm'].reshape((X.shape[0],m2,m1)).transpose((0,2,1))
imgs_S = result['X2_admm'].reshape((X.shape[0],m2,m1)).transpose((0,2,1))
imgs_L = result['X3_admm'].reshape((X.shape[0],m2,m1)).transpose((0,2,1))

plt.figure(figsize=(10, 10), dpi=100)
plt.subplot(1,3,1)
plt.imshow(imgs[369],cmap='gray')
ax = plt.gca()
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
plt.subplot(1,3,2)
plt.imshow(imgs_L[369],cmap='gray')
ax = plt.gca()
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
plt.subplot(1,3,3)
plt.imshow(imgs_S[369],cmap='gray')
ax = plt.gca()
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(wspace=0)
plt.savefig("./ouput/admm.jpg")

S_mask = imgs_S[369].copy()
S_mask[np.abs(S_mask)<1e-2] = 0
S_mask[S_mask!=0] = 255
plt.imshow(S_mask,cmap='gray')
plt.savefig("./ouput/admm_mask.jpg")