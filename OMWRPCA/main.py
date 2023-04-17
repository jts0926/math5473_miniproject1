import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from rpca.omwrpca import omwrpca

data = scio.loadmat('./data/shoppingmall.mat')
m1 = data['frame_m'][0][0].astype(int)
m2 = data['frame_n'][0][0].astype(int)
X = data['shoppingmall']
imgs = X.reshape((X.shape[0],m2,m1)).transpose((0,2,1))

Lhat, Shat, rank = omwrpca(np.transpose(X), burnin=200, win_size=200, lambda1=1.0/np.sqrt(80000), lambda2=1.0/np.sqrt(80000)*(100))

imgs_L = Lhat.T.reshape((X.shape[0],m2,m1)).transpose((0,2,1))
imgs_S = Shat.T.reshape((X.shape[0],m2,m1)).transpose((0,2,1))

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
plt.savefig("./output/OMWRPCA.jpg")

S_mask = imgs_S[369].copy()
S_mask[np.abs(S_mask)<10] = 0
S_mask[S_mask!=0] = 255
plt.imshow(S_mask,cmap='gray')
plt.savefig("./ouput/OMWRPCA_mask.jpg")