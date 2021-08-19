import numpy as np
import pandas as pd
import math
import time
import argparse
from tqdm import tqdm
import scipy.linalg
from scipy.linalg import lstsq
import sklearn.metrics
from tqdm import tqdm

tt=time.time()
Y=np.load('bert/Y.npy')
print(time.time()-tt)


tt=time.time()
X=np.load('bert/X.npy', mmap_mode='r')
print(time.time()-tt)


vec=np.load("bert/vec_bert.npy")
kill=vec[0, :].reshape((1, 768))
injured=vec[1, :].reshape((1, 768))

class_weight = {0: 9.1, 1: 2.78, 2: 1.92}

Y_train=np.zeros((Y.shape[0], 768))

for i in range(Y.shape[0]):
    if Y[i, 0]==1: aux_emb=injured #np.ones((1, 300))  #np.concatenate((np.ones((1, 100)), np.zeros((1, 100)), np.zeros((1, 100))), axis=1)    #5*(injured-kill) #np.concatenate((np.ones((1, 100)), np.zeros((1, 100)), np.zeros((1, 100))), axis=1) #kill-injured   #np.concatenate((np.ones((1, 100)), np.zeros((1, 100)), np.zeros((1, 100))), axis=1)   #np.ones((1, 300))
    if Y[i, 1]==1: aux_emb=kill #-np.ones((1, 300))  #np.concatenate((np.zeros((1, 100)), np.ones((1, 100)), np.zeros((1, 100))), axis=1)     #5*(kill-injured)  ##injured-kill #np.concatenate((np.zeros((1, 100)), np.ones((1, 100)), np.zeros((1, 100))), axis=1)   #-np.ones((1, 300))
    if Y[i, 2]==1: aux_emb=np.zeros((1, 768)) #np.zeros((1, 300))  #np.concatenate((np.zeros((1, 100)), np.zeros((1, 100)), np.ones((1, 100))), axis=1) #np.zeros((1, 300))  #np.concatenate((np.zeros((1, 100)), np.zeros((1, 100)), np.ones((1, 100))), axis=1) #np.zeros((1, 300))   #np.concatenate((np.zeros((1, 100)), np.zeros((1, 100)), np.ones((1, 100))), axis=1)
    Y_train[i, :]=aux_emb #+X[i, 300*5:300*6])/2
    #Y_train[i, :]=(np.sqrt(class_weight[Y[i]]))*Y_train[i, :] #/2
np.save("Y_train", Y_train)


y=np.zeros(Y.shape[0])
for i in range(Y.shape[0]):
    if Y[i, 0]==1: y[i]=0
    if Y[i, 1]==1: y[i]=1 
    if Y[i, 2]==1: y[i]=2 




'''
bad_idx=[]
for idx in range(len(Y)):
    contador=0
    aux=np.zeros((1, 300))
    for idx_2 in range(11):
        if (X[idx, idx_2*300:(idx_2+1)*300]**2).sum()!=0.0: ++contador
        aux=X[idx, idx_2*300:(idx_2+1)*300]+aux
    if contador==0:
        bad_idx.append(idx)
    else: X[idx, 1500:1800]=(np.sqrt(class_weight[Y[idx]]))*aux #/contador

X=X[:, 1500:1800]
'''

X=X[:, :768*11]
print("We are doing dead, injury,... AVERAGING!")

print("Sil score for the 200-220k")
print(sklearn.metrics.silhouette_score(X[200000:220000, 5*768:6*768], y[200000:220000]))

for size in tqdm([100000, 200000, 150000]):
    print("New size is "+ str(size))
    p, res, rnk, s = lstsq(X[:size], Y_train[:size])
    np.save("bert/A/A_co"+str(size), p)
    print("Silhouette_score")
    #print("Train")
    #print(sklearn.metrics.silhouette_score(X[:min(size, 20000), 768*5:768*6], y[:min(size, 20000)]))
    print("Val")
    #print(sklearn.metrics.silhouette_score(X[200000:220000, 5*768:6*768], y[200000:220000]))
    print("New Silhouette_score")
    print("Train")
    print(sklearn.metrics.silhouette_score(np.matmul(X[:min(size, 20000)], p), y[:min(size, 20000)]))
    print("Val")
    print(sklearn.metrics.silhouette_score(np.matmul(X[200000:220000], p), y[200000:220000]))
    print("__________________")
