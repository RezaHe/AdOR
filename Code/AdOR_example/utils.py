import numpy as np
import scipy as sc
from sklearn.neighbors import NearestNeighbors



def accuracy_MSE(y_pred, y_true, ReduceBias=True):
    if ReduceBias is True:
        y_pred = y_pred-np.mean(y_pred)+np.mean(y_true)
    return np.mean(np.square(y_pred - y_true))



def accuracy_MAE(y_pred, y_true, ReduceBias=True):
    if ReduceBias is True:
        y_pred = y_pred-np.mean(y_pred)+np.mean(y_true)
    return np.mean(np.abs(y_pred - y_true))



def kNNEntropy(Samples, KOfNN= 20):
    n_samples, dimdata = np.shape(Samples)
    Samples = Samples - np.mean(Samples)
    
    nbrs = NearestNeighbors(n_neighbors=KOfNN, algorithm='ball_tree', metric='euclidean').fit(Samples)
    distances, _  = nbrs.kneighbors(Samples)
    KDists = distances[:,-1]
    
    cd = np.pi**(dimdata/2)/sc.special.gamma(1+dimdata/2)
    Entrop = np.log(n_samples)-sc.special.psi(KOfNN)+np.log(cd)+dimdata/n_samples*np.sum(np.log(KDists))
    return Entrop



def batch_index_generator(N_Samples, Batch_Size, NumColumn):
    NumbersArray = np.arange(0, N_Samples)
    ExtendedNums = np.tile(NumbersArray, np.int(np.ceil(NumColumn*Batch_Size/N_Samples)))
    MixedNumbers = np.random.permutation(ExtendedNums)
    MixedNumbers = MixedNumbers[0:NumColumn*Batch_Size]
    return np.reshape(MixedNumbers, (Batch_Size, NumColumn))