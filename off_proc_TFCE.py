# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 09:05:04 2021

@author: elico
"""

import numpy as np
from numba import prange,njit
from scipy.stats import wilcoxon,pearsonr
import scipy.io as sio
from tqdm import tqdm
import random



#@njit(parallel=True)
def clustermap(arr,windows,f1s,f2s,cluster_map):
    """
    

    Parameters
    ----------
    arr : 3D array
    windows : int
        size of first axis of arr
    f1s : int
        size of second axis of arr
    f2s : int
        size of third axis of arr
    cluster_map: 

    Returns
    -------
    cluster_map : 3D array, same shape as arr
        Runs through each row of 3D array for each window(first axis) and sets each pixel equal to the number
        of non-zero pixels it is adjacent too in that row.

    """

    for w in prange(windows):
        for f1 in prange(f1s):
                clustsize = 0
                start_idx = np.Inf
                for r in range(f2s):
                    #Only start counting size of cluster if value is non-zero and greater than f1, as this function is meant to run only on the upper half of 
                    #each window
                    if arr[w,f1,r] != 0.0 and f1<r:
                        #Start counting cluster size
                        clustsize += 1
                        
                        #np.Inf == start_idx if this is the first element of a new cluster, so safe to initilize start_idx to index of this value
                        if start_idx == np.Inf:
                            start_idx = r
                        #If r is the last element, save cluster size and exist loop
                        if r + 1 == f2s:
                            cluster_map[w,f1,start_idx:r+1]+=clustsize
                            break
                        #If next value after r is zero, save cluster size, re-initilize cluster_size and start_idx for next cluster (if exists)
                        elif arr[w,f1,r+1] == 0:
                            cluster_map[w,f1,start_idx:r+1]+=clustsize
                            clustsize = 0 
                            start_idx = np.Inf
        

    return cluster_map

"""
=====
Optional Testing for clustermap
=====
"""
def test_clustermap():
    third_axis = 999
    
    #Generate 3 random indexs
    a = random.randint(0,998)
    b = random.randint(0,998)
    c = random.randint(0,998)
    randos = np.sort(np.asarray([a,b,c]))
    zero = np.asarray([0])
    size = np.asarray([998])
    arr = np.ones((1,1,999))
    
    #index/edges of all contiguous non-zero clusters
    perm_idxs = np.concatenate((zero,randos,size),axis=0)
    
    corr_ans = np.zeros((999,))
    for ip in range(1,5):
        last_p = perm_idxs[ip-1]
        p = perm_idxs[ip]
        
        if last_p < p:
            size = p - last_p - 1
            corr_ans[last_p+1:p] = size * np.ones((size,))
            arr[0,0,last_p] = 0
        
        arr[0,0,p] = 0
            
                        
                                       
    assert(np.array_equal(corr_ans,clustermap(arr,1,1,999,np.zeros((1,1,999)))[0,0,:]))
       

for test in tqdm(range(1000)):
    test_clustermap()



        
def TFCE(arr):
    """
    Parameters
    ----------
    arr : 1 or 3D numpy array 
        of (either (999,) or (9,100,100)), though can be altered to deal with different dimensions

    Returns
    -------
    1 or 3D numpy array of same shape as input array
        Performs Threshold-Free Cluster Enhancement (https://www.sciencedirect.com/science/article/pii/S1053811908002978?via%3Dihub)
        on a numpy array. 

    """
    #Initialize series of thresholds
    thresh = np.arange(0,np.max(arr),0.01)
    
    
    #if 1D array, reshape. Clustermap function accepts only 3D functions  
    dim = arr.ndim
    arr,orig_shape = process(arr)
    new_shape = arr.shape
    
    #Intilize zero vector, which will store values obtained by TFCE algorithm
    tfce_mat = np.zeros((new_shape))
    

    for threshold in thresh[2:]:
        #Set all array elements below threshold to zero
        arr[arr<threshold] = 0
        
        #Fill cluster_map with output of clustermap function run on arr. If arr has more than one column (axis = 1), runs clustermap
        #on columns as well.
        cluster_map = np.zeros(new_shape)
        if dim > 2:
            
            contig_clusters = clustermap(arr,9,100,100,cluster_map) + clustermap(np.transpose(arr,axes=(0,2,1)),9,100,100,cluster_map)
        else:
            contig_clusters = clustermap(arr,1,1,999,cluster_map)
            
        #For each pixel, raises the value representing how many/large the clusters
        #it belongs too are by the square of the threshold
        tfce_mat += np.power(contig_clusters,0.5) * (threshold**2)
        
    return np.reshape(tfce_mat,orig_shape)


@njit(parallel=True)
def calc_enot(sessions,windows,f1s,f2s,results):
    """
    

    Parameters
    ----------
    sessions : 3D array, (44,999,1182)


    results : 3D array, (44,999,1182), zeros 

    Returns
    -------
    results : Filled with Lempel-Zev complexity along 2nd axis

    """
    for sess in prange(44):
        for freq in prange(999):
            ss = sessions[sess,freq,:]
            ss[ss>np.mean(ss)]=1
            ss[ss<=np.mean(ss)]=0
            i, k, l = 0, 1, 1
            c, k_max = 1, 1
            n = 1182
            while True:
                if ss[i + k - 1] == ss[l + k - 1]:
                    k = k + 1
                    if l + k > n:
                        c = c + 1
                        break
                else:
                    if k > k_max:
                        k_max = k
                    i = i + 1
                    if i == l:
                        c = c + 1
                        l = l + k_max
                        if l + 1 > n:
                            break
                        else:
                            i = 0
                            k = 1
                            k_max = 1
                    else:
                        k = 1
            results[f,s] = c
        return results
    
    
@njit(parallel=True)
def calc_enot_fr(sessions,results):
    """
    

    Parameters
    ----------
    sessions : 3D array, (44,999,1182)

    results : 3D array, (9,100,100), zeros

    Returns
    -------
    results : Fills results with Lempel-Zev complexity values of power ratios of nearby frequencies (<20 Hz) over all time/sessions

    """
    f1 = 0
    f2 = 0
    for w in prange(9):
        for f_1 in prange(100):
            for f_2 in prange(100):
                if f_1<f_2:
                    f1 = f1 + f_1
                    f2 = f2 + f_1
                    ss = np.divide(sessions[:,f1,:],sessions[:,f2,:])
                    ss[ss == np.Inf]=0
                    ss[np.isnan(ss)]=0
                    ss[ss>np.mean(ss)]=1
                    ss[ss<=np.mean(ss)]=0
                    i, k, l = 0, 1, 1
                    c, k_max = 1, 1
                    n = len(ss)
                    while True:
                        if ss[i + k - 1] == ss[l + k - 1]:
                            k = k + 1
                            if l + k > n:
                                c = c + 1
                                break
                        else:
                            if k > k_max:
                                k_max = k
                            i = i + 1
                            if i == l:
                                c = c + 1
                                l = l + k_max
                                if l + 1 > n:
                                    break
                                else:
                                    i = 0
                                    k = 1
                                    k_max = 1
                            else:
                                k = 1
                    results[w,f1,f2] = c
        f1 += 100
        f2 += 100
    return results

def enot_manager(sessions):
    results = np.zeros((999,44))
    calc_enot(sessions,results)
    return results

@njit
def pearson_fr(trials,labels,results):
    """
    

    Parameters
    ----------
    trials : (9,100,100) numpy array
    
    labels : 1D numpy array

    results : (9,100,100) numpy array


    Returns
    -------
    (9,100,100) numpy array, each entry is pearson correlation value between average power ratio of those frequencies and drinking labels

    """

    for split in prange(9):
        for f1 in prange(100):
            for f2 in prange(100):
                if f1 < f2:
                  results[split,f1,f2] = np.corrcoef(trials[split,f1,f2,:],labels)[0,1]
    

def process(arr):
    """
    

    Parameters
    ----------
    arr : 1-3D numpy array
        

    Returns
    -------
    (3D array, seq) - If array is 1D (999,), reshapes it to 3D (1,1,999). Else passes back. Returns with seq (3 elements) representing shape

    """
    #Save original shape of the array
    orig_shape = arr.shape
    
    #if 1D array, reshape. Clustermap function accepts only 3D functions
    if arr.ndim == 1:
        arr = np.reshape(arr,(1,1,-1))
    elif arr.ndim == 2:
        r,c = arr.shape
        arr = np.reshape(arr,(1,r,c))
        
    return arr,orig_shape

def run_pearson(trials,labels,opt):
    """
    

    Parameters
    ----------
    trials : 2D numpy array
        
    labels : 1D array
    
    opt: string, determines whether or not to perform correlation between average power over session or average power over session ratios

    Returns
    -------
    results : 3D
        returns pearson correlation values between labels average power ratios between nearby frequencies

    """
    #Change for just binges or binge/breaks, between 92 and 44
    #If array is 1D, reshape to 3D and save original shape
    if opt == 'fr':
        results = np.zeros((9,100,100))
        pearson_fr(trials,labels,results)
    else:
        results = pearson_avg(trials,labels)
    return results



def pearson_avg(sessions,drinks):
    """
    

    Parameters
    ----------
    sessions : (999,n) array, where n is int

    drinks : (n,) array


    Returns
    -------
    results : (999,)
        pearson correlation along second dimension of sessions and drinks

    """
    
    results = np.zeros((999,))
    
    for i in range(999):
        results[i],_ = pearsonr(sessions[i,:],drinks)
        
    return results

def prep_pearson(pfr_table):
    """
    

    Parameters
    ----------
    pfr_table : 1 or 2D array


    Returns
    -------
    pos_pearson : same shape as pfr_table
        All pfr_table values above 0
    neg_pearson : same shape as pfr_table
         All pfr_table values below 0

    """
    pos_pearson = np.zeros((pfr_table.shape))
    neg_pearson = np.zeros((pfr_table.shape))
    pos_idxs = np.where(pfr_table>0)
    neg_idxs = np.where(pfr_table<0)
    
    pos_pearson[pos_idxs] = pfr_table[pfr_table>0]
    neg_pearson[neg_idxs] = np.abs(pfr_table[pfr_table<0])
    
    return pos_pearson,neg_pearson

def wilcoxon_avg(sessions):
    """
    

    Parameters
    ----------
    sessions : (999,n) array of session differences


    Returns
    -------
    results : 
        (999,) array, wilcoxon statistic (greater) between corresponding frequencies

    """
    
    results = np.zeros((999,))
    
    for i in range(999):
        results[i],_ = wilcoxon(sessions[i,:],alternative='greater')
        
    return results

def convert_to_zval(arr1,arr2):
    """
    

    Parameters
    ----------
    arr1 : ndarray

    arr2 : ndarray

    Returns
    -------
    (arr1,arr2) converted to zvals, determined by mean and standard deviation of both

    """
    full_dataset = np.concatenate(arr1.flatten(),arr2.flatten())
    arr1 = (arr1 - np.mean(full_dataset)) / np.std(full_dataset)
    arr2 = (arr2 - np.mean(full_dataset)) / np.std(full_dataset)
    
    return (arr1,arr2)

def build_w_null_distro(binges,breaks):
    """
    Parameters
    ----------
    binges : 2D array

    breaks : 2D array


    Returns
    -------
    real_scores : 1D array
        TFCE-enhanced Wilcoxon difference scores between corresponding columns (along second dim) of binges and breaks
    null_distro : 1D array
       Max TFCE-enhanced Wilcoxon difference scores between random groupings of binges and breaks across 1000 iterations

    """
    iter = 1000
    all_trials = np.hstack((binges,breaks))
    idxs = np.linspace(0,87,88).astype(np.int)
    
    null_distro = np.zeros((iter,2))
    perms = np.zeros((iter,999))
    
    for i in tqdm(range(iter)):

        np.random.shuffle(idxs)
        perm = all_trials[:,idxs]
        perm_binges = perm[:,:44]
        perm_breaks = perm[:,44:]
        results = wilcoxon_avg(perm_binges - perm_breaks)
        perms[i,:] = results
    
    real_scores = wilcoxon_avg(binges-breaks)
    real_scores,perms = convert_to_zval(real_scores,perms)
    pos_perms,neg_perms = prep_pearson(perms)
    pos_scores,neg_scores = prep_pearson(real_scores)
    
    for i in tqdm(range(iter)):  
        null_distro[i,1] = np.max(TFCE(pos_perms[i,:]))
        null_distro[i,2] = np.max(TFCE(neg_perms[i,:]))

    pos_scores,neg_scores = TFCE(pos_scores),TFCE(neg_scores)
    
    return real_scores,null_distro


def build_p_null_distro(sessions,drinks):
    """
    Parameters
    ----------
    binges : 2D array

    drinks : 1D array


    Returns
    -------
    real_scores : 1D array
        TFCE-enhanced correlation coef. between each column (2nd dim) of sessions and drinks
    null_distro : 2D array
       Max TFCE-enhanced positive and negative correlation coef. between binges and random permutations of drinking values across 1000 iterations
    """
    iter = 1000
    null_distro = np.zeros((iter,2))

    
    for i in tqdm(range(iter)):
        np.random.shuffle(drinks)
        pos_pearson,neg_pearson = prep_pearson(run_pearson(sessions,drinks))
        pos_pearson_TFCE = TFCE(pos_pearson)
        neg_pearson_TFCE = TFCE(neg_pearson)
        null_distro[i,0] = np.max(pos_pearson_TFCE)
        null_distro[i,1] = np.max(neg_pearson_TFCE)
    return null_distro


@njit(parallel=True)
def build_freqratio_numba(sessions,results):
    """
    

    Parameters
    ----------
    sessions : (100,n) array
    results : (100,100,n) array of zeros


    Returns
    -------
    results : (100,100,n) array
        fills results with (x/y,) where x and y are rows in sessions (x < y i.e. no recipricols)

    """
    for f1 in prange(100):
        for f2 in prange(100):
            if f1 < f2:
                results[f1,f2,:] = np.divide(sessions[f1,:],sessions[f2,:])
                results[f1,f2,:][results[f1,f2,:] == np.Inf]=0
                results[f1,f2,:][np.isnan(results[f1,f2,:])]=0
                
    return results
                
def build_freqratio_table(sessions,arg):
    """
    

    Parameters
    ----------
    sessions : (999,arg) array
    arg : int


    Returns
    -------
    results : 
        (9,100,100,arg) array, each window (first axis, 9) is matrix of power ratios of frequencies within 100 elements (20 Hz) of one another

    """
    results = np.zeros((9,100,100,arg))
    start = 0
    end = 100
    for piece in range(9):
        results[piece] = build_freqratio_numba(sessions[start:end],results[piece])
        start = end
        end = end+100
    return results             


def sep_pos_neg(null_distro):
    """
    

    Parameters
    ----------
    null_distro : (1000,2)

    Returns
    -------
    (1D array, 1D array)
        Seperates 2D array with depth of two along first axis into two arrays, splitting between the first and second element of the second dimension

    """
    return np.reshape(null_distro[:,0],(-1,)),np.reshape(null_distro[:,1],(-1,))

    
def make_nparray(data):
    """
    

    Parameters
    ----------
    data : (999,n) array, with 100000 as a delineator between seperate recordings.

    Returns
    -------
    (44,999,1182) array. For each session (44) and each frequency (999), the power time series (cut to a uniform 1182 SFFT durations in length)

    """
    trial_dfs = []
    neg_ind = [0]
    indices = np.asarray(np.where(data[0,:] == 100000))[0]
    for idx in indices:
        neg_ind.append(idx+1)
    for i in range(44):
        session = data[:,neg_ind[i]:indices[i]]
        trial_dfs.append(session[:,:1182])
    return np.asarray(trial_dfs)
        
folder = 'C:\\Users\\elico\\Desktop\\ETOH_Export\\Nondrunkbinges_removed\\DataFiles\\'
freqs = np.linspace(0.4,200,999)

#Load non-drinking binges removed dataset (average power over sessions and drinking values) 
#binges_avg = np.asarray(sio.loadmat(r'C:\Users\elico\Desktop\ETOH_Export\Nondrunkbinges_removed\Avg_BLApwr_persession\binges.mat')['binges'])
#breaks_avg = np.asarray(sio.loadmat(r'C:\Users\elico\Desktop\ETOH_Export\Nondrunkbinges_removed\Avg_BLApwr_persession\breaks.mat')['breaks'])[:,:44]
#binges_ts = make_nparray(np.asarray(sio.loadmat(r'C:\Users\elico\Desktop\ETOH_Export\Nondrunkbinges_removed\BLA\binges.mat')['binges']))
#breaks_ts = make_nparray(np.asarray(sio.loadmat(r'C:\Users\elico\Desktop\ETOH_Export\Nondrunkbinges_removed\BLA\breaks.mat')['breaks']))
#labels = np.asarray(sio.loadmat(r'C:\Users\elico\Desktop\ETOH_Export\Nondrunkbinges_removed\drinks.mat')['drinks'])[0]