# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 09:05:04 2021

@author: elico

Usage Docs/Examples at bottom
"""

import numpy as np
from numba import prange,njit
from scipy.stats import wilcoxon,pearsonr
import scipy.io as sio
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt




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
       
    
"""
Uncomment to test clustermap

for test in tqdm(range(1000)):
    test_clustermap()
"""


        
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
        on either the last (if 1D) or last 2 (if 3D) dimensions

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




@njit
def pearson_fr(trials,labels,results):
    """
    

    Parameters
    ----------
    trials : (9,100,100,44) numpy array
    
    labels : 1D numpy array

    results : empty (9,100,100) numpy array


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
    (3D array, seq) - If array is 1D (999,), reshapes it to 3D (1,1,999). Else passes back. seq = original shape of array

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
    trials : 2D or 3D numpy array, depending on whether you want to perform tests on individual frequencies or frequency ratios
        
    labels : 1D array
    
    opt: string, determines whether or not to perform correlation between average power over session or average power over session ratios. 
    "fr" if you wish to perform tests on frequency ratios

    Returns
    -------
    results : 1D or 3D array
        returns pearson correlation values between labels average power ratios (between nearby frequencies/or for individual frequencies)

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
        All pfr_table values above 0 (positive correlation for example)
    neg_pearson : same shape as pfr_table
         absolute value of all pfr_table values below 0 (negative correlation for example)

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
    full_dataset = np.hstack((arr1.flatten(),arr2.flatten()))
    arr1 = (arr1 - np.mean(full_dataset)) / np.std(full_dataset)
    arr2 = (arr2 - np.mean(full_dataset)) / np.std(full_dataset)
    
    return (arr1,arr2)

def get_threshold(null_distro):
    """
    

    Parameters
    ----------
    null_distro : array (1000,), Null distribution

    Returns
    -------
    int, returns 95% significance threshold

    """
    alpha_threshold = 50
    sorted_null = np.sort(null_distro)
    return np.floor(sorted_null[1000-alpha_threshold])

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
    idxs = np.linspace(0,87,88).astype(int)
    
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
        null_distro[i,0] = np.max(TFCE(pos_perms[i,:]))
        null_distro[i,1] = np.max(TFCE(neg_perms[i,:]))

    pos_scores,neg_scores = TFCE(pos_scores),TFCE(neg_scores)
    
    return pos_scores,neg_scores,null_distro


def build_p_null_distro(sessions,drinks):
    """
    Parameters
    ----------
    binges : 2D or 3D array

    drinks : 1D array


    Returns
    -------
    null_distro : 2D array
       Max TFCE-enhanced positive and negative correlation coef. between binges and random permutations of drinking values across 1000 iterations
    """
    if sessions.ndim > 2:
        opt = 'fr'
    else:
        opt = ''
    iter = 1000
    null_distro = np.zeros((iter,2))
    print('Permutation Testing...')
    
    for i in tqdm(range(iter)):
        np.random.shuffle(drinks)
        pos_pearson,neg_pearson = prep_pearson(run_pearson(sessions,drinks,opt))
        null_distro[i,0] = np.max(TFCE(pos_pearson))
        null_distro[i,1] = np.max(TFCE(neg_pearson))
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

def normalize(arr):
    """
    

    Parameters
    ----------
    arr : 2D array

    Returns
    -------
    norm : 2D normalized array (min-max normalization)


    """
    minim = np.reshape(np.min(arr,axis=1),(999,1))
    maxim = np.reshape(np.max(arr,axis=1),(999,1))
    norm = (arr - minim) / (maxim-minim)
    norm[norm == np.Inf]=0
    norm[np.isnan(norm)]=0
    return norm

def run_pearson_all(conv,labels):
    """
    

    Parameters
    ----------
    conv : binges (999,44) or (9,100,100,44) array

    labels : drinks (44,) array

    Returns
    -------
    binges are run through full pipeline i.e. correlation and then
    enhancement by TFCE
    tfce_reg_pos_pearson : positive correlations, same shape as input minus
    the last dimension ((999,44 -> (999,)))
    
    tfce_reg_neg_pearson : negative correlations, same shape as tfce_reg_pos_pearson

    """
    if conv.ndim > 2:
        opt = 'fr'
    else:
        opt = ''
    print('Computing Correlation Scores...')
    reg_pos_pearson,reg_neg_pearson = prep_pearson(run_pearson(conv,labels,opt))
    print('Running TFCE...')
    tfce_reg_pos_pearson = TFCE(np.copy(reg_pos_pearson))
    tfce_reg_neg_pearson = TFCE(np.copy(reg_neg_pearson))
    
    return (tfce_reg_pos_pearson,tfce_reg_neg_pearson)
    
def run_wilcoxon_all(conv):
    """
    

    Parameters
    ----------
    conv : (999,44) array = binges - breaks

    Returns
    -------
    binges/break difference are run through full pipeline i.e. wilcoxon signed rank test
    and then enhancement by TFCE
    tfce_pos_w : (999,) - TFCE-enhnaced difference scores per frequency, for positive z-vals
    tfce_neg_w : (999,) - TFCE-enhnaced difference scores per frequency, for negative z-vals

    """
    print('Computing Difference Scores...')
    wilcox = wilcoxon_avg(conv) 
    pos_w,neg_w = prep_pearson((wilcox - np.mean(wilcox))/np.std(wilcox))
    print('Running TFCE...')
    tfce_pos_w,tfce_neg_w = TFCE(pos_w),TFCE(neg_w)
    
    return tfce_pos_w,tfce_neg_w

"""
The following four functions are for visualization
for saving/showing the end-product of the power ratio analysis
and for saving/showing the end-product of the single frequency analysis

fr_table,results,arr - are the end_result of running a dataset 
through their respective pipelines. Results and arr through 
single frequency, fr_table through frequency ratio.

title - used in title generation (...) i.e. "positive" or "negative"

show - 0 indicates the end_product should not be displayed, whereas 1...
used by the save functions to supress output
"""
def graph_ratio_table(fr_table,title,show):
    fig,axes=plt.subplots(nrows=3,ncols=3,figsize=(30,30))
    x = 0
    y = 20
    
    
    cntr=0
    for row in axes:
        for col in row:
            freqs_marks = np.linspace(0,100,20)
            freqs_labels = np.floor(np.linspace(x,y,20)).astype(np.int16)
            sns.heatmap(fr_table[cntr],ax=col)
            max_score = np.round(np.max(fr_table[cntr]),decimals=2)
            col.set_xticks(freqs_marks)
            col.set_xticklabels(freqs_labels)
            col.set_yticks(freqs_marks)
            col.set_yticklabels(freqs_labels,rotation=0)
            col.text(50,5,'Max Score:'+str(max_score), ha='left',va='top',fontsize=20,
                 bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
            col.set_xlabel('Frequency (Hz)')
            col.set_ylabel('Frequency (Hz)')
            col.set_title(str(x)+'-'+str(y)+'/'+str(x)+'-'+str(y),fontdict={'fontsize': 20})
            cntr+=1
            x+=20
            y+=20
        
    fig.suptitle(title + ' of alcohol levels with power ratios across binges',fontsize=40)
    if show == 1:
        plt.show()
        plt.close()
    return fig

def graph_freqdomain(results,threshold,title,show):
    sns.lineplot(x=freqs,y=results)
   
    
    if threshold > 0:
        xtics,_ = plt.xticks()
        plt.text(np.percentile(xtics,90), threshold-1,'95% Significance', ha='right',va='top',fontsize=10)
        plt.axhline(y=threshold,color='r')

    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('TFCE-enhanced Score')
    plt.title(title + ' with alcohol levels across binges',fontsize=20)
    if show == 1:
        plt.show()
        plt.close()
    return plt
    
def save_freqdomain(arr,title,folder):
    plt = graph_freqdomain(arr,0.0,title,0)
    plt.title(title,fontsize=20)
    plt.savefig(folder+title+'.png')
    plt.close()

def save_ratio_table(arr,title,folder):
    fig = graph_ratio_table(arr,title,0)
    plt.savefig(folder+title+'.png')
    plt.close()

"""
Usage docs/Examples

"""
#Initialize basic values, load data, binges and breaks must be of the same shape (note the cutoff of 44 in the case of breaks)
#Binge/Break shape (999,44), labels shape (44,)
freqs = np.linspace(0.4,200,999)
binges = np.asarray(sio.loadmat(r'C:\Users\elico\Desktop\ETOH_Export\Nondrunkbinges_removed\Avg_BLApwr_persession\binges.mat')['binges'])
breaks = np.asarray(sio.loadmat(r'C:\Users\elico\Desktop\ETOH_Export\Nondrunkbinges_removed\Avg_BLApwr_persession\breaks.mat')['breaks'])[:,:44]
labels = np.asarray(sio.loadmat(r'C:\Users\elico\Desktop\ETOH_Export\Nondrunkbinges_removed\drinks.mat')['drinks'])[0]
no_binges = binges.shape[1]

"""
Example Use: TFCE-enhanced Pearson correlation of drinking values and average power over sessions for a single frequency
"""
#Compute correlation for each frequency and run TFCE on the resulting vector
#pos_p and neg_p are (999,) arrays. They are TFCE-enhanced positive and negative
#correlation respectively
pos_p, neg_p = run_pearson_all(binges,labels)

#Generate null distribution to compute significance
pos_p_nd,neg_p_nd = sep_pos_neg(build_p_null_distro(binges,labels))

#Get 95% percent signficance threshold
pos_p_thresh = get_threshold(pos_p_nd)
neg_p_thresh = get_threshold(neg_p_nd)

#Show the resulting graph, with a line for significance...
graph_freqdomain(pos_p,pos_p_thresh,'Positive correlation',1)
graph_freqdomain(neg_p,neg_p_thresh,'Negative correlation',1)


#...or save it to a folder of your choice. NOTE: do not do both, as the graphs 
# will not save properly if they have been displayed
#folder ='C:\\Users\\elico\\Desktop\\Graphs\\' 
save_freqdomain(pos_p,pos_p_thresh,'Positive correlation',folder)
save_freqdomain(neg_p,neg_p_thresh,'Negative correlation',folder)

"""
Example Use: TFCE-enhanced Difference Score between binges and breaks for corresponding frequencies
"""
#Compute Wilcoxon signed rank scores for each average power difference between conditions
# and run TFCE on the resulting vector. Also compute w_nd (null_distro) of max TFCE values.
#pos_w and neg_w are (999,) arrays. Pos_w contains values which are greater 
#in binges than breaks. Neg_w contains values which are smaller in binges than breaks.
#They have both been run through TFCE.
pos_w, neg_w, w_nd = build_w_null_distro(binges,breaks)


#Seperate null distribution into null_distro for positive and negative difference scores
pos_w_nd,neg_w_nd = sep_pos_neg(w_nd)

#Get 95% percent signficance threshold
pos_w_thresh = get_threshold(pos_w_nd)
neg_w_thresh = get_threshold(neg_w_nd)

#Show the resulting graph, with a line for significance...
graph_freqdomain(pos_w,pos_w_thresh,'Positive difference',1)
graph_freqdomain(neg_w,neg_w_thresh,'Negative difference',1)

#...or save it to a folder of your choice. NOTE: do not do both, as the graphs 
# will not save properly if they have been displayed
folder = 'C:\\Users\\elico\\Desktop\\Graphs\\' 
save_freqdomain(pos_p,pos_p_thresh,'Positive correlation',folder)
save_freqdomain(neg_p,neg_p_thresh,'Negative correlation',folder)


"""
Example Use: TFCE-enhanced correlation of the power ratios of neighboring frequencies with alcohol consumed
"""
#Build frequency ratio table.
#pfr_table = build_freqratio_table(binges,no_binges)

#Compute correlation for each frequency and run TFCE on the resulting vector
#pos_p and neg_p are (999,) arrays. They are TFCE-enhanced positive and negative
#correlation respectively
pos_pfr, neg_pfr = run_pearson_all(pfr_table,labels)

#Generate null distribution to compute significance. NOTE: this operation is take a while.
#Only run if you want a break to stretch and get a snack
pos_pfr_nd,neg_pfr_nd = sep_pos_neg(build_p_null_distro(pfr_table,labels)

#Get 95% percent signficance threshold
pos_pfr_thresh = get_threshold(pos_pfr_nd)
neg_pfr_thresh = get_threshold(neg_pfr_nd)

#Show the resulting graph, with a line for significance...
graph_ratio_table(pos_pfr,pos_pfr_thresh,'Positive correlation',1)
graph_ratio_table(neg_pfr,neg_pfr_thresh,'Negative correlation',1)


#...or save it to a folder of your choice. NOTE: do not do both, as the graphs 
# will not save properly if they have been displayed
folder ='C:\\Users\\elico\\Desktop\\Graphs\\' 
save_ratio_table(pos_pfr,pos_pfr_thresh,'Positive correlation',folder)
save_ratio_table(neg_pfr,neg_pfr_thresh,'Negative correlation',folder)


















