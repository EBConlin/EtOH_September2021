"""
Created on Mon Sep 27 13:17:29 2021

@author: elico
"""
import math
import numpy as np
import random
from off_proc_TFCE import run_wilcoxon_all, run_pearson_all, normalize, build_freqratio_table, save_freqdomain, save_ratio_table


def generate_avg_session():
    """
    Returns 1/f scaled average powers in (999,) frequency bins
    """

    powers = np.ones((999,)) / freqs
    
    powers += np.asarray([random.random() for f in range(999)])
    
    return powers

    
def generate_full_trials():
    """
    Returns full synthetic data set. ndarray (999,44)
    """
    sessions= np.ones((999,1))
    for s in range(44):
        sessions = np.hstack((sessions,np.reshape(generate_avg_session(),(999,1))))
              
    return sessions[:,1:]

def random_array():
    """
    

    Returns
    -------
    (999,44) array of random values between 0 and 1

    """
    return np.reshape(np.asarray([random.randint(0,1) for x in range(43956)]),(999,44))
def add_noise(sessions):
    """
    

    Parameters
    ----------
    sessions : (999,44) ndarray


    Returns
    -------
    None. Shifts each value in sessions by a random amount

    """

    sessions+=(random_array()**(10*random_array()) - random_array()**(10*random_array()))
 

#Initialize frequency bins and indices for delta, alpha, beta, and gamma bands
freqs = np.linspace(0.4,200,999)
idx_franges = [(8,23),(24,58),(73,150),(200,350),(398,598)]


#These tests output graphs. If you wish to save the resulting test graphs to somewhere other than where this script resides, 
#change folder variable to a pathname. If you wish to save it to this folder, change folder to be an empty string ('')
folder = 'C:\\Users\\elico\\Desktop\\Graphs\\'  

#Generate synthetic data for two sets of trials (999,44) (frequency x sessions)
binges = generate_full_trials()
breaks = generate_full_trials()

#add noise
add_noise(binges)
add_noise(breaks)

#Generate drinking values
labels = np.asarray([random.randint(0,1) for x in range(44)])

#Generate frequency ratio table (9,100,100) of the power ratios of nearby frequencies across sessions
binges_fr = np.copy(binges)
binges_fr = normalize(binges_fr)
pfr_table = build_freqratio_table(binges_fr,44)


#Create three copies of each frequency ratio table to perform tests with drinking values 
conv_fr = np.copy(pfr_table)
inv_fr = np.copy(pfr_table)
shuff_fr = np.copy(pfr_table)
    
#drinks will test the pipeline for detecting positive correlations, inv_drinks will test for negative correlations, and shuffle
#for no correlation
drinks = np.copy(labels)
inv_drinks = 0 - labels
shuff_drinks = np.random.random(44)

#Use drinks, inv_drinks, and shuff_drinks to alter a patch of frequency ratios in different frequency ranges
for w in range(9):
    f1 = 50
    f2 = 50

    for i in range(40):
        for y in range(40):
            conv_fr[w,f1+i,f2+y,:]= conv_fr[w,f1+i,f2+y,:]*drinks*0.01
            inv_fr[w,f1+i,f2+y,:]= inv_fr[w,f1+i,f2+y,:]*inv_drinks*0.01
            shuff_fr[w,f1+i,f2+y,:]= shuff_fr[w,f1+i,f2+y,:]*shuff_drinks*0.01

#After convolution in a particular frequency ratio range, run the pipeline 
#with the original drinking values to detect positive, negative, and no correlation.
#This does not include the generation of a null distribution. That must be done independently
pos_reg,neg_reg = run_pearson_all(conv_fr,labels)
pos_inv,neg_inv = run_pearson_all(inv_fr,labels)
pos_shuff,neg_shuff = run_pearson_all(shuff_fr,labels)

save_ratio_table(pos_reg,'Regular labels-Positive correlation',folder)
save_ratio_table(neg_reg,'Regular labels-Negative correlation',folder)
save_ratio_table(pos_inv,'Inverse labels-Positive correlation',folder)
save_ratio_table(neg_inv,'Inverse labels-Negative correlation',folder)
save_ratio_table(pos_shuff,'Shuffled labels-Positive correlation',folder)
save_ratio_table(neg_shuff,'Shuffled labels-Negative correlation',folder)



#Transform drinks, inv_drinks, and shuff_drinks with delta, alpha, beta, and gamma ranges
for f1,f2 in idx_franges:
    #Create three copies of binges to perform transformation with 3 different sets of drinking values 
    conv = np.copy(binges)
    inv = np.copy(binges)
    shuff = np.copy(binges) 
    
    for flen in range(f2-f1):
        conv[f1+flen,:]= conv[f1+flen,:]*drinks*0.01
        inv[f1+flen,:]= inv[f1+flen,:]*inv_drinks*0.01
        shuff[f1+flen,:]= shuff[f1+flen,:]*shuff_drinks*0.01
        
    

    #After transformation in a particular frequency range, run the pipeline 
    #with the original drinking values to detect positive, negative, and no correlation, 
    #as well as significant difference between binges and breaks.
    #This does not include the generation of a null distribution. That must be done independently
    pos_reg,neg_reg = run_pearson_all(conv,labels)
    pos_inv,neg_inv = run_pearson_all(inv,labels)
    pos_shuff,neg_shuff = run_pearson_all(shuff,labels)
    pos_w_reg,neg_w_reg = run_wilcoxon_all(conv-breaks)
    pos_w_inv,neg_w_inv = run_wilcoxon_all(inv-breaks)
    
    #Save graphs
    save_freqdomain(pos_reg,'reg-pos '+str(freqs[f1])+'-'+str(np.floor(freqs[f2])),folder)
    save_freqdomain(neg_reg,'reg-neg'+str(freqs[f1])+'-'+str(np.floor(freqs[f2])),folder)
    save_freqdomain(pos_inv,'inv-pos'+str(freqs[f1])+'-'+str(np.floor(freqs[f2])),folder)
    save_freqdomain(neg_inv,'inv-neg '+str(freqs[f1])+'-'+str(np.floor(freqs[f2])),folder)
    save_freqdomain(pos_shuff,'shuff-pos'+str(freqs[f1])+'-'+str(np.floor(freqs[f2])),folder)
    save_freqdomain(neg_shuff,'shuff-neg'+str(freqs[f1])+'-'+str(np.floor(freqs[f2])),folder)
    
    save_freqdomain(pos_w_reg,'wilcox-reg-pos '+str(freqs[f1])+'-'+str(np.floor(freqs[f2])),folder)
    save_freqdomain(neg_w_reg,'wilcox-reg-neg'+str(freqs[f1])+'-'+str(np.floor(freqs[f2])),folder)
    save_freqdomain(pos_w_inv,'wilcox-inv-pos'+str(freqs[f1])+'-'+str(np.floor(freqs[f2])),folder)
    save_freqdomain(neg_w_inv,'wilcox-inv-neg '+str(freqs[f1])+'-'+str(np.floor(freqs[f2])),folder)