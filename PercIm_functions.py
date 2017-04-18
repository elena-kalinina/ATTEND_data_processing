# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:36:22 2015

@author: elena
"""

import nilearn
import numpy as np
import scipy
from scipy import stats
import nibabel
import sklearn as sck
import os
import functools
#import mvpa2
#from mvpa2 import *
from nilearn.masking import apply_mask
from nilearn.masking import compute_epi_mask
#from nipy.modalities.fmri import hrf
import pylab as pl
from nilearn.input_data import NiftiMasker
from sklearn import preprocessing
from sklearn import svm
from sklearn import lda
from sklearn.lda import LDA
from sklearn.svm import SVC
import random 
from spm_hrf import spm_hrf
import VS_functions
from VS_functions import *


#before you run it check the paths to the data and the masks !

def load_data_PercIm(pipeline, subj, mask):
    
 #   datapath=input("Please enter the path to the preprocessed data:")
 #   mask_path=input("Please enter the path to the MNI based mask files:")
 #   protocol_path=input("Please enter the path to the protocol files:")
 #   ExpType=input("Please enter experiment type you wish to analyze - Perc or Im:")
    filename_pattern=".nii.gz"
    
    datapath="/home/elena/ATTEND/validataset/data/PI"
    mask_path="/home/elena/ATTEND/MASKS"
    
    print (subj)       
    print ("Loading the data")
    print (mask)
            
       
            #MNI SPACE            
    subj_path=os.path.join(datapath, subj)
            
            #NATIVE SPACE            
    #subj_path=os.path.join(datapath, subjName, 'preprocessed_native')
            
    print (subj_path)             
#    path, dirs, files = os.walk(subj_path).next()
    n_run = 4 #int(len(dirs)) #int(len(files))
            #NATIVE SPACE
            #n_run = len(files)/2
            
            #MNI SPACE
    #n_run = int(len(dirs))#len(files)-2
    print (n_run)
            #because apart from run files, coregistration files are also saved in the preprocessed folder, might be subject to change
    masked_area=nibabel.load(os.path.join(mask_path, mask))
    current_mask=np.array((masked_area.get_data()>0)) #==1))
    
            #print current_mask.shape
    masked_data=[0]*n_run
    masked_data_scaled=[0]*n_run
    min_max_scaler = preprocessing.MinMaxScaler()
    for r in range (0, n_run):
                #NATIVE SPACE                
                #run_image = nibabel.load(os.path.join(subj_path, subj+'_native_run'+str(r+1)+filename_pattern))  
                #smoothed
                #run_image = nibabel.load(os.path.join(subj_path, subj+'_native_run'+str(r+1)+'_sps6'+filename_pattern))                  
          
                if pipeline=='fts':
                    run_image = nibabel.load(os.path.join(subj_path,'r0'+str(r+1), 'to_standard_fts', subj+'.PI.r0'+str(r+1)+'.tstost.fts' +filename_pattern))  
              #  else: 
                    #nibabel.load(os.path.join(subj_path, 'r0'+str(r+1), 'to_standard_fts', subj+'.PI''.r0'+str(r+1)+'.tstost.fts' +filename_pattern))  
                else: 
                    if pipeline=='stf':
                        run_image = nibabel.load(os.path.join(subj_path, 'r0'+str(r+1), 'to_standard_stf', 'PI.'+subj+'.r0'+str(r+1)+'.tost_stf' +filename_pattern)) 
                masked_data[r]=run_image.get_data()[current_mask]
                print (masked_data[r].shape)
                masked_data[r]=masked_data[r].T
                masked_data_scaled[r] = min_max_scaler.fit_transform(masked_data[r])
                print (len(masked_data_scaled[r]))
                print (masked_data_scaled[r].shape)
    
    return n_run, masked_data_scaled
    
    
def get_protocol_data_Perc(n_run, subjName, block_dur_invol, N_volumes_run, N_removed):
     protocol_path="/home/elena/ATTEND/PROTOCOLS/PI"   
     myhrf=spm_hrf(2)
     
     labels=[0]*n_run
     onsetsP=[0]*n_run
     onsetsC=[0]*n_run
     volume_num=[0]*n_run                     
     d_matrP=np.zeros([n_run, N_volumes_run])
     d_matrC=np.zeros([n_run, N_volumes_run])
            
     #d_matrP=np.zeros([n_run, N_volumes_run*TR])
     #d_matrC=np.zeros([n_run, N_volumes_run*TR])
             
     print ("Preparing the labels")           
     for r in range(0, n_run):
                print ('run', r)
                protocol = np.loadtxt(os.path.join(protocol_path, subjName+'-r'+str(r+1)+'.txt'))
                protocol[:, 0:2]=protocol[:, 0:2].astype(int)
                runlabel=[]
                
                onsets=[]
                onsetsPr=[]
                onsetsCr=[]
                            
                for j in range (0, int(len(protocol[:, 1]))):
                        
                        if protocol[j, 1]==0: #0 perception, 1 imagery
                            
                            runlabel.append(protocol[j, 2])
                            #onsets.append(protocol[j, 0])
                            onsets.append(int(protocol[j, 0]))
                labels[r]=runlabel    
                       
                for l in range (0, int(len(runlabel))):
                    
                        if (runlabel[l]==0) :
                            onsetsPr.append(onsets[l])
                        else:
                            onsetsCr.append(onsets[l])
                onsetsP[r]=onsetsPr        
                onsetsC[r]=onsetsCr
                for stim in range(0, int(len(onsetsP[r]))):
                            #d_matrP[r][onsetsP[r][stim]:onsetsP[r][stim]+block_dur_invol]=1
                            #d_matrC[r][onsetsC[r][stim]:onsetsC[r][stim]+block_dur_invol]=1
                            
                            d_matrP[r][onsetsP[r][stim]:onsetsP[r][stim]+int(block_dur_invol)]=1
                            d_matrC[r][onsetsC[r][stim]:onsetsC[r][stim]+int(block_dur_invol)]=1
               # print "Preparing the data"
                    
                    #P is for person, C is for car, could be also cat1 and cat2                    
                    
                hrf_conv_P=np.convolve(myhrf,d_matrP[r])
                    
                hrf_conv_C=np.convolve(myhrf,d_matrC[r])
                    
                nP = len(hrf_conv_P) - len(hrf_conv_P)%len(myhrf)
                # ignore tail
                slicesP = [hrf_conv_P[i:nP:int(len(myhrf))] for i in range(0, len(myhrf))]
                peak_maxP = functools.reduce(np.maximum, slicesP)
                peak_maxP=np.unique(peak_maxP)
                #print peak_maxP
                peak_maxP=functools.reduce(np.maximum, peak_maxP)
                #print peak_maxP                
                nC = len(hrf_conv_C) - len(hrf_conv_C)%len(myhrf)
                # ignore tail
                slicesC = [hrf_conv_C[i:nC:int(len(myhrf))] for i in range(0, len(myhrf))]
                peak_maxC = functools.reduce(np.maximum, slicesC)
                peak_maxC=np.unique(peak_maxC)
                    #print peak_maxC
                peak_maxC=functools.reduce(np.maximum, peak_maxC)
                    #print peak_maxC                    
                    #creating training and test data
                vols=[]
                    
                for ind in range (0, len(hrf_conv_P)):
                        if round(hrf_conv_P[ind], 3)==round(peak_maxP, 3): #maxs:
                            #vols.append(ind-5) 
                            vols.append(int(ind-N_removed))
                for ind in range (0, len(hrf_conv_C)):
                        if round(hrf_conv_C[ind], 3)==round(peak_maxC, 3): #maxs:
                            #vols.append(ind-5)        
                            vols.append(int(ind-N_removed))
                #    volume_num[r]=sorted(vols[::2])
                volume_num[r]=sorted(vols)           
     
     return labels, volume_num
     
     
def get_protocol_data_Im(n_run, subjName, TR, block_dur_invol, N_volumes_run, N_removed):
    protocol_path="/home/elena/ATTEND/PROTOCOLS/PI" 
    
    labels=[0]*n_run
    volume_num=[0]*n_run  
    rts=[0]*n_run 
    print ("Preparing the labels")
    myhrf=spm_hrf(2)
    for r in range(0, n_run):
        print ('run', r)
        first_press=[]
        vols=[]                
        runlabel=[]
        onsets=[]
        protocol = np.loadtxt(os.path.join(protocol_path, subjName+'-r'+str(r+1)+'.txt'))
        protocol[:, 0:2]=protocol[:, 0:2].astype(int)
        for j in range (0, len(protocol[:, 1])):
                        
                if protocol[j, 1]==1: #0 perception, 1 imagery
                            
                            runlabel.append(protocol[j, 2])
                            onsets.append(protocol[j, 0])
                            
                            if protocol[j, 4]!=0:
                                first_press.append(protocol[j, 4])
                             #   onsets_train.append(protocol[j, 4])
                            else:
                               # if protocol[j, 4]!=0:
                                #    first_press.append(protocol[j, 4])
                               #     onsets_train.append(protocol[j, 3])
                                #else:
                                first_press.append(protocol[j, 3])
        labels[r]=runlabel
      #  print len(labels[r])
     #   print first_press
        rts[r]=first_press
                    
        d_matr_trial=np.zeros([block_dur_invol])
                    # 1. get the press, calculate hrf
        for pr in range (0, len(first_press)):
                    #print first_press[pr]
                    #print onsets[pr]*TR
                        rt=first_press[pr]-onsets[pr]*TR
                        #rt=first_press[pr]-onsets_train[pr]
                        
                        
                        #print 'doing a block'
                        #print peak_max
                        #4. find the volume num corresponding to the max
                        #maxs:
                        ############################
                        #vols1.append(onsets[pr]+delay-5)
                        #############################
                 #       d_matr_trial[0:TR]=1
                        if rt<TR:
                            #d_matr_trial[TR:TR+TR]=1
                            d_matr_trial[0:(TR)]=1
                        else:
                            #d_matr_trial[TR:int(rt)]=1
                            d_matr_trial[0:int(rt/TR)]=1
                            
                            #print d_matr_trial
                            #plt.plot(d_matr_trial)
                            #plt.show()
                            #3. convolve with the hrf, get the maximum
                        trial_hrf=np.convolve(myhrf,d_matr_trial)
                        #plt.plot(trial_hrf)
                        #plt.show()
                        peak_max = functools.reduce(np.maximum, trial_hrf)
                        #print peak_max
                        peak_max=np.unique(peak_max)
                        #print peak_max
                        peak_max=functools.reduce(np.maximum, peak_max)
                        #print peak_max
                        #4. find the volume num corresponding to the max
                        for ind in range (0, int(len(trial_hrf))):
                            if trial_hrf[ind]==peak_max: #maxs:
                                vols.append(int(onsets[pr]+ind-N_removed))
                              #  vols_train.append(int(onsets_train[pr]/TR)+int(ind/TR)-5)
        volume_num[r]=vols
      #  print volume_num[r]
    return labels, rts, volume_num
    #not to make crash decoding functions
    #return labels, volume_num

def calculate_subj_maps_pi(ExpType, n_run, data, volumes, for_dm, labels, stim_post, N_volumes_run1, N_removed, TR, hrf_lag):
        categories=np.unique(np.reshape(np.array(labels), -2))
        print (categories)
        subj_map=[0]*len(categories)
        #    print categories
        cat_indices=[0]*len(categories)
            #TODO add here if one is odd take the smaller number for both to account for 39 examples
        cat_indices[1]=np.count_nonzero(np.reshape(np.array(labels), -2)) #labels
        
        cat_indices[0]=cat_indices[1] #labels_vs.count(0)
        
     #   print cat_indices[0]
     #   print cat_indices[1]
        for cat in categories:
            #    print cat
                counter=0
                beta_maps=[0]*cat_indices[int(cat)]
                if ExpType=="Perc":
                        dm_param=for_dm
                for r in range(0, n_run):
                    indices=np.array(np.where(labels[r]==cat))
            #        indices=indices.tolist()
              #      print indices
     #               volumes_pi_run=volumes_pi[r][indices]
                    
                    
                    
                    
#                        rts_pi_run=rts_pi[r][indices]
                    for vol in indices[0]:
                   #     print vol
                        if ExpType=="Im":
    
                            #print rts_pi[r][vol]-(volumes_pi[r][vol]*TR)/TR
                 #           print r
                 #           print vol
                            dm_param=int((for_dm[r][vol]-volumes[r][vol]*TR)/TR)
                            if dm_param<1:
                                dm_param=8
                   #     print counter
                        beta_maps[counter]=calculate_beta_maps(data[r], volumes[r][vol], hrf_lag,stim_post, dm_param, N_volumes_run1)
                   #     print len(beta_maps[counter])
                        counter=counter+1
           
                subj_map[int(cat)]=np.mean(beta_maps, axis=0)
         #       print subj_maps_pi[int(cat)][int(subj)]
      #  subj_map=np.reshape(np.asarray(subj_map), 2)
             #   print subj_map[int(cat)].shape
                subj_map=np.asarray(subj_map)
     #   print subj_map.shape     
        return subj_map   
     
    
def load_data_PercIm_whb(pipeline, subj, newmask):
    
 #   datapath=input("Please enter the path to the preprocessed data:")
 #   mask_path=input("Please enter the path to the MNI based mask files:")
 #   protocol_path=input("Please enter the path to the protocol files:")
 #   ExpType=input("Please enter experiment type you wish to analyze - Perc or Im:")
    filename_pattern=".nii.gz"
    
    datapath="/home/elena/ATTEND/validataset/data/PI"
    mask_path="/home/elena/ATTEND/MASKS"
    
    print (subj)       
    print ("Loading the data")
  #  print mask
            
       
            #MNI SPACE            
    subj_path=os.path.join(datapath, subj)
            
            #NATIVE SPACE            
    #subj_path=os.path.join(datapath, subjName, 'preprocessed_native')
            
    print (subj_path)             
   # path, dirs, files = os.walk(subj_path).next()
    n_run = 4 #int(len(dirs)) #int(len(files))
            #NATIVE SPACE
            #n_run = len(files)/2
            
            #MNI SPACE
    #n_run = int(len(dirs))#len(files)-2
    print (n_run)
            #because apart from run files, coregistration files are also saved in the preprocessed folder, might be subject to change
    
    
    nifti_masker = NiftiMasker(mask_img=newmask, detrend=True, standardize=True) #, memory_level=3, memory="/home/elena/ATTEND/validataset/TEMP/")
    
    
    
    
            #print current_mask.shape
    masked_data=[0]*n_run
    masked_data_scaled=[0]*n_run
    min_max_scaler = preprocessing.MinMaxScaler()
    for r in range (0, n_run):
                #NATIVE SPACE                
                #run_image = nibabel.load(os.path.join(subj_path, subj+'_native_run'+str(r+1)+filename_pattern))  
                #smoothed
                #run_image = nibabel.load(os.path.join(subj_path, subj+'_native_run'+str(r+1)+'_sps6'+filename_pattern))                  
          
                if pipeline=='fts':
                    run_image = nibabel.load(os.path.join(subj_path,'r0'+str(r+1), 'to_standard_fts', subj+'.PI.r0'+str(r+1)+'.tstost.fts' +filename_pattern))  
              #  else: 
                    #nibabel.load(os.path.join(subj_path, 'r0'+str(r+1), 'to_standard_fts', subj+'.PI''.r0'+str(r+1)+'.tstost.fts' +filename_pattern))  
                else: 
                    if pipeline=='stf':
                        run_image = nibabel.load(os.path.join(subj_path, 'r0'+str(r+1), 'to_standard_stf', 'PI.'+subj+'.r0'+str(r+1)+'.tost_stf' +filename_pattern)) 
                nifti_masker.fit(run_image)    
                masked_data[r] = nifti_masker.transform(run_image)
                masked_data_scaled[r] = min_max_scaler.fit_transform(masked_data[r])
                print (len(masked_data_scaled[r]))
                print (masked_data_scaled[r].shape)
    
    return n_run, masked_data_scaled, nifti_masker
    
    
    
def get_protocol_data_Im_ons(n_run, subjName, TR, block_dur_invol, N_volumes_run, N_removed):
    protocol_path="/home/elena/ATTEND/PROTOCOLS/PI" 
  #  dm_param_Im=[0]*n_run
    labels=[0]*n_run
    volume_num=[0]*n_run  
    rts=[0]*n_run 
    print ("Preparing the labels")
  #  myhrf=spm_hrf(2)
    for r in range(0, n_run):
        print ('run', r)
        first_press=[]
    #    vols=[]                
        runlabel=[]
        onsets=[]
        protocol = np.loadtxt(os.path.join(protocol_path, subjName+'-r'+str(r+1)+'.txt'))
        protocol[:, 0:2]=protocol[:, 0:2].astype(int)
        for j in range (0, len(protocol[:, 1])):
                        
                if protocol[j, 1]==1: #0 perception, 1 imagery
                            
                            runlabel.append(protocol[j, 2])
                            onsets.append(int(protocol[j, 0])-N_removed)
                            
                            if protocol[j, 4]!=0:
                                first_press.append(protocol[j, 4])
                             #   onsets_train.append(protocol[j, 4])
                            else:
                               # if protocol[j, 4]!=0:
                                #    first_press.append(protocol[j, 4])
                               #     onsets_train.append(protocol[j, 3])
                                #else:
                               first_press.append(protocol[j, 3])
        labels[r]=runlabel
      #  print len(labels[r])
     #   print first_press
      #  rts[r]=first_press
        
   #     d_matr_trial=np.zeros([block_dur_invol])
                    # 1. get the press, calculate hrf
        rts[r]=[]
        for pr in range (0, len(first_press)):
                    #print first_press[pr]
                    #print onsets[pr]*TR
                        rt=first_press[pr]-(onsets[pr]+N_removed)*TR
                        #rt=first_press[pr]-onsets_train[pr]
                        if rt<TR:
                            rts[r].append(8)
                        else:
                            rts[r].append(int(rt/TR))
        print ("We are in the right place Im")                
        volume_num[r]=onsets
        
      #  print volume_num[r]
    return labels, rts, volume_num
    
def get_protocol_data_Perc_ons(n_run, subjName, block_dur_invol, N_volumes_run, N_removed):
     protocol_path="/home/elena/ATTEND/PROTOCOLS/PI"   
   #  myhrf=spm_hrf(2)
     
     labels=[0]*n_run
#     onsetsP=[0]*n_run
#     onsetsC=[0]*n_run
     volume_num=[0]*n_run                     
#     d_matrP=np.zeros([n_run, N_volumes_run])
#     d_matrC=np.zeros([n_run, N_volumes_run])
            
     #d_matrP=np.zeros([n_run, N_volumes_run*TR])
     #d_matrC=np.zeros([n_run, N_volumes_run*TR])
             
     print ("Preparing the labels")           
     for r in range(0, n_run):
                print ('run', r)
                protocol = np.loadtxt(os.path.join(protocol_path, subjName+'-r'+str(r+1)+'.txt'))
                protocol[:, 0:2]=protocol[:, 0:2].astype(int)
                runlabel=[]
                
                onsets=[]
#                onsetsPr=[]
#                onsetsCr=[]
                            
                for j in range (0, int(len(protocol[:, 1]))):
                        
                        if protocol[j, 1]==0: #0 perception, 1 imagery
                            
                            runlabel.append(protocol[j, 2])
                            #onsets.append(protocol[j, 0])
                            onsets.append(int(protocol[j, 0]-N_removed))
                print ("We are in the right place Perc")
                labels[r]=runlabel    
                volume_num[r]=onsets
#       
     
     return labels, volume_num