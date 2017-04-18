# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:38:54 2015

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
from sklearn import linear_model
import random 
from spm_hrf import spm_hrf


#before you run it check the paths to the data and the masks !

def load_data_VS(pipeline, subj, mask):
    
 #   datapath=input("Please enter the path to the preprocessed data:")
 #   mask_path=input("Please enter the path to the MNI based mask files:")
 #   protocol_path=input("Please enter the path to the protocol files:")
 #   ExpType=input("Please enter experiment type you wish to analyze - Perc or Im:")
    filename_pattern=".nii.gz"
    datapath="/home/elena/ATTEND/validataset/data/VS/"
    mask_path="/home/elena/ATTEND/MASKS/"
    
    
    print (subj)       
    print ("Loading the data")
    print (mask)
            
       
            #MNI SPACE            
    subj_path=os.path.join(datapath, subj)
            
            #NATIVE SPACE            
    #subj_path=os.path.join(datapath, subjName, 'preprocessed_native')
            
    print (subj_path)             
#    path, dirs, files = os.walk(subj_path).next()
            
            #NATIVE SPACE
            #n_run = len(files)/2
            
            #MNI SPACE
    n_run = 4 #int(len(dirs)) #int(len(files))#len(files)-2
    print (n_run)
            #because apart from run files, coregistration files are also saved in the preprocessed folder, might be subject to change
    masked_area=nibabel.load(os.path.join(mask_path, mask))
    current_mask=np.array((masked_area.get_data()==1))
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
                    run_image = nibabel.load(os.path.join(subj_path, 'r0'+str(r+1), 'to_standard_fts',subj+'.VS''.r0'+str(r+1)+'.tstost.fts' +filename_pattern))  
                else: 
                    if pipeline=='stf':
                        run_image = nibabel.load(os.path.join(subj_path, 'r0'+str(r+1), 'to_standard_stf', 'VS.'+subj+'.r0'+str(r+1)+'.tost_stf' +filename_pattern)) 
                masked_data[r]=run_image.get_data()[current_mask]
                masked_data[r]=masked_data[r].T
                masked_data_scaled[r] = min_max_scaler.fit_transform(masked_data[r])
                print (len(masked_data_scaled[r]))
                print (masked_data_scaled[r].shape)
    
    return n_run, masked_data_scaled
    
#TOWORK ON IT!!!!!!!!!!!
def get_protocol_data_VS(n_run, delay, subjName, N_volumes_run, N_removed):
     protocol_path="/home/elena/ATTEND/PROTOCOLS/VS"   
     
     labels_test=[0]*n_run
     delays=[0]*n_run
     volume_num_test=[0]*n_run
     for r in range(0, n_run):
        protocol = np.loadtxt(os.path.join(protocol_path, subjName+'-r'+str(r+1)+'.txt'))
        protocol[:, 0:2]=protocol[:, 0:2].astype(int)
        runlabel_test=[]
        delay_run=[]
        vols_test=[]
        if delay ==0:
            for j in range (0, int(len(protocol[:, 2]))):
            
                runlabel_test.append(protocol[j, 1])
                delay_run.append(protocol[j, 2])
                vols_test.append(int(protocol[j, 0]-N_removed))
              #  vols_test.append(protocol[j, 0]+d+hrf_lag-N_removed)
            labels_test[r]=runlabel_test
            delays[r]=delay_run
            volume_num_test[r]=vols_test   

        else:     
                                    
            for j in range (0, int(len(protocol[:, 2]))):
                if protocol_test[j, 2]==delay:
                    runlabel_test.append(protocol[j, 1])
                            #vols_test.append(protocol_test[j, 0]+d+hrf_lag-N_removed)
                    vols_test.append(int(protocol[j, 0]-N_removed))
            labels_test[r]=runlabel_test
                    #print labels_test[r]
            volume_num_test[r]=vols_test    
             
     
     return labels_test, delays, volume_num_test
     
def get_protocol_data_VS_full_spec(n_run, block_dur_invol, subjName, N_volumes_run, N_removed):
    
    protocol_path="/home/elena/ATTEND/PROTOCOLS/VS"   
     
    labels_test=[0]*n_run
    delays=[0]*n_run
    volume_num_test=[0]*n_run
    for r in range(0, n_run):
        protocol = np.loadtxt(os.path.join(protocol_path, subjName+'-r'+str(r+1)+'.txt'))
        protocol[:, 0:2]=protocol[:, 0:2].astype(int)
        runlabel_test=[]
        delay_run=[]
        vols_test=[]
     
            
        for j in range (0, len(protocol[:, 2])):
                                d=protocol[j, 2]
                                delay_run.append(d)

                                runlabel_test.append(protocol[j, 1])
                                
                                dm_trial=np.zeros([block_dur_invol])
     
                                dm_trial[0:int(d+1)]=1
                                trial_hrf=np.convolve(spm_hrf(2),dm_trial)
                                peak_max = functools.reduce(np.maximum, trial_hrf)
                                for ind in range (0, len(trial_hrf)):
                                    if trial_hrf[ind]==peak_max: 
                                        hrf_lag=int(ind) 
                            #vols_test.append(protocol_test[j, 0]+d+hrf_lag-N_removed)
                                vols_test.append(int(protocol[j, 0]+hrf_lag-N_removed))
        labels_test[r]=runlabel_test
        delays[r]=delay_run
        volume_num_test[r]=vols_test


    return labels_test, delays, volume_num_test





def get_protocol_data_VS_delay(n_run, delay, block_dur_invol, subjName, N_volumes_run, N_removed):
     protocol_path="/home/elena/ATTEND/PROTOCOLS/VS"   
     
     labels_test=[0]*n_run
  #   delays=[0]*n_run
     volume_num_test=[0]*n_run
     for r in range(0, n_run):
        protocol = np.loadtxt(os.path.join(protocol_path, subjName+'-r'+str(r+1)+'.txt'))
        protocol[:, 0:2]=protocol[:, 0:2].astype(int)
        
#        dm_trial=np.zeros([block_dur_invol])
#     
#        dm_trial[0:int(delay+1)]=1
#        trial_hrf=np.convolve(spm_hrf(2),dm_trial)
#        peak_max = reduce(np.maximum, trial_hrf)
#        for ind in range (0, len(trial_hrf)):
#            if trial_hrf[ind]==peak_max: 
#                hrf_lag=int(ind) 
        runlabel_test=[]
     #   delay_run=[]
        vols_test=[]
        
                                    
        for j in range (0, int(len(protocol[:, 2]))):
                
                if protocol[j, 2]==delay:
                    runlabel_test.append(protocol[j, 1])
                            #vols_test.append(protocol_test[j, 0]+d+hrf_lag-N_removed)
               #     delay_run.append(protocol[j, 2])
                            #vols_test.append(protocol_test[j, 0]+d+hrf_lag-N_removed)
                    vols_test.append(int(protocol[j, 0])-N_removed)
        print ("We are in the right place VS delay") 
        labels_test[r]=runlabel_test
                    #(int(protocol[j, 0]+hrf_lag-N_removed))
        volume_num_test[r]=vols_test    
      #  print len(volume_num_test[r])
    #    delays[r]=delay_run
       # print len(delays[r])
     return  labels_test, volume_num_test   #labels_test, delays, volume_num_test    
     


def get_protocol_data_VS_full(n_run, block_dur_invol, subjName, N_volumes_run, N_removed):
    
    protocol_path="/home/elena/ATTEND/PROTOCOLS/VS"   
    
    labels_test=[0]*n_run
    delays=[0]*n_run
    volume_num_test=[0]*n_run
    for r in range(0, n_run):
        protocol = np.loadtxt(os.path.join(protocol_path, subjName+'-r'+str(r+1)+'.txt'))
        protocol[:, 0:2]=protocol[:, 0:2].astype(int)
        runlabel_test=[]
        delay_run=[]
        vols_test=[]
     
        
        for j in range (0, len(protocol[:, 2])):
            
            d=protocol[j, 2]
            delay_run.append(d)
            runlabel_test.append(protocol[j, 1])
            vols_test.append(int(protocol[j, 0])-N_removed)
                                #If we want to do delays 6-10
#                                if d > 1:
#                                    delay_run.append(d)
#
#                                    runlabel_test.append(protocol[j, 1])
#                  
#                                    vols_test.append(int(protocol[j, 0]-N_removed))
        print ("We are in the right place VS") 
        labels_test[r]=runlabel_test
        delays[r]=delay_run
        volume_num_test[r]=vols_test


    return labels_test, delays, volume_num_test

def calculate_beta_maps(data, onset, hrf_lag, stim_post, dm_param, N_volumes_run):
    clf_lin = linear_model.LinearRegression(copy_X=True, fit_intercept=True, normalize=False)    
    dm_trial=np.zeros(14)    
    dm_trial[0:dm_param]=1
              #      dm_trial[0:TR*(k+1)]=1
    trial_hrf=np.convolve(spm_hrf(2),dm_trial)
    trial_hrf=trial_hrf[0:hrf_lag+stim_post]
                 #           print len(trial_hrf)
    trial_hrf=trial_hrf.reshape(len(trial_hrf), 1)
                         #   print trial_hrf
    
                            #      mytrial=masked_data_scaled[r][vol:vol+block_dur_invol+stim_post]
    if (onset+hrf_lag+stim_post)<=(N_volumes_run):
        mytrial=data[onset:onset+hrf_lag+stim_post]
    else:
                     #               print vol+hrf_lag+stim_post
                     #               print len(mytrial)
        n_vol_to_fill=(onset+hrf_lag+stim_post)-(N_volumes_run)
                      #              print n_vol_to_fill   
        mytrial=data[onset:N_volumes_run]
        trial_mean=np.mean(mytrial, axis=0)
                      #              print trial_mean.shape
                                    
        for fv in range (0, int(n_vol_to_fill)):
            mytrial=np.vstack([mytrial, trial_mean])
                         #       print mytrial.shape
                                    
                                        
                            #    print mytrial.shape
                            #    print trial_hrf.shape
    clf_lin.fit(trial_hrf, mytrial)
    betas=np.asarray(clf_lin.coef_)
    betas=np.reshape(betas, -1) 
                      #          print betas.shape
    
    return betas


def calculate_test_statistics(data1, data2, n_of_subj):  
    
    subjs_corr=[0]*n_of_subj
    for subj in range(0, n_of_subj):
        subj=int(subj)
     #   print data1[0][subj].shape
     #   print data2[0][subj].shape
        
      #  print data1[1][subj].shape
      #  print data2[1][subj].shape
        cor1=scipy.stats.pearsonr(data1[0][subj], data2[0][subj])
        cor2=scipy.stats.pearsonr(data1[1][subj], data2[1][subj])
        
        cor3=scipy.stats.pearsonr(data1[0][subj], data2[1][subj])
        cor4=scipy.stats.pearsonr(data1[1][subj], data2[0][subj])
        subjs_corr[subj]=cor1[0]+cor2[0]-cor3[0]-cor4[0]
  #  print subjs_corr
    
    
  #  print subj_corr_val
    return subjs_corr
     
     
     
     
def calculate_subj_maps_vs(n_run, data, delays, labels, volumes, stim_post, N_volumes_run2, N_removed, hrf_lag):
        
        
        
              #Get beta maps for VS            
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
              #  print cat
                counter=0
                beta_maps=[0]*cat_indices[int(cat)]
                
                    
                for r in range(0, n_run):
                    
              #      print r
                    
                    indices=np.where(labels[r]==cat)
                 #   volumes_vs=volumes_vs[indices]
                  #  delays=delays[indices]
                #    print indices
                    
                    for vol in indices[0]:
                #        print vol
                #        print len(delays[r])
                        if vol>=len(delays[r]):
                            continue
                        
                       #     print vol
                        dm_param=delays[r][vol]+1
                        
                        beta_maps[counter]=calculate_beta_maps(data[r], volumes[r][vol], hrf_lag, stim_post, dm_param, N_volumes_run2)
                      #  print len(beta_maps[counter])
                subj_map[int(cat)]=np.mean(beta_maps, axis=0)
             #   print subj_map[int(cat)].shape
    #    subj_map=np.reshape(np.asarray(subj_map), -1)  
        
                subj_map=np.asarray(subj_map) 
     #   print subj_map.shape   
        return subj_map        


def load_data_VS_whb(pipeline, subj, newmask):
    
 #   datapath=input("Please enter the path to the preprocessed data:")
 #   mask_path=input("Please enter the path to the MNI based mask files:")
 #   protocol_path=input("Please enter the path to the protocol files:")
 #   ExpType=input("Please enter experiment type you wish to analyze - Perc or Im:")
    filename_pattern=".nii.gz"
    datapath="/home/elena/ATTEND/validataset/data/VS/"
    mask_path="/home/elena/ATTEND/MASKS/"
    
    
    print (subj)       
    print ("Loading the data")
   # print mask
            
       
            #MNI SPACE            
    subj_path=os.path.join(datapath, subj)
            
            #NATIVE SPACE            
    #subj_path=os.path.join(datapath, subjName, 'preprocessed_native')
            
    print (subj_path)             
#    path, dirs, files = os.walk(subj_path).next()
            
            #NATIVE SPACE
            #n_run = len(files)/2
            
            #MNI SPACE
    n_run = 4 #int(len(dirs)) #int(len(files))#len(files)-2
    print (n_run)
            #because apart from run files, coregistration files are also saved in the preprocessed folder, might be subject to change
    
    nifti_masker = NiftiMasker(mask_img=newmask, detrend=True, standardize=True) #, memory_level=1, memory="/home/elena/ATTEND/validataset/TEMP/")
    
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
                    run_image = nibabel.load(os.path.join(subj_path, 'r0'+str(r+1), 'to_standard_fts',subj+'.VS''.r0'+str(r+1)+'.tstost.fts' +filename_pattern))  
                else: 
                    if pipeline=='stf':
                        run_image = nibabel.load(os.path.join(subj_path, 'r0'+str(r+1), 'to_standard_stf', 'VS.'+subj+'.r0'+str(r+1)+'.tost_stf' +filename_pattern)) 
                nifti_masker.fit(run_image)    
                masked_data[r] = nifti_masker.transform(run_image)
                masked_data_scaled[r] = min_max_scaler.fit_transform(masked_data[r])
                print (len(masked_data_scaled[r]))
                print (masked_data_scaled[r].shape)
    
    return n_run, masked_data_scaled, nifti_masker           