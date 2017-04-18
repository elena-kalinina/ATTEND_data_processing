# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:30:55 2016
This is the one I used to create ultimate maps !!!
@author: elenka
"""

import pdb
import nilearn
import numpy as np
import scipy
from scipy import stats
import nibabel as nib
import sklearn as sck
import os
import matplotlib as plt
import pylab as pl
from nilearn.input_data import NiftiMasker
from sklearn import preprocessing
from sklearn import svm
from sklearn import lda
from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn import linear_model
import random 
import spm_hrf
from spm_hrf import spm_hrf
import VS_functions
from VS_functions import *
import PercIm_functions
from PercIm_functions import *
import nilearn.datasets


def load_and_save_vol_SL(ExpType, pipeline, block_dur_invol, delay, ncond, stim_post, N_volumes_run2, N_volumes_run1, N_removed, TR, *subjID):
    
    gm_mask = nib.load('/home/elena/ATTEND/MASKS/mygreymask.hdr')
    #nilearn.datasets.fetch_icbm152_brain_gm_mask(data_dir='/home/elena/ATTEND/cross_modal/', threshold=0.2, resume=True, verbose=1)
    print (gm_mask.shape)
    
    filename_pattern=".nii.gz"
    cwd="/home/elena/ATTEND/cross_modal/code/"
    os.chdir(cwd)
    if ExpType=="VS":
        datapath="/home/elena/ATTEND/validataset/data/VS"
    else:
        datapath="/home/elena/ATTEND/validataset/data/PI"
    n_run=4
    n_of_subj=int(len(subjID))
    for subj in range(0, n_of_subj):
            subjName=subjID[subj][-4:]
           # subj_path=os.path.join(datapath, subjName)
            results_path=("/home/elena/ATTEND/cross_modal/results/whole_brain_VS_delay_2/%s_%s" %(subjName, ExpType))
          
            if os.path.isdir(results_path)==False:
                os.mkdir(results_path)
            
             
            if ExpType=="Perc":
                n_run, data1, masker=load_data_PercIm_whb(pipeline, subjName, gm_mask)
                labels, volumes=get_protocol_data_Perc_ons(n_run, subjName, block_dur_invol, N_volumes_run1, N_removed)
                
                dm_param=block_dur_invol
                
                for r in range(0, n_run):
               
                    for vol in range(0, len(volumes[r])):
                        
                 
                        beta_map=calculate_beta_maps(data1[r], volumes[r][vol], 4,stim_post, dm_param, N_volumes_run1)
                     #   print beta_map.shape
                        niimg = masker.inverse_transform(beta_map.T)
                        
                        betafile=subjName+'_'+str(r+1)+'_'+str(volumes[r][vol])+'_'+str(labels[r][vol])+'.nii'
                        nib.save(niimg, os.path.join(results_path, betafile)) 
             #   print for_dm
                #print volumes[1]
            else:
                if ExpType=="Im":
                    n_run, data1, masker=load_data_PercIm_whb(pipeline, subjName, gm_mask)
                    labels, for_dm, volumes=get_protocol_data_Im(n_run, subjName, TR, block_dur_invol, N_volumes_run1, N_removed)
                 #   for_dm=rts_pi
                                
                    for r in range(0, n_run):
               
                        for vol in range(0, len(volumes[r])):
                            
                            dm_param=int((for_dm[r][vol]-volumes[r][vol]*TR)/TR)
                            if dm_param<1:
                                dm_param=8

                            '''
                            BIG BUG !!!!!!
                            '''
                            beta_map=calculate_beta_maps(data1[r], volumes[r][vol], 4,stim_post, dm_param, N_volumes_run1) #int(rts_pi[r][vol])
                            niimg = masker.inverse_transform(beta_map.T)
                            
                            betafile=subjName+'_'+str(r+1)+'_'+str(volumes[r][vol])+'_'+str(labels[r][vol])+'.nii'
                            nib.save(niimg, os.path.join(results_path, betafile)) 
                    
                    
                else:
                    if ExpType=="VS":
                      #  delay=8
                        n_run, data1, masker=load_data_VS_whb(pipeline, subjName, gm_mask)
                     #   labels, volumes=get_protocol_data_VS_delay(n_run, delay, block_dur_invol, subjName, N_volumes_run2, N_removed)
                        if delay==None:
                            labels, delays, volumes=get_protocol_data_VS_full_spec(n_run, block_dur_invol, subjName, N_volumes_run2, N_removed)
                        else:
                            labels, volumes=get_protocol_data_VS_delay(n_run, delay, block_dur_invol, subjName, N_volumes_run2, N_removed)
                            dm_param=delay 
                        for r in range(0, n_run):
                        
                            for vol in range (0, len(volumes[r])):
                                if delay==None:
                                    dm_param=delays[r][vol]+1
#                            print vol
                                
                                    #delays[r][vol]+1
                             #   print "delay=", dm_param
                                beta_map=calculate_beta_maps(data1[r], volumes[r][vol], 3, stim_post, dm_param, N_volumes_run2)
                                niimg = masker.inverse_transform(beta_map.T)
                                betafile=subjName+'_'+str(r+1)+'_'+str(volumes[r][vol])+'_'+str(labels[r][vol])+'.nii'
                                nib.save(niimg, os.path.join(results_path, betafile)) 
                    
                   
                   
  #  os.chdir(results_path)
  #  nib.save(mymask, maskname)    
    
    return 


def save_VS_delays(pipeline, block_dur_invol, delay, ncond, stim_post, N_volumes_run2, N_volumes_run1, N_removed, *subjID):  


    masks= ['left.OSC.625.nii.gz', 'right.OSC.bin.nii.gz', 'OSC.625.nii.gz']
    masknames= ['l_OSC', 'r_OSC', 'l+r_OSC']
    results_path=("/home/elena/ATTEND/cross_modal/results/BOLD_VS_delays_%s" %pipeline)
    n_of_subj=int(len(subjID))
    if os.path.isdir(results_path)==False:
        os.mkdir(results_path)
    cwd="/home/elenka/ATTEND/cross_modal/code/"
    os.chdir(cwd)
    
    for masknum in range(0, int(len(masks))):
        mask=masks[masknum]
       
        runs=4
        for d in range(0, int(delay)):
            print ("delay=", d)
            data_all=[0]*n_of_subj
        
            subj_maps_all=[0]*n_of_subj
           
            labels_all=[0]*n_of_subj
            
            for subj in range(0, n_of_subj):
                subjName=subjID[subj][-4:]
                subj_data=[]
                subj_labels=[]
            
                n_run, data1=load_data_VS(pipeline, subjName, mask)
                labels, delays, volumes=get_protocol_data_VS_delay(n_run, d, block_dur_invol, subjName, N_volumes_run2, N_removed)
              #  print len(labels), len(volumes)
                #    print labels
                        
                for r in range(0, runs):
                    subj_labels.append(labels[r])
                    for vol in volumes[r]:
                            subj_data.append(data1[r][vol])
                #    print data1[r][vol][1:5]
             
            
                subj_maps=calc_subj_maps_vs_nonav(n_run, data1, delays, labels, volumes, stim_post, N_volumes_run2, N_removed, 3) #hrf_lag - try 4 if not
       #     print len(subj_map1)
       #     print len(subj_map2)
       #     print subj_map1.shape
       #     print subj_map2.shape
                        
                subj_maps_all[subj]=subj_maps       
                labels_all[subj]=subj_labels
            
                data_all[subj]=subj_data
          #  data1=None
            os.chdir(results_path)
            filename_data_BOLD=("VS_delay_%d_%s_BOLD" %(d, masknames[masknum]))
            filename_data_maps=("VS_delay_%d_%s_betamaps" %(d, masknames[masknum]))
       
            np.save(filename_data_BOLD, data_all)
            np.save(filename_data_maps, subj_maps_all)
    
    
            filename_labels=("VS_delay_%d_labels" %d)
            
            np.save(filename_labels, labels_all)


    return

   
def calculate_beta_maps(data, onset, hrf_lag, stim_post, dm_param, N_volumes_run):
  #  print len(data)
  #  print onset
    clf_lin = linear_model.LinearRegression(copy_X=True, fit_intercept=True, normalize=False)    
    dm_trial=np.zeros(14)   
   # print dm_param
    dm_trial[0:dm_param]=1
   
    trial_hrf=np.convolve(spm_hrf(2),dm_trial)
    trial_hrf=trial_hrf[0:hrf_lag+stim_post]
                 #           print len(trial_hrf)
    trial_hrf=trial_hrf.reshape(len(trial_hrf), 1)
                         #   print trial_hrf
    
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
    
    coef = clf_lin.coef_
    
    #betas=np.asarray(clf_lin.coef_)
    #betas=np.reshape(betas, -1) 
                      #          print betas.shape
    
    return coef

if __name__ == "__main__":
    #save_VS_delays('fts', 8, 5, 2, 6, 188, 170, 5, "19890126ANPS", "19901103GBTE", "19900422ADDL", "19850630IAAD", "19881016MCBL", "19851030DNGL", "19750827RNPL", "19830905RBMS", "19861104GGBR")
    #load_and_save_vol_SL('Perc', 'fts', 4, 8, 2, 6, 188, 170, 5, 2, "19881016MCBL", "19890126ANPS", "19901103GBTE", "19900422ADDL", "19850630IAAD", "19851030DNGL", "19750827RNPL", "19830905RBMS", "19861104GGBR")
    #load_and_save_vol_SL('Im', 'fts', 8, 4, 2, 6, 188, 170, 5, 2, "19890126ANPS", "19901103GBTE", "19900422ADDL", "19850630IAAD", "19881016MCBL", "19851030DNGL", "19750827RNPL", "19830905RBMS", "19861104GGBR")              
    load_and_save_vol_SL('VS', 'fts', 8, 0, 2, 6, 188, 170, 5, 2, "19750827RNPL", "19830905RBMS", "19861104GGBR", "19881016MCBL", "19900422ADDL", "19850630IAAD", "19890126ANPS", "19901103GBTE", "19851030DNGL") 
    print ("Done with that !!")


              
