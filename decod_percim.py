# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:24:14 2015

@author: elena
"""

# In this file, there are all perception imagery classifiers:
#same modality, within subject
#same modality between subject
#across modality, within subject
#across modality between subject

import PercIm_functions
import PercIm_classification
from PercIm_functions import *
from PercIm_classification import *
from sklearn import linear_model
from sklearn.linear_model import LogisticRegressionCV
from save_BOLD import calculate_beta_maps

#same modality between subject
def samemod_percim_across_subj(ExpType, pipeline, *subjID):

    masks=['left.OSC.625.nii.gz', 'right.OSC.bin.nii.gz', 'OSC.625.nii.gz']
    masknames=['l_OSC', 'r_OSC', 'l+r_OSC']
    results_path=("/home/elena/ATTEND/results/samemod_acr_subj_%s_%s" %(ExpType, pipeline))
    n_of_subj=len(subjID)
    if os.path.isdir(results_path)==False:
        os.mkdir(results_path)
    cwd="/home/elena/FMRI_decoding/decoding_scripts"
    os.chdir(cwd)
    mask_accuracies=[0]*len(masknames)
    mask_pvals=[0]*len(masknames)
    for masknum in range(0, int(len(masks))):
        mask=masks[masknum]
        all_subj_data=[0]*len(subjID)
   # all_subj_vols=[0]*len(subjID)
        subject_labels=[0]*len(subjID)
        for subj in range(0, int(len(subjID))):
            subjName=subjID[subj][-4:]
  
            n_run, data=load_data_PercIm(pipeline, subjName, mask)
            if ExpType=="Perc":
                labels, volumes=get_protocol_data_Perc(n_run, subjName, block_dur_invol=8, N_volumes_run=170, N_removed=5)
            else:
                if ExpType=="Im":
                    labels, volumes=get_protocol_data_Im(n_run, subjName, TR=2, block_dur_invol=8, N_volumes_run=170, N_removed=5)
            subject_data=[]
            for r in range(0, n_run):
                
                for vol in volumes[r]:
                    subject_data.append(data[r][vol])
                    subject_labels[subj]=np.reshape(np.array(labels), -2)
            all_subj_data[subj]=subject_data
        
        acc, training_data, training_labels, test_data, test_labels=leave_one_subj_out_clsf(n_of_subj, all_subj_data, subject_labels)
        mask_accuracies[masknum]=acc
        mask_pvals[masknum]=permutation_test_acr_subj(acc, training_data, training_labels, test_data, test_labels, n_permutations=300)
        
        
    os.chdir(results_path)
    np.savetxt('samemod_accs_acr_subj'+'_'+ExpType+'.txt', mask_accuracies)
    np.savetxt('samemod_pvals_acr_subj'+'_'+ExpType+'.txt', mask_pvals)
        
        
#across modality, within subject        
def difmod_percim(ExpType, pipeline, *subjID):
    
    masks=['sl_Perc_Im_mask.nii.gz', 'sl_Perc_Im_mask_new.nii.gz'] #['left.OSC.625.nii.gz', 'right.OSC.bin.nii.gz', 'OSC.625.nii.gz']
    masknames=['SL_big', 'SL_small']  #['l_OSC', 'r_OSC', 'l+r_OSC'
    results_path=("/home/elena/ATTEND/results/difmod_%s_%s" %(ExpType, pipeline))

    if os.path.isdir(results_path)==False:
        os.mkdir(results_path)
    cwd="/home/elena/ATTEND/cross_modal/code"
    os.chdir(cwd)
    
    for subj in subjID:
        subjName=subj[-4:]
        roi_acc=[0]*len(masks)
        roi_pval=[0]*len(masks)
        for masknum in range(0, int(len(masks))):
            mask=masks[masknum]
            n_run, data=load_data_PercIm(pipeline, subjName, mask)
            if ExpType=="Perc":
                labels_train, volumes_train=get_protocol_data_Perc(n_run, subjName, block_dur_invol=8, N_volumes_run=170, N_removed=5)
                
                
                labels_test, rts, volumes_test=get_protocol_data_Im(n_run, subjName, TR=2, block_dur_invol=8, N_volumes_run=170, N_removed=5)
                
            else:
                if ExpType=="Im":
                    labels_train, rts, volumes_train=get_protocol_data_Im(n_run, subjName, TR=2, block_dur_invol=8, N_volumes_run=170, N_removed=5)
                    labels_test, volumes_test=get_protocol_data_Perc(n_run, subjName, block_dur_invol=8, N_volumes_run=170, N_removed=5)
            labels_train, data_train=pack_training_data(n_run, volumes_train,labels_train,data)
        
            acc=leave_one_run_out_difmod(n_run, data_train, labels_train, volumes_test,  labels_test, data)
        
            pval=permutation_test_difmod(n_run, acc, data_train, labels_train,volumes_test, labels_test, data, n_permutations=1000)
            roi_acc[masknum]=acc
            roi_pval[masknum]=pval
        os.chdir(results_path)
        np.savetxt('difmod_accs_'+subj+'_'+ExpType+'.txt', roi_acc)
        np.savetxt('difmod_pvals_'+subj+'_'+ExpType+'.txt', roi_pval)


def difmod_percim_betamaps(ExpType, pipeline, block_dur_invol, stim_post,N_volumes_run, TR, N_removed, *subjID):
    
    masks=['sl_Perc_Im_mask.nii', 'sl_Perc_Im_mask_new.nii'] #['left.OSC.625.nii.gz', 'right.OSC.bin.nii.gz', 'OSC.625.nii.gz']
    masknames=['SL_big', 'SL_small']  #['l_OSC', 'r_OSC', 'l+r_OSC'
    results_path=("/home/elena/ATTEND/results/betamaps_difmod_%s_%s" %(ExpType, pipeline))

    if os.path.isdir(results_path)==False:
        os.mkdir(results_path)
    cwd="/home/elena/ATTEND/cross_modal/code"
    os.chdir(cwd)
    
    for subj in subjID:
        subjName=subj[-4:]
        roi_acc=[0]*len(masks)
      #  roi_pval=[0]*len(masks)
        for masknum in range(0, int(len(masks))):
            mask=masks[masknum]
            n_run, data=load_data_PercIm(pipeline, subjName, mask)
            if ExpType=="Perc":
                
                labels_train, volumes_train=get_protocol_data_Perc(n_run, subjName, block_dur_invol, N_volumes_run, N_removed)
                training_data=calc_subj_maps_pi_nonavp(n_run,  data, volumes_train, block_dur_invol, labels_train, stim_post, N_volumes_run, N_removed, TR, 4) #hrf_lag
                
                labels_test, rts, volumes_test=get_protocol_data_Im(n_run, subjName, TR, block_dur_invol, N_volumes_run, N_removed)
                test_data=calc_subj_maps_pi_nonavi(n_run,  data, volumes_test, rts, labels_test, stim_post, N_volumes_run, N_removed, TR, 4) #hrf_lag
            else:
                if ExpType=="Im":
                    labels_train, rts, volumes_train=get_protocol_data_Im(n_run, subjName, TR, block_dur_invol, N_volumes_run, N_removed)
                    training_data=calc_subj_maps_pi_nonavi(n_run,  data, volumes_train, rts, labels_train, stim_post, N_volumes_run, N_removed, TR, 4) #hrf_lag
                    
                    labels_test, volumes_test=get_protocol_data_Perc(n_run, subjName, block_dur_invol, N_volumes_run, N_removed)
                    test_data=calc_subj_maps_pi_nonavp(n_run,  data, volumes_test, block_dur_invol, labels_test, stim_post, N_volumes_run, N_removed, TR, 4) #hrf_lag
            clf = LogisticRegressionCV(Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', scoring=None, solver='lbfgs', tol=1e-4, max_iter=100, class_weight=None, n_jobs=1, verbose=0, refit=True, intercept_scaling=1., multi_class='ovr', random_state=None).fit(np.asarray(training_data), np.reshape(np.array(labels_train), -2), sample_weight=None)
            new_test_labels=clf.predict(test_data)
                              #  print test_labels
            new_array=new_test_labels.tolist()
            test_labels=np.reshape(np.array(labels_train), -2)
            print (len(test_labels))
            #    print new_array
            #    print test_labels
            roi_acc[masknum]=((sum(np.array(test_labels)==np.array(new_array))).astype(float)/len(test_labels)).astype(float)
            
        os.chdir(results_path)
        np.savetxt('betamaps_difmod_accs_'+subj+'_'+ExpType+'.txt', roi_acc)
      #  np.savetxt('_betamaps_difmod_pvals_'+subj+'_'+ExpType+'.txt', roi_pval)

def calc_subj_maps_pi_nonavp(n_run,  data, volumes, for_dm, labels, stim_post, N_volumes_run1, N_removed, TR, hrf_lag): #hrf_lag
        subj_map=[]
        dm_param=for_dm
        for r in range(0, n_run):
                  #  print "beta maps run", r
                 #   print len(data[r])
   
                    for vol in range(0, len(volumes[r])):
                        
                                            #    print counter
                    #    print data[r].shape
                    #    print volumes[r][vol]
                  #      print dm_param
                        beta_map=calculate_beta_maps(data[r], volumes[r][vol], hrf_lag,stim_post, dm_param, N_volumes_run1)
                   #     print len(beta_maps[counter])
                        
           
                        subj_map.append(beta_map)
         
      #  print subj_map.shape     
        return subj_map   


def calc_subj_maps_pi_nonavi(n_run,  data, volumes, for_dm, labels, stim_post, N_volumes_run1, N_removed, TR, hrf_lag): #hrf_lag
        subj_map=[]
        
        for r in range(0, n_run):
                  #  print "beta maps run", r
                 #   print len(data[r])
   
                    for vol in range(0, len(volumes[r])):
                        
                           
                        dm_param=int((for_dm[r][vol]-volumes[r][vol]*TR)/TR)
                        if dm_param<1:
                                dm_param=8
                    #    print counter
                    #    print data[r].shape
                    #    print volumes[r][vol]
                  #      print dm_param
                        beta_map=calculate_beta_maps(data[r], volumes[r][vol], hrf_lag,stim_post, dm_param, N_volumes_run1)
                   #     print len(beta_maps[counter])
                        
           
                        subj_map.append(beta_map)
         
      #  print subj_map.shape     
        return subj_map   

#same modality, within subject
def samemod_percim(ExpType, pipeline, *subjID):

    masks=['left.OSC.625.nii.gz', 'right.OSC.bin.nii.gz', 'OSC.625.nii.gz']
    masknames=['l_OSC', 'r_OSC', 'l+r_OSC']
    results_path=("/home/elena/ATTEND/results/samemod_%s_%s" %(ExpType, pipeline))

    if os.path.isdir(results_path)==False:
        os.mkdir(results_path)
    cwd="/home/elena/FMRI_decoding/decoding_scripts"
    os.chdir(cwd)

    for subj in subjID:
        subjName=subj[-4:]
        roi_acc=[0]*len(masks)
        roi_pval=[0]*len(masks)
        for masknum in range(0, int(len(masks))):
            mask=masks[masknum]
            n_run, data=load_data_PercIm(pipeline, subjName, mask)
            if ExpType=="Perc":
                labels, volumes=get_protocol_data_Perc(n_run, subjName, block_dur_invol=8, N_volumes_run=170, N_removed=5)
            else:
                if ExpType=="Im":
                    labels, volumes=get_protocol_data_Im(n_run, subjName, TR=2, block_dur_invol=8, N_volumes_run=170, N_removed=5)
                
            acc=leave_one_run_out_clsf(n_run, volumes, labels, data)
            pval=permutation_test(n_run, acc, volumes, labels, data, n_permutations=300)
            roi_acc[masknum]=acc
            roi_pval[masknum]=pval
        os.chdir(results_path)
        np.savetxt('samemod_accs_'+subj+'_'+ExpType+'.txt', roi_acc)
        np.savetxt('samemod_pvals_'+subj+'_'+ExpType+'.txt', roi_pval)
        
        
        
        
def samemod_percim_percent_test(ExpType, pipeline, *subjID):

    masks=['left.OSC.625.nii.gz', 'right.OSC.bin.nii.gz', 'OSC.625.nii.gz']
   # masknames=['l_OSC', 'r_OSC', 'l+r_OSC']
    results_path=("/home/elena/ATTEND/results/samemod_tests_%s_%s" %(ExpType, pipeline))

    if os.path.isdir(results_path)==False:
        os.mkdir(results_path)
    cwd="/home/elena/FMRI_decoding/decoding_scripts"
    os.chdir(cwd)

    for subj in subjID:
        subjName=subj[-4:]
        roi_acc=[0]*len(masks)
    #    roi_pval=[0]*len(masks)
        for masknum in range(0, int(len(masks))):
            mask=masks[masknum]
            n_run, data=load_data_PercIm(pipeline, subjName, mask)
            if ExpType=="Perc":
                labels, volumes=get_protocol_data_Perc(n_run, subjName, block_dur_invol=8, N_volumes_run=170, N_removed=5)
            else:
                if ExpType=="Im":
                    labels,rts, volumes=get_protocol_data_Im(n_run, subjName, TR=2, block_dur_invol=8, N_volumes_run=170, N_removed=5)
                
            acc=leave_more_than_one_run_out_clsf(n_run, volumes, labels, data)
        #    pval=permutation_test(n_run, acc, volumes, labels, data, n_permutations=300)
            roi_acc[masknum]=acc
            print (roi_acc)
        #    roi_pval[masknum]=pval
        os.chdir(results_path)
        np.savetxt('samemod_test_accs_'+subjName+'_'+ExpType+'.txt', roi_acc)
      #  np.savetxt('samemod_pvals_'+subj+'_'+ExpType+'.txt', roi_pval)
        

def samemod_percim_across_subj_percent_test(ExpType, pipeline, *subjID):

    masks=['left.OSC.625.nii.gz', 'right.OSC.bin.nii.gz', 'OSC.625.nii.gz']
    masknames=['l_OSC', 'r_OSC', 'l+r_OSC']
    results_path=("/home/elena/ATTEND/results/samemod_acr_subj_checks_%s_%s" %(ExpType, pipeline))
    n_of_subj=len(subjID)
 #   print n_of_subj
    if os.path.isdir(results_path)==False:
        os.mkdir(results_path)
    cwd="/home/elena/FMRI_decoding/decoding_scripts"
    os.chdir(cwd)
    mask_accuracies=[0]*len(masknames)
  #  mask_pvals=[0]*len(masknames)
    for masknum in range(0, int(len(masks))):
        mask=masks[masknum]
        all_subj_data=[0]*len(subjID)
   # all_subj_vols=[0]*len(subjID)
        subject_labels=[0]*len(subjID)
        for subj in range(0, int(len(subjID))):
            subjName=subjID[subj][-4:]
  
            n_run, data=load_data_PercIm(pipeline, subjName, mask)
            if ExpType=="Perc":
                labels, volumes=get_protocol_data_Perc(n_run, subjName, block_dur_invol=8, N_volumes_run=170, N_removed=5)
            else:
                if ExpType=="Im":
                    labels, rts, volumes=get_protocol_data_Im(n_run, subjName, TR=2, block_dur_invol=8, N_volumes_run=170, N_removed=5)
            subject_data=[]
            for r in range(0, n_run):
                
                for vol in volumes[r]:
                    subject_data.append(data[r][vol])
                    subject_labels[subj]=np.reshape(np.array(labels), -2)
            all_subj_data[subj]=subject_data
        
        acc=leave_more_than_one_subj_out_clsf(n_of_subj, all_subj_data, subject_labels)
        mask_accuracies[masknum]=acc
        print (mask_accuracies)
       # mask_pvals[masknum]=permutation_test_acr_subj(acc, training_data, training_labels, test_data, test_labels, n_permutations=300)
        
        
    os.chdir(results_path)
    np.savetxt('samemod_accs_acr_subj_check'+'_'+ExpType+'.txt', mask_accuracies)
   # np.savetxt('samemod_pvals_acr_subj'+'_'+ExpType+'.txt', mask_pvals)
        

       

#across modality between subject
def difmod_percim_across_subj(ExpType, pipeline, *subjID):
    n_of_subj=len(subjID)
    masks=['left.OSC.625.nii.gz', 'right.OSC.bin.nii.gz', 'OSC.625.nii.gz']
    masknames=['l_OSC', 'r_OSC', 'l+r_OSC']
    results_path=("/home/elena/ATTEND/results/difmod_acr_subj_%s_%s" %(ExpType, pipeline))

    if os.path.isdir(results_path)==False:
        os.mkdir(results_path)
    cwd="/home/elena/FMRI_decoding/decoding_scripts"
    os.chdir(cwd)
    mask_accuracies=[0]*len(masknames)
    mask_pvals=[0]*len(masknames)
    for masknum in range(0, int(len(masks))):
        mask=masks[masknum]
        all_subj_data_train=[0]*len(subjID)
        all_subj_data_test=[0]*len(subjID)
   # all_subj_vols=[0]*len(subjID)
        subject_labels_train=[0]*len(subjID)
        subject_labels_test=[0]*len(subjID)
        for subj in range(0, int(len(subjID))):
            subjName=subjID[subj][-4:]
  
            n_run, data=load_data_PercIm(pipeline, subjName, mask)
            if ExpType=="Perc":
                labels_train, volumes_train=get_protocol_data_Perc(n_run, subjName, block_dur_invol=8, N_volumes_run=170, N_removed=5)
                labels_test, volumes_test=get_protocol_data_Im(n_run, subjName, TR=2, block_dur_invol=8, N_volumes_run=170, N_removed=5)
            else:
                if ExpType=="Im":
                    labels_train, volumes_train=get_protocol_data_Im(n_run, subjName, TR=2, block_dur_invol=8, N_volumes_run=170, N_removed=5)
                    labels_test, volumes_test=get_protocol_data_Perc(n_run, subjName, block_dur_invol=8, N_volumes_run=170, N_removed=5)
            labels_train, data_train=pack_training_data(n_run, volumes_train,labels_train,data)
            
            subject_data_train=[]
            subject_data_test=[]
            for r in range(0, n_run):
                
                for vol in volumes_train[r]:
                    subject_data_train.append(data[r][vol])
                    subject_labels_train[subj]=np.reshape(np.array(labels_train), -2)
                for vol in volumes_test[r]:
                    subject_data_test.append(data[r][vol])
                    subject_labels_test[subj]=np.reshape(np.array(labels_train), -2)
            
            
            all_subj_data_train[subj]=subject_data_train
            all_subj_data_test[subj]=subject_data_test
            
        
        acc, training_data, training_labels, test_data, test_labels =leave_one_subj_out_difmod(n_of_subj, all_subj_data_train, subject_labels_train, all_subj_data_test, subject_labels_test)
        
        mask_accuracies[masknum]=acc
        mask_pvals[masknum]=permutation_test_acr_subj(acc, training_data, training_labels, test_data, test_labels, n_permutations=300)
        
        
    os.chdir(results_path)
    np.savetxt('difmod_accs_acr_subj'+'_'+ExpType+'.txt', mask_accuracies)
    np.savetxt('difmod_pvals_acr_subj'+'_'+ExpType+'.txt', mask_pvals)
          
          
          
##########################          
          
def difmod_percim_percent_test(ExpType, pipeline, *subjID):

    masks=['left.OSC.625.nii.gz', 'right.OSC.bin.nii.gz', 'OSC.625.nii.gz']
   # masknames=['l_OSC', 'r_OSC', 'l+r_OSC']
    results_path=("/home/elena/ATTEND/results/samemod_tests_%s_%s" %(ExpType, pipeline))

    if os.path.isdir(results_path)==False:
        os.mkdir(results_path)
    cwd="/home/elena/FMRI_decoding/decoding_scripts"
    os.chdir(cwd)

    for subj in subjID:
        subjName=subj[-4:]
        roi_acc=[0]*len(masks)
    #    roi_pval=[0]*len(masks)
        for masknum in range(0, int(len(masks))):
            mask=masks[masknum]
            n_run, data=load_data_PercIm(pipeline, subjName, mask)
            if ExpType=="Perc":
                labels_train, volumes_train=get_protocol_data_Perc(n_run, subjName, block_dur_invol=8, N_volumes_run=170, N_removed=5)
                labels_test, rts, volumes_test=get_protocol_data_Im(n_run, subjName, TR=2,block_dur_invol=8, N_volumes_run=170, N_removed=5)
           
           
            else:
                if ExpType=="Im":
                    labels_train,rts, volumes_train=get_protocol_data_Im(n_run, subjName, TR=2, block_dur_invol=8, N_volumes_run=170, N_removed=5)
                    labels_test, volumes_test=get_protocol_data_Perc(n_run, subjName, block_dur_invol=8, N_volumes_run=170, N_removed=5)
  
            
            acc=leave_more_than_one_run_out_difmod(n_run, volumes_train, labels_train, volumes_test, labels_test, data)
        #    pval=permutation_test(n_run, acc, volumes, labels, data, n_permutations=300)
            roi_acc[masknum]=acc
            print (roi_acc)
        #    roi_pval[masknum]=pval
        os.chdir(results_path)
        np.savetxt('difmod_test_accs_'+subjName+'_'+ExpType+'.txt', roi_acc)
      #  np.savetxt('samemod_pvals_'+subj+'_'+ExpType+'.txt', roi_pval)
        

def difmod_percim_across_subj_percent_test(ExpType, pipeline, *subjID):

    masks=['left.OSC.625.nii.gz', 'right.OSC.bin.nii.gz', 'OSC.625.nii.gz']
    masknames=['l_OSC', 'r_OSC', 'l+r_OSC']
    results_path=("/home/elena/ATTEND/results/samemod_acr_subj_checks_%s_%s" %(ExpType, pipeline))
    n_of_subj=len(subjID)
 #   print n_of_subj
    if os.path.isdir(results_path)==False:
        os.mkdir(results_path)
    cwd="/home/elena/FMRI_decoding/decoding_scripts"
    os.chdir(cwd)
    mask_accuracies=[0]*len(masknames)
  #  mask_pvals=[0]*len(masknames)
    for masknum in range(0, int(len(masks))):
        mask=masks[masknum]
        all_subj_data_train=[0]*len(subjID)
        all_subj_data_test=[0]*len(subjID)
   # all_subj_vols=[0]*len(subjID)
        subject_labels_train=[0]*len(subjID)
        subject_labels_test=[0]*len(subjID)
        for subj in range(0, int(len(subjID))):
            subjName=subjID[subj][-4:]
  
            n_run, data=load_data_PercIm(pipeline, subjName, mask)
            if ExpType=="Perc":
                labels_train, volumes_train=get_protocol_data_Perc(n_run, subjName, block_dur_invol=8, N_volumes_run=170, N_removed=5)
                labels_test, rts, volumes_test=get_protocol_data_Im(n_run, subjName, TR=2,block_dur_invol=8, N_volumes_run=170, N_removed=5)
            
            
            
            else:
                if ExpType=="Im":
                    labels_test, volumes_test=get_protocol_data_Perc(n_run, subjName, block_dur_invol=8, N_volumes_run=170, N_removed=5)
                    labels_train, rts, volumes_train=get_protocol_data_Im(n_run, subjName, TR=2, block_dur_invol=8, N_volumes_run=170, N_removed=5)
            subject_data=[]
            for r in range(0, n_run):
                
                for vol in volumes_train[r]:
                    subject_data.append(data[r][vol])
                    subject_labels_train[subj]=np.reshape(np.array(labels_train), -2)
            all_subj_data_train[subj]=subject_data
            
            subject_data=[]
            for r in range(0, n_run):
                for vol in volumes_test[r]:
                    subject_data.append(data[r][vol])
                    subject_labels_test[subj]=np.reshape(np.array(labels_test), -2)
            all_subj_data_test[subj]=subject_data
            
        
        acc=leave_more_than_one_subj_out_difmod(n_of_subj, all_subj_data_train, subject_labels_train, all_subj_data_test, subject_labels_test)
        mask_accuracies[masknum]=acc
        print (mask_accuracies)
       # mask_pvals[masknum]=permutation_test_acr_subj(acc, training_data, training_labels, test_data, test_labels, n_permutations=300)
        
        
    os.chdir(results_path)
    np.savetxt('difmod_accs_acr_subj_check'+'_'+ExpType+'.txt', mask_accuracies)
   # np.savetxt('samemod_pvals_acr_subj'+'_'+ExpType+'.txt', mask_pvals)
        

                 
          
