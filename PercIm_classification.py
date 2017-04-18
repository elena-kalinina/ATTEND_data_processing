# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 16:15:22 2015

@author: elena
"""

import nilearn
import numpy as np
import scipy
from scipy import stats
import nibabel
import sklearn as sck
import os
#import mvpa2
#from mvpa2 import *
#from nilearn.masking import apply_mask
#from nilearn.masking import compute_epi_mask
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

def leave_one_run_out_clsf(n_run, volume_num, labels, masked_data_scaled):

    cv_accuracy=[0]*n_run
 
    print ("Splitting the data into training and test samples for cross-validation")     
    for r in range(0, n_run):
                test_run = r
                #print test_run
                training_runs = [x for x in range (0, n_run) if x!=test_run]                
                #print training_runs
                test_labels=labels[test_run]
                
                
                test_data=[]
                #print volume_num[r]
                #print len(masked_data_scaled[test_run])
                for vol in volume_num[test_run]:
                   # print vol
                    
                    #construct test data taking volumes corresponding to stimuli where hrf reaches its maximum
                    test_data.append(masked_data_scaled[test_run][vol])
                
                training_labels= []#[0]*len(training_runs)
                training_data=[]
                #k=0
                for tr in training_runs:
 
                    #print tr
                    training_labels.append(labels[tr])                    
                    #training_labels[k]=labels[tr]
                    #k=k+1
                    for vol in volume_num[tr]: #range(0, len(volume_num[r])):
                       # print vol
                    #one_run.append(masked_data_scaled[tr][vol])
                    #training_data[counter]=np.array(one_run)
                        training_data.append(masked_data_scaled[tr][vol])                  
                   #     training_labels.append(labels[tr][vol])
                training_labels=np.reshape(np.array(training_labels), -2)
                #print training_labels
                #print test_labels
                #print size(training_data)
                #print size(test_data)                 
                #Running the classifier
                #clf = lda.LDA(n_components=None, priors=None).fit(training_data, training_labels)
                clf = svm.SVC(kernel='linear', C=1).fit(training_data, training_labels)
                cv_accuracy[r]=clf.score(test_data, test_labels) 
              #  print cv_accuracy[r]
    accuracy=np.mean(cv_accuracy)
   # print accuracy
 #   accuracy_sem=np.std(cv_accuracy)/np.sqrt(len(cv_accuracy))
    return accuracy #, accuracy_sem
    
def pack_training_data(n_run, volume_num_train,labels_train,masked_data_scaled):    
    
    for r in range(0, n_run):
                        
            training_runs = range(0, int(n_run))
            #training_labels=labels_train
            #training_labels=np.reshape(np.array(training_labels), -2)
            training_labels=np.reshape(np.array(labels_train), -2)
            training_data=[] #[0]*len(training_runs)
            
            for tr in training_runs:
                #print tr
                for vol in volume_num_train[tr]:
                    #print vol
                #for vol in range(0, len(volume_num_train[r])):
                    #one_run.append(masked_data_scaled[tr][vol])
                    #training_data[counter]=np.array(one_run)
                    training_data.append(masked_data_scaled[tr][vol])  
            #        training_labels.append(labels_train[tr][vol])
    
    return training_labels, training_data
    
def leave_one_run_out_difmod(n_run, training_data, training_labels, volume_num_test,  labels_test, masked_data_scaled):

    cv_accuracy=[0]*n_run
 
    print ("Splitting the data into training and test samples for cross-validation")     
    
  
    for r in range(0, n_run):
                test_run = r
                
                test_labels=labels_test[test_run]
                #print test_labels
                test_data=[]
           #     print volume_num_test[test_run]
                for vol in volume_num_test[test_run]:
                    #print vol
                    #construct test data taking volumes corresponding to stimuli where hrf reaches its maximum
                    test_data.append(masked_data_scaled[test_run][vol])
             #   print len(test_data)   
                #Running the classifier
                #clf = lda.LDA(n_components=None, priors=None).fit(training_data, training_labels)
                clf = svm.SVC(kernel='linear', C=1).fit(training_data, training_labels)
                cv_accuracy[r]=clf.score(test_data, test_labels) 
                
    accuracy=np.mean(cv_accuracy)
   # print accuracy
 #   accuracy_sem=np.std(cv_accuracy)/np.sqrt(len(cv_accuracy))
    return accuracy #, accuracy_sem    
    
    
     
def scrambled(orig):
    dest = orig[:]
    random.shuffle(dest)
    return dest     
     

def permutation_test(n_run, accuracy, volume_num, labels, masked_data_scaled, n_permutations):
    print ("Starting permutation tests")
    acc_dist=np.zeros(n_permutations)
    for perm in range(0, n_permutations):
                #print perm
                #scrambling labels, running the classifier
                cv_accuracy=[0]*n_run
                #scrambled_labels=[0]*n_run
                #for r in range(0, n_run):
                 #   scrambled_labels[r]=scrambled(labels[r])
                
                
                for r in range(0, n_run):
                    test_run = r
                    #print test_run
                    training_runs = [x for x in range (0, n_run) if x!=test_run]
                    #print training_runs
                    test_labels=labels[test_run]
                    test_data=[]
                    for vol in volume_num[r]:
                        #construct test data taking volumes corresponding to stimuli where hrf reaches its maximum
                        test_data.append(masked_data_scaled[test_run][vol])
                    training_labels=[]
                    training_data=[] #[0]*len(training_runs)
                    
                    for tr in training_runs:
                        
                        #for vol in range(0, len(volume_num[r])):
                        for vol in volume_num[tr]:
                             
                            training_data.append(masked_data_scaled[tr][vol])        
                        training_labels.append(scrambled(labels[tr]))
                    training_labels=np.reshape(np.array(training_labels), -2)
                    #Running the classifier  
                    #clf = lda.LDA(n_components=None, priors=None).fit(training_data, training_labels)
                    clf = svm.SVC(kernel='linear', C=1).fit(training_data, training_labels)
                    cv_accuracy[r]=clf.score(test_data, test_labels)
                acc_dist[perm]=np.mean(cv_accuracy)
            
    p_val=1-scipy.stats.norm(np.mean(acc_dist), np.std(acc_dist)).cdf(accuracy)
    print ('p-value of the accuracy=', accuracy, 'is', p_val)
    return p_val
    
     
     
def permutation_test_difmod(n_run, accuracy, training_data, training_labels, volume_num_test, labels_test, masked_data_scaled, n_permutations):
    acc_dist=np.zeros(n_permutations)
    print ("Starting permutation tests")
    for perm in range(0, int(n_permutations)):
                
                #scrambling labels, running the classifier
                cv_accuracy=[0]*n_run
                training_labels=scrambled(training_labels) 
                for r in range(0, n_run):
                    test_run = r
                    test_labels=labels_test[test_run]
                    test_data=[]
                    for vol in volume_num_test[test_run]:
                        #construct test data taking volumes corresponding to stimuli where hrf reaches its maximum
                        test_data.append(masked_data_scaled[test_run][vol])
                    #clf = lda.LDA(n_components=None, priors=None).fit(training_data, training_labels)
                    clf = svm.SVC(kernel='linear', C=1).fit(training_data, training_labels)
                    cv_accuracy[r]=clf.score(test_data, test_labels)
                acc_dist[perm]=np.mean(cv_accuracy)
            
    p_val=1-scipy.stats.norm(np.mean(acc_dist), np.std(acc_dist)).cdf(accuracy)
            
    print ('p-value of the accuracy=', accuracy, 'is', p_val)
    return p_val
    
def leave_one_subj_out_clsf(n_of_subj, all_subj_data, subject_labels):

    
 
    print ("Splitting the data into training and test samples for cross-validation")     
    for s in range(0, int(n_of_subj)):
            test_subj = s
            training_labels=[] #[0]*(len(subjID)-1)
            training_data=[]#[0]*(len(subjID)-1)
          #  print subjID[s]
            training_subj = [x for x in range (0, int(n_of_subj)) if x!=test_subj]                
       #     print training_subj
            test_labels=subject_labels[s]
            test_data=all_subj_data[s]
        #    k=0            
            for tr in training_subj:
                
                for v in range (0, int(len(all_subj_data[tr]))):
                    training_data.append(all_subj_data[tr][v])
                    training_labels.append(subject_labels[tr][v])
        #        k=k+1
         #   training_labels=np.reshape(np.array(training_labels), -2)#[0]*len(training_runs)
           # training_data=np.reshape(np.array(training_data), -2)
            #Running the classifier
            #clf = lda.LDA(n_components=None, priors=None).fit(training_data, training_labels)
            print (len(training_labels))
        #    print size(training_data)
            clf = svm.SVC(kernel='linear', C=1).fit(training_data, training_labels)
            accuracy=clf.score(test_data, test_labels)
   # print accuracy
 #   accuracy_sem=np.std(cv_accuracy)/np.sqrt(len(cv_accuracy))
    return accuracy, training_data, training_labels, test_data, test_labels #, accuracy_sem    

def permutation_test_acr_subj(acc, training_data, training_labels, test_data, test_labels, n_permutations):
    acc_dist=np.zeros(n_permutations)
    print (len(test_labels))
    print (len(training_labels))
    for perm in range(0, int(n_permutations)):
                training_labels=scrambled(training_labels)
                    #Running the classifier  
                    #clf = lda.LDA(n_components=None, priors=None).fit(training_data, training_labels)
                clf = svm.SVC(kernel='linear', C=1).fit(training_data, training_labels)
                acc_dist[perm]=clf.score(test_data, test_labels)
                
    p_val=1-scipy.stats.norm(np.mean(acc_dist), np.std(acc_dist)).cdf(acc)
    print ('p-value of the accuracy=', acc, 'is', p_val) 
    return p_val 
    
def leave_one_subj_out_difmod(n_of_subj, all_subj_data_train, subject_labels_train, all_subj_data_test, subject_labels_test):

    print ("Splitting the data into training and test samples for cross-validation")     
    for s in range(0, int(n_of_subj)):
            test_subj = s
            training_labels=[] #[0]*(len(subjID)-1)
            training_data=[]#[0]*(len(subjID)-1)
         #   print subjID[s]
            training_subj = [x for x in range (0, int(n_of_subj)) if x!=test_subj]                
          #  print training_subj
            test_labels=subject_labels_test[s]
            test_data=all_subj_data_test[s]
        #    k=0            
            for tr in training_subj:
                
                for v in range (0, int(len(all_subj_data_train[tr]))):
                    training_data.append(all_subj_data_train[tr][v])
                    training_labels.append(subject_labels_train[tr][v])
        #        k=k+1
         #   training_labels=np.reshape(np.array(training_labels), -2)#[0]*len(training_runs)
           # training_data=np.reshape(np.array(training_data), -2)
            #Running the classifier
            #clf = lda.LDA(n_components=None, priors=None).fit(training_data, training_labels)
            print (len(training_labels))
        #    print size(training_data)
            clf = svm.SVC(kernel='linear', C=1).fit(training_data, training_labels)
            accuracy=clf.score(test_data, test_labels)
   # print accuracy
 #   accuracy_sem=np.std(cv_accuracy)/np.sqrt(len(cv_accuracy))
    return accuracy, training_data, training_labels, test_data, test_labels
    
    
    
    
    
    
    
    
def leave_more_than_one_run_out_clsf(n_run, volume_num, labels, masked_data_scaled):
        options=[1,2,3]        
        ind=np.array(range(0, n_run))
        all_accs=[0]*len(options)
        counter1=0
        for opt in options:
            
            print ("Now testing leaving" + str(opt) + "runs out as test runs")
            accs_opt=[0]*len(ind)
            for i in range(0, len(ind)):
                
                ind=np.roll(ind, opt, axis=0)
          #      print ind
                test_run=ind[0:opt]
            #    print test_run
                training_runs=list(set(ind) - set(test_run))
            #    print training_runs#[x for x in range (0, n_run) if x!=test_run] 
                
 
                print ("Splitting the data into training and test samples for cross-validation")     
            
            
             
                training_labels= []#[0]*len(training_runs)
                training_data=[]
                #k=0
                for tr in training_runs:
 
                    #print tr
                    training_labels.append(labels[tr])                    
                    #training_labels[k]=labels[tr]
                    #k=k+1
                    for vol in volume_num[tr]: #range(0, len(volume_num[r])):
                       # print vol
                    #one_run.append(masked_data_scaled[tr][vol])
                    #training_data[counter]=np.array(one_run)
                        training_data.append(masked_data_scaled[tr][vol])                  
                   #     training_labels.append(labels[tr][vol])
                training_labels=np.reshape(np.array(training_labels), -2)
            #    print len(training_labels)
                #print test_labels
            #    print len(training_data)
                #print size(test_data)                 
                #Running the classifier
                #clf = lda.LDA(n_components=None, priors=None).fit(training_data, training_labels)
                cv_accuracy=[0]*len(test_run)
                counter=0
                for r in test_run:
                    
                    test_data=[]
                    test_labels=[]
                       
                #print training_runs
                
                
                    test_labels=labels[r]        
                
                #print volume_num[r]
                #print len(masked_data_scaled[test_run])
                    for vol in volume_num[r]:
                   # print vol
                    
                    #construct test data taking volumes corresponding to stimuli where hrf reaches its maximum
                        test_data.append(masked_data_scaled[r][vol])
            
            
              #      print len(test_data)
               #     print len(test_labels)
                    clf = svm.SVC(kernel='linear', C=1).fit(training_data, training_labels)
                    cv_accuracy[counter]=clf.score(test_data, test_labels)
               #     print cv_accuracy
                    counter=counter+1
              #  print cv_accuracy[r]
            #    print np.mean(cv_accuracy)
                
                accs_opt[i]=np.mean(cv_accuracy)
            all_accs[counter1]=np.mean(accs_opt)
          #  print all_accs[counter1]
            counter1=counter1+1
   # print accuracy
 #   accuracy_sem=np.std(cv_accuracy)/np.sqrt(len(cv_accuracy))
        return all_accs #, accuracy_sem
        
        


def leave_more_than_one_run_out_difmod(n_run, volume_num_train, labels_train, volume_num_test, labels_test, masked_data_scaled):
        options=[1,2,3]        
        ind=np.array(range(0, n_run))
        all_accs=[0]*len(options)
        counter1=0
        for opt in options:
            
            print ("Now testing leaving" + str(opt) + "runs out as test runs")
            accs_opt=[0]*len(ind)
            for i in range(0, len(ind)):
                
                ind=np.roll(ind, opt, axis=0)
          #      print ind
                test_run=ind[0:opt]
            #    print test_run
                training_runs=list(set(ind) - set(test_run))
            #    print training_runs#[x for x in range (0, n_run) if x!=test_run] 
                
                
                 
                training_labels= []#[0]*len(training_runs)
                training_data=[]
                #k=0
                for tr in training_runs:
 
                    #print tr
                    training_labels.append(labels_train[tr])                    
                    #training_labels[k]=labels[tr]
                    #k=k+1
                    for vol in volume_num_train[tr]: #range(0, len(volume_num[r])):
                       # print vol
                    #one_run.append(masked_data_scaled[tr][vol])
                    #training_data[counter]=np.array(one_run)
                        training_data.append(masked_data_scaled[tr][vol])                  
                   #     training_labels.append(labels[tr][vol])
                training_labels=np.reshape(np.array(training_labels), -2)
            #    print len(trai
                
                
                print ("Splitting the data into training and test samples for cross-validation")     
            
            
                #print test_labels
            #    print len(training_data)
                #print size(test_data)                 
                #Running the classifier
                #clf = lda.LDA(n_components=None, priors=None).fit(training_data, training_labels)
                cv_accuracy=[0]*len(test_run)
                counter=0
                for r in test_run:
                    
                    test_data=[]
                    test_labels=[]
                       
                #print training_runs
                
                
                    test_labels=labels_test[r]        
                
                #print volume_num[r]
                #print len(masked_data_scaled[test_run])
                    for vol in volume_num_test[r]:
                   # print vol
                    
                    #construct test data taking volumes corresponding to stimuli where hrf reaches its maximum
                        test_data.append(masked_data_scaled[r][vol])
            
            
              #      print len(test_data)
               #     print len(test_labels)
                    clf = svm.SVC(kernel='linear', C=1).fit(training_data, training_labels)
                    cv_accuracy[counter]=clf.score(test_data, test_labels)
               #     print cv_accuracy
                    counter=counter+1
              #  print cv_accuracy[r]
            #    print np.mean(cv_accuracy)
                
                accs_opt[i]=np.mean(cv_accuracy)
            all_accs[counter1]=np.mean(accs_opt)
          #  print all_accs[counter1]
            counter1=counter1+1
   # print accuracy
 #   accuracy_sem=np.std(cv_accuracy)/np.sqrt(len(cv_accuracy))
        return all_accs #, accuracy_sem
                
        
        
        
def leave_more_than_one_subj_out_difmod(n_of_subj, all_subj_data_train, subject_labels_train, all_subj_data_test, subject_labels_test):

    
    options=[3,6,9]#[3,6,9] 
    
    ind=np.array(range(0, n_of_subj))
    all_accs=[0]*len(options)
    counter1=0
    for opt in options:
            print ("Now testing leaving" + str(opt) + "subjects out as test subjects")
            accs_opt=[0]*len(ind)
            for i in range(0, len(ind)):
                ind=np.roll(ind, opt, axis=0)
                test_subj=ind[0:opt]
                training_subj=list(set(ind) - set(test_subj))#[x for x in range (0, n_run) if x!=test_run] 
           
                print ("Splitting the data into training and test samples for cross-validation")   
                training_labels= []#[0]*len(training_runs)
                training_data=[]
              
                for s in training_subj:#range(0, int(n_of_subj)):
                    for v in range (0, int(len(all_subj_data_train[s]))):
                        training_data.append(all_subj_data_train[s][v])
                        training_labels.append(subject_labels_train[s][v])
       #     print training_subj
                
        #    k=0
                print (len(training_labels))
                print (len(training_data))
                cv_accuracy=[0]*len(test_subj)
                counter2=0
                for st in test_subj:
                    test_data=[]
                    test_labels=[]
                 
                    for v in range (0, int(len(all_subj_data_test[st]))):
                        test_data.append(all_subj_data_test[st][v])
                        test_labels.append(subject_labels_test[st][v])
                 
        #        k=k+1
         #   training_labels=np.reshape(np.array(training_labels), -2)#[0]*len(training_runs)
           # training_data=np.reshape(np.array(training_data), -2)
            #Running the classifier
            #clf = lda.LDA(n_components=None, priors=None).fit(training_data, training_labels)
                    print (len(test_labels))
                    print (len(test_data))
                    clf = svm.SVC(kernel='linear', C=1).fit(training_data, training_labels)
                    cv_accuracy[counter2]=clf.score(test_data, test_labels)
                    counter2=counter2+1
                    print (cv_accuracy)
                accs_opt[i]=np.mean(cv_accuracy)
                print (accs_opt)
            all_accs[counter1]=np.mean(accs_opt)
            counter1=counter1+1
   # print accuracy
 #   accuracy_sem=np.std(cv_accuracy)/np.sqrt(len(cv_accuracy))
    return all_accs #, accuracy_sem    
