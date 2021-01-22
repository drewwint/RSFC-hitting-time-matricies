#!/usr/bin/env python3


# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:46:27 2021

@author: wintersd
"""


import os, glob, pathlib
import nilearn
import numpy as np
import scipy
import math
from nilearn import datasets
from nilearn.input_data import NiftiMapsMasker
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.image import high_variance_confounds
from nilearn import datasets
from nilearn import plotting, image
import nibabel as nib
import matplotlib.pyplot as plt
import astropy.table as table
import astropy.units as u
import sys
import csv
import getopt
import scipy.stats
import pandas as pd
from numpy import genfromtxt


## this function calculates hitting-time matrix
def hitting_matrix(correlation_matrix):
    L = np.size(correlation_matrix,axis = 0)
    A_matrix = np.array(correlation_matrix)
    D_matrix = np.zeros((L,L))
    for i in range(L):
        D_matrix[i,i] = np.sum(A_matrix[i])
        
    d_max = np.max(D_matrix)
    
    for j in range(L):
        if np.max(A_matrix[j,:]) < .05:
            A_matrix[j,j] = d_max - D_matrix[j,j]
    
    D_matrix = np.zeros((L,L))
    D_inv = np.zeros((L,L))
    D_sqrt = np.zeros((L,L))
    D_sqrt_inv = np.zeros((L,L))
    for i in range(L):
        D_matrix[i,i] = np.sum(A_matrix[i])
        D_inv[i,i] = 1./D_matrix[i,i]
        D_sqrt[i,i] = np.sqrt(D_matrix[i,i])
        D_sqrt_inv[i,i] = 1./D_sqrt[i,i]

    p_matrix = np.dot(D_inv, A_matrix)

    # Graph Laplacian
    eye_matrix = np.eye(L,L)
    eye_P = eye_matrix - p_matrix

    G_Lap = np.dot(D_sqrt,eye_P)
    G_Lap_n = np.dot(G_Lap, D_sqrt_inv)

    [eig_val, eig_vec] = np.linalg.eigh(G_Lap_n)
    H = np.zeros((L,L))
    d = np.sum(D_matrix)
    for i in range(L):
        for j in range(L):
            deg_i = D_matrix[i,i]
            deg_j = D_matrix[j,j]
            for k in range(L):
                if eig_val[k] != min(eig_val):
                    t_i = (eig_vec[i,k]*eig_vec[i,k])/deg_i
                    t_j = (eig_vec[j,k]*eig_vec[j,k])/deg_j
                    t_ij = (eig_vec[i,k]*eig_vec[j,k])/np.sqrt(deg_i*deg_j)
                    H[i,j] = H[i,j] + d*(1./(eig_val[k]))*(t_j-t_ij)

    H = np.transpose(H)
    return H


def main(argv):
## atlas 3
    atlas = "Schaefer200Yeo17Pauli" # msdl or haox or mmp
    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17, resolution_mm=1, data_dir=None, base_url=None, resume=True, verbose=1) #atlas_filename = "MMP1_rois.nii" #Glasser et al., 2016
    schaefer_filename = schaefer_atlas.maps
    schaefer_labels = schaefer_atlas.labels
    schaefer_masker = NiftiLabelsMasker(labels_img=schaefer_filename, standardize=True,
                               memory='nilearn_cache', verbose=5) 
    pauli_atlas = datasets.fetch_atlas_pauli_2017()
    pauli_filename = pauli_atlas.maps
    pauli_labels = pauli_atlas.labels
    pauli_masker = NiftiMapsMasker(maps_img=pauli_filename, standardize=True, verbose=5) 
    
    all_labels = np.hstack([schaefer_labels, pauli_labels])
    #print(all_labels)
    
    correlation_measure = ConnectivityMeasure(kind='correlation')




    #for this subject generate partial correlation matrix and hitting time matrix
   # parse command line input to find subject director "sub_data" 
   try:
      opts, args = getopt.getopt(argv,"hi:",["ifile="])
   except getopt.GetoptError:
      print ('test.py -i <sub_data>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ('test.py -i <sub_data>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
   print ('sub_data is "', sub_data)
    
    
    # generate subject number and position from sub_data
    subnum = sub_data.split(os.sep)[-4]
    #subnum_fmt = "{:06}".format(int(subnum))
    
    
    out_base= os.sep.join(sub_data.split(os.sep)[:-3])
    out_dir= out_base + os.sep + "deriv"+ os.sep + "snag"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)



    #extract time series from a atlas(es)
    
    file_base= "_".join(sub_data.split(os.sep)[-1].split("_")[:-2])
    
    
    adj_out_file = out_dir + os.sep + file_base + "_timeseries-corr_" + atlas + "_data_filt"
    if not(pathlib.Path(adj_out_file).exists()):
        func_dir_in = sub_data #+ os.sep + 'restEPI'  #directory with func images
        #funcs_filenames = glob.glob(func_dir_in + os.sep + '*.nii') #find all funcs in this directory
        confounds = high_variance_confounds(sub_data)
        #schafer cortical atlas 
        schaefer_time_series = schaefer_masker.fit_transform(sub_data, confounds=confounds)  #cortical segments
        print("schaefer ts shape: ")
        print(schaefer_time_series.shape)
        #subcortical atlas
        pauli_time_series = pauli_masker.fit_transform(sub_data, confounds=confounds)  #subccortical segments
        print("pauli ts shape: ")
        print(pauli_time_series.shape)
    
        #stack time series and determine adjacency matrix from the resulting set of time series
        full_ts_set = np.hstack((schaefer_time_series, pauli_time_series))
        print("concatenated ts shape: ")
        print(full_ts_set.shape)
        correlation_matrix = correlation_measure.fit_transform([full_ts_set])[0]
        np.savetxt(adj_out_file, correlation_matrix, delimiter=",")
        print(correlation_matrix.shape[0],correlation_matrix.shape[1])
    else:
        correlation_matrix = genfromtxt(adj_out_file, delimiter=',') #load the file if the correlation matrix was pre-computed
    correlation_matrix = abs(correlation_matrix)  #absolute value to make all transition probabilities positive
    np.fill_diagonal(correlation_matrix, 0) #set self connections to zero 
    p_corr_all[:,:,sub_ind] = correlation_matrix.copy()  #stack correlation matrices for later analysis

    #build hitting time matrix
    H_out_file = out_dir + os.sep + file_base + "_normedH_" + atlas + "_corr"  #file where hitting-time matrix will be saved
    print(H_out_file)
    if not(pathlib.Path(H_out_file).exists()): #compute hitting time matrix if it isn't already saved
        H = hitting_matrix(correlation_matrix)
        H_all[:,:,sub_ind] = H
        np.savetxt(H_out_file, H, delimiter=",")
        print("saved "+Data_Name)
    else:
        H_all[:,:,sub_ind] = genfromtxt(H_out_file, delimiter=',') #load the file if the hitting-time mat was already computed



if __name__ == "__main__":
   main(sys.argv[1:])






