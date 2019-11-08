'''
Calculate the information in the network
Can be by the full distribution rule (for small netowrk) or bt diffrenet approximation method
'''
import os
import time
import _pickle as cPickle
import multiprocessing
import warnings
import contextlib
import numpy as np
import keras.backend as K
import tensorflow as tf
import idnns.information.information_utilities as inf_ut
import idnns.information.entropy_estimators as ent
import idnns.information.nsb_entropy as nsb
import idnns.information.kde as kde
from idnns.networks import model as mo
from idnns.information.mutual_info_estimation import calc_varitional_information
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
NUM_CORES = multiprocessing.cpu_count()
from idnns.information.mutual_information_calculation import *
import matplotlib.pyplot as plt
from multiprocessing import Pool



def calc_information_sampling(data, bins, pys1, pxs, label, b, b1, len_unique_a, p_YgX, unique_inverse_x,
                              unique_inverse_y):
    bins = bins.astype(np.float32)
    #num_of_bins = bins.shape[0]
    # bins = stats.mstats.mquantiles(np.squeeze(data.reshape(1, -1)), np.linspace(0,1, num=num_of_bins))
    # hist, bin_edges = np.histogram(np.squeeze(data.reshape(1, -1)), normed=True)
    digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
    b2 = np.ascontiguousarray(digitized).view(
		np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
		np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_ts = unique_counts / float(sum(unique_counts))
    PXs, PYs = np.asarray(pxs).T, np.asarray(pys1).T
    local_IXT, local_ITY = calc_information_from_mat(PXs, PYs, p_ts, digitized, unique_inverse_x, unique_inverse_y,unique_array)
    return local_IXT, local_ITY

def calc_kde_info_for_layer (data, unique_inverse_x, unique_inverse_y, label, len_unique_a, pys, pxs, py_x, pys1, model_path,x,lower):
    #local_IXT=ent.entropy(data)
    #local_ITY=ent.micd(data,label)
    sig=0.01
    inds01=[]
    inds10=[]
    np.random.seed(None)
    data+=np.random.normal(0,sig,(len(data),len(data[0])))
    data=abs(data)
    for i in range(len(unique_inverse_y)):
        if unique_inverse_y[i]==0:
            inds01.append(i)
        else:
            inds10.append(i)
    data_given_y=[]
    data_given_y.append(data[inds01])
    data_given_y.append(data[inds10])
    h_t_given_y=0
    tf.reset_default_graph()
    sess = tf.Session()
    for i in range(len(pys1)):
        if not lower:
            h_t_given_y+=pys1[i]*kde.entropy_estimator_kl(tf.convert_to_tensor(data_given_y[i]),sig)
        else:
            h_t_given_y+=pys1[i]*kde.entropy_estimator_bd(tf.convert_to_tensor(data_given_y[i]),sig)
    datatf=tf.convert_to_tensor(data)
    if not lower:
        data_entropy=kde.entropy_estimator_kl(datatf,sig)
    else:
        data_entropy=kde.entropy_estimator_bd(datatf,sig)
    local_IXT=data_entropy-kde.kde_condentropy(data, sig)
    local_ITY=data_entropy-h_t_given_y
    sess = tf.Session()
    mi1,mi2=sess.run([local_IXT,local_ITY])
    sess.close()
    params = {}
    params['local_IXT'] = mi1/np.log(2)
    params['local_ITY'] = mi2/np.log(2)
    return params



def calc_information_for_layer_with_other(data, bins, unique_inverse_x, unique_inverse_y, label,
                                          b, b1, len_unique_a, pxs, p_YgX, pys1,layer,num_of_bins):
    local_IXT, local_ITY = calc_information_sampling(data, bins, pys1, pxs, label, b, b1,
                             len_unique_a, p_YgX, unique_inverse_x,unique_inverse_y)
    params = {}
    params['local_IXT'] = local_IXT
    params['local_ITY'] = local_ITY
    return params


def calc_kde_information_for_epoch(arg):
    iter_index, interval_information_display,ws_iter_index, unique_inverse_x, \
    unique_inverse_y, label, len_unique_a, pys, pxs, py_x, pys1, model_path,x,lower=arg
    params=np.array(
            [calc_kde_info_for_layer(ws_iter_index[i], unique_inverse_x,
                               unique_inverse_y, label,
                               len_unique_a, pys, pxs, py_x, pys1, model_path,x,lower)
            for i in range(len(ws_iter_index))])
    if np.mod(iter_index, interval_information_display) == 0:
        print('Calculated The information of epoch number - {0}'.format(iter_index))
    return params


def calc_information_for_epoch(iter_index, interval_information_display,ws_iter_index, bins, unique_inverse_x,
                               unique_inverse_y, label, b, b1,
                               len_unique_a, pys, pxs, py_x, pys1, model_path, input_size, layerSize, num_of_bins,per_layer_bins=False,
                               num_of_samples=100, sigma=0.5, ss=[], ks=[]):
    """Calculate the information for all the layers for specific epoch"""
    np.random.seed(None)
    if per_layer_bins:
        params = np.array(
		[calc_information_for_layer_with_other(data=ws_iter_index[i], bins=bins[i], unique_inverse_x=unique_inverse_x,
			                    unique_inverse_y=unique_inverse_y, label=label,
	                             b=b, b1=b1, len_unique_a=len_unique_a, pxs=pxs,
                                 p_YgX=py_x, pys1=pys1,layer=i,num_of_bins=num_of_bins)
			 for i in range(len(ws_iter_index))])
    else:
        params = np.array(
		[calc_information_for_layer_with_other(data=ws_iter_index[i], bins=bins, unique_inverse_x=unique_inverse_x,
			                    unique_inverse_y=unique_inverse_y, label=label,
	                             b=b, b1=b1, len_unique_a=len_unique_a, pxs=pxs,
                                 p_YgX=py_x, pys1=pys1,layer=i,num_of_bins=num_of_bins)
			 for i in range(len(ws_iter_index))])
    if np.mod(iter_index, interval_information_display) == 0:
        print('Calculated The information of epoch number - {0}'.format(iter_index))
    return params


def extract_probs(label, x):
    """calculate the probabilities of the given data and labels p(x), p(y) and (y|x) """
    pys = np.sum(label, axis=0) / float(label.shape[0])
    b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    unique_array, unique_indices, unique_inverse_x, unique_counts = \
    np.unique(b, return_index=True, return_inverse=True, return_counts=True)
    unique_a = x[unique_indices]
    b1 = np.ascontiguousarray(unique_a).view(np.dtype((np.void, unique_a.dtype.itemsize * unique_a.shape[1])))
    pxs = unique_counts / float(np.sum(unique_counts))
    p_y_given_x = []
    for i in range(0, len(unique_array)):
        indexs = unique_inverse_x == i
        py_x_current = np.mean(label[indexs, :], axis=0)
        p_y_given_x.append(py_x_current)
    p_y_given_x = np.array(p_y_given_x).T
    b_y = np.ascontiguousarray(label).view(np.dtype((np.void, label.dtype.itemsize * label.shape[1])))
    unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
    np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
    pys1 = unique_counts_y / float(np.sum(unique_counts_y))
    return pys, pys1, p_y_given_x, b1, b, unique_a, unique_inverse_x, unique_inverse_y, pxs

def get_information(activation,ws, x, label, num_of_bins, interval_information_display, \
                    model, layerSize, py_hats=0,multiple_bins=True,kde=False,
                    per_layer_bins=False, lower=False, maxentropy=True):
    """Calculate the information for the network for all the epochs and all the layers"""
    pys, pys1, p_y_given_x, b1, b, unique_a, unique_inverse_x, unique_inverse_y, pxs = extract_probs(label, x)
    np.set_printoptions(edgeitems=30)
    if per_layer_bins:
        multiple_bins=per_layer_bins
    print('Start calculating the information...')

    if(activation==0 and maxentropy==False): #tanh
        multiple_bins=False
        bins = np.linspace(-1, 1, num_of_bins)
        label = np.array(label).astype(np.float)
        params = np.array(Parallel(n_jobs=NUM_CORES
            )(delayed(calc_information_for_epoch)
            (i, interval_information_display, multiple_bins, ws[i], bins, unique_inverse_x, unique_inverse_y,
             label,
             b, b1, len(unique_a), pys,
             pxs, p_y_given_x, pys1, model.save_file, x.shape[1], layerSize,num_of_bins=num_of_bins)
            for i in range(len(ws))))
        return params
    elif (activation==1 or activation==3 or (activation==0 and maxentropy==True)): #relu
        if (kde==False):
            max_val,bins=set_up_bins(ws,activation,num_of_bins,multiple_bins=multiple_bins,per_layer_bins=per_layer_bins,maxentropy=maxentropy)
            if multiple_bins:
                params = np.array(Parallel(n_jobs=NUM_CORES
                                           )(delayed(calc_information_for_epoch)
                                           (i, interval_information_display, ws[i], bins[i], unique_inverse_x, unique_inverse_y,
                                            label,
                                            b, b1, len(unique_a), pys,
                                            pxs, p_y_given_x, pys1, model.save_file, x.shape[1], layerSize,
                                            num_of_bins=num_of_bins, per_layer_bins=per_layer_bins)
                                           for i in range(len(ws))))
            else:
                params = np.array(Parallel(n_jobs=NUM_CORES
                                           )(delayed(calc_information_for_epoch)
                                           (i, interval_information_display, ws[i], bins, unique_inverse_x, unique_inverse_y,
                                            label,
                                            b, b1, len(unique_a), pys,
                                            pxs, p_y_given_x, pys1, model.save_file, x.shape[1], layerSize,
                                            num_of_bins=num_of_bins, per_layer_bins=per_layer_bins)
                                           for i in range(len(ws))))

        else:
            inputs=custom_zip(interval_information_display, ws, unique_inverse_x, unique_inverse_y,
                                        label,
                                        len(unique_a), pys,
                                        pxs, p_y_given_x, pys1, model.save_file,x,lower,len(ws))
            with contextlib.closing( Pool() ) as pool:
                params=pool.map(calc_kde_information_for_epoch,inputs)
        return params
    elif (activation==2): #softplus
        if (kde==False):
            max_val,bins=set_up_bins(ws,activation,num_of_bins,multiple_bins=multiple_bins,per_layer_bins=per_layer_bins,maxentropy=maxentropy)
            if multiple_bins:
                params = np.array(Parallel(n_jobs=NUM_CORES
                                           )(delayed(calc_information_for_epoch)
                                           (i, interval_information_display, ws[i], bins[i], unique_inverse_x, unique_inverse_y,
                                            label,
                                            b, b1, len(unique_a), pys,
                                            pxs, p_y_given_x, pys1, model.save_file, x.shape[1], layerSize,
                                            num_of_bins=num_of_bins, per_layer_bins=per_layer_bins)
                                           for i in range(len(ws))))
            else:
                params = np.array(Parallel(n_jobs=NUM_CORES
                                           )(delayed(calc_information_for_epoch)
                                           (i, interval_information_display, ws[i], bins, unique_inverse_x, unique_inverse_y,
                                            label,
                                            b, b1, len(unique_a), pys,
                                            pxs, p_y_given_x, pys1, model.save_file, x.shape[1], layerSize,
                                            num_of_bins=num_of_bins, per_layer_bins=per_layer_bins)
                                           for i in range(len(ws))))

        else:
            inputs=custom_zip(interval_information_display, ws, unique_inverse_x, unique_inverse_y,
                                        label,
                                        len(unique_a), pys,
                                        pxs, p_y_given_x, pys1, model.save_file,x,lower,len(ws))
            with contextlib.closing( Pool() ) as pool:
                params=pool.map(calc_kde_information_for_epoch,inputs)
        return params

def custom_zip(interval_information_display, ws, unique_inverse_x, unique_inverse_y,
              label, len_unique_a, pys, pxs, p_y_given_x, pys1, save_file,x,lower,len_ws):
    tlist=[]
    for i in range(len_ws):
        tlist.append((i,interval_information_display,ws[i],unique_inverse_x, unique_inverse_y,
                   label,len_unique_a, pys, pxs, p_y_given_x, pys1,save_file,x,lower))
    return tlist


def set_up_bins(ws,activation, num_of_bins, multiple_bins=True, plot_hist=False,
                per_layer_bins=False,maxentropy=False):
    if (plot_hist==True):
        for i in range(len(ws)):
            if(i%50==0):
                print(i)
            for j in range(len(ws[i])):
                if(i%50==0):
                    print(j+1)
                    data=np.squeeze(ws[i][j].reshape(1, -1))
                    mx=max(data)
                    hist,_ =np.histogram(data,np.linspace(0,mx,30))
                    plt.hist(hist)
                    plt.title("Histogram")
                    plt.xlabel("Value")
                    plt.ylabel("Frequency")
                    plt.show()
    if multiple_bins:
            if per_layer_bins:
                max_val=find_per_layer_max_value(ws)
                #print("Max ReLU values:",max_val)
                bins=[]
                if activation==3 or activation==0:
                    low=-1.000001
                else:
                    low=0
                if maxentropy:
                    for i in range(len(ws)):
                        epoch_bins=[]
                        for j in range(len(ws[i])):
                            layer_bins=[low]
                            unique=np.unique(np.squeeze(ws[i][j].reshape(1,-1)))
                            unique=np.sort(np.setdiff1d(unique,layer_bins))
                            if unique.size>0:
                                for k in range(num_of_bins):
                                    ind=int(k*(unique.size/num_of_bins))
                                    layer_bins.append(unique[ind])
                                layer_bins.append(unique[-1])
                            epoch_bins.append(np.asarray(layer_bins))
                        bins.append(epoch_bins)
                else:
                    for i in range(len(max_val)):
                        epoch_bins=[]
                        for j in range(len(max_val[i])):
                            epoch_bins.append(np.linspace(low, max_val[i][j], num_of_bins))
                        bins.append(np.asarray(epoch_bins))
            else:
                max_val=find_max_value(ws)
                #print("Max ReLU values:",max_val)
                bins=[]
                for i in range(len(max_val)):
                    bins.append(np.linspace(0, max_val[i], num_of_bins))
    else:
            max_val=find_one_max_value(ws)
            print("Max ReLU values:",max_val)
            bins=np.linspace(0, max_val, num_of_bins)
    print('binning is done')
    bins=np.asarray(bins)
    return max_val,bins

def find_per_layer_max_value(ws):
    max_val=[]
    for i in range(len(ws)):
        epch=[]
        for j in range(len(ws[i])):
            epch.append(max(np.squeeze(ws[i][j].reshape(1,-1))))
        max_val.append(epch)
    for i in range(len(max_val)):
        for j in range(len(max_val)):
            if j>i:
                for k in range(len(max_val[i])):
                    if max_val[j][k]<max_val[i][k]:
                        max_val[j][k]=max_val[i][k]
    return max_val


def find_min_value(ws):
    min=[]
    for i in range(len(ws)):
        min.append(1)
        for j in range(len(ws[i])):
            for k in range(len(ws[i][j])):
                for l in range(len(ws[i][j][k])):
                    if ws[i][j][k][l]<min[i]:
                        min[i]=ws[i][j][k][l]
    for i in range(len(min)):
        for j in range(len(min)):
            if (min[j]>min[i] and j>i):
                min[j]=min[i]
    return min

def find_one_min_value(ws):
    min=1
    for i in range(len(ws)):
        for j in range(len(ws[i])):
            for k in range(len(ws[i][j])):
                for l in range(len(ws[i][j][k])):
                    if ws[i][j][k][l]<min:
                        min=ws[i][j][k][l]
    return min

def find_max_value(ws):
    max=[]
    for i in range(len(ws)):
        max.append(0)
        for j in range(len(ws[i])):
            for k in range(len(ws[i][j])):
                for l in range(len(ws[i][j][k])):
                    if ws[i][j][k][l]>max[i]:
                        max[i]=ws[i][j][k][l]
    for i in range(len(max)):
        for j in range(len(max)):
            if (max[j]<max[i] and j>i):
                max[j]=max[i]
    return max

def find_one_max_value(ws):
    max=0
    for i in range(len(ws)):
        for j in range(len(ws[i])):
            for k in range(len(ws[i][j])):
                for l in range(len(ws[i][j][k])):
                    if ws[i][j][k][l]>max:
                        max=ws[i][j][k][l]
    return max
