#Clustering over pmap
#Me devuelve una lista con los centros de cada vesicula, separa si quedaron vesiculas juntas, y filtra por tamaño si quedaron cosas mal predichas y muy chicas


import numpy as np
from matplotlib.figure import Figure
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.backends.backend_tkagg as tkagg
import seaborn as sns
from tkinter import filedialog
import xlsxwriter
import pandas as pd
from time import sleep
import os
from pathlib import Path
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import cv2
from scipy import ndimage
import PIL
from PIL import Image, ImageOps
from CNNs_GaussianNoiseAdder import MultiClass, MultiClassPost

pixel_size_final=25
proc_pmap=cv2.imread("/Users/Milagros/Desktop/pmap_zt2_slice98.jpg")
proc_pmap_gray = cv2.cvtColor(proc_pmap, cv2.COLOR_BGR2GRAY)

# Convertir la imagen a un array numpy
proc_pmap = np.array(proc_pmap_gray)

imagen_res=np.zeros_like(proc_pmap)
map=np.zeros_like(proc_pmap)
map[proc_pmap>=10]=255



labelarray, counts = ndimage.measurements.label(map)
print(counts)
x_labels = []
y_labels = []



# iterate over the found objects (labelarray)
for i in range(counts):
    x,y = np.where(labelarray == i + 1)
    #print(x,y)
    
    temp = []
    if type(x) == int:
        temp.append(x, y)
    else:
        for j in range(len(x)):
            temp.append((x[j], y[j]))
    try:
        euc_distances = euclidean_distances(temp, temp)

    except MemoryError:
        break
    
    
    # check what is the most likely number of vesicles in each object
    # this will define the number of clusters to use in Kmeans
    max_distance = np.max(euc_distances)
    numb_clusters = 1

    # if the max distance in an object > 25, check for peaks 
    if max_distance > 95:
        peaks = []
        for j in range(len(x)):
            temp_peak = proc_pmap[x[j], y[j]]
            temp_peak_idx = (x[j], y[j])
            challenge_peak = np.zeros((8))
            gapx = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
            gapy = np.array([-1, 0, 1, -1, 1, -1, 0, 1])

            # look around temp_peak to check if it is a peak
            for g in range(8):
                challenge_peak[g] = proc_pmap[x[j] + gapx[g],
                                                y[j] + gapy[g]]
            if (np.max(challenge_peak) <= temp_peak):
                peaks.append(temp_peak_idx)

        # calculate the distance between the peaks,
        # the number peaks with distance > 15 will define n_clusters
        if len(peaks) > 1:
            print("peaks")
            print(len(peaks))
            euc_distances = euclidean_distances(peaks, peaks)
            euc_distances[np.where(euc_distances == 0.0)] = 10000
            numb_clusters = 0
            very_close = []
            for e in range(euc_distances.shape[0]):
                if np.min(euc_distances[e, :]) <= 50:
                    very_close.append(np.min(euc_distances[e, :]))
                else:
                    numb_clusters += 1

            very_close = list(dict.fromkeys(very_close))
            print(numb_clusters)
            print(len(very_close))
            numb_clusters = numb_clusters + len(very_close)
    numb_clusters=2

    kmeans = KMeans(n_clusters=numb_clusters, n_init=10).fit(temp)

    # arbitrary minimal cluster dimention in pixel: 64
    if pixel_size_final < 2.3:
        min_cluster = 64
    # correction minimal cluster dimension if pixel size >= 2.3 nm
    elif pixel_size_final < 3.3:
        min_cluster = 79
    elif pixel_size_final < 4.3:
        min_cluster = 94
    elif pixel_size_final < 5.3:
        min_cluster = 109
    elif pixel_size_final < 6.3:
        min_cluster = 124
    else:
        min_cluster = 340 #cambiar por 300

    # check again the distance between peaks (centers of each cluster)
    if numb_clusters > 1:
        euc_distances = euclidean_distances(kmeans.cluster_centers_,
                                            kmeans.cluster_centers_)
        euc_distances[np.where(euc_distances == 0.0)] = 10000
        potential_numb_vesicles = 0
        very_close = []
        for e in range(euc_distances.shape[0]):
            if np.min(euc_distances[e, :]) < 50:
                very_close.append(np.min(euc_distances[e, :]))
            if np.min(euc_distances[e, :]) >= 50:
                potential_numb_vesicles += 1
        very_close = list(dict.fromkeys(very_close))
        potential_numb_vesicles = potential_numb_vesicles + len(
            very_close)

        # if not each cluster is considered a vesicle
        if potential_numb_vesicles < numb_clusters:
            cluster_size = []
            cluster_label = []
            for k in range(numb_clusters):
                cluster_size.append((kmeans.labels_ == k).sum())
                cluster_label.append(k)
            clu = list(zip(cluster_size, cluster_label))
            clu.sort(reverse=True)
            clu = clu[:potential_numb_vesicles]

            # exclude cluster smaller then min_cluster
            for k in range(len(clu)):
                if (kmeans.labels_ == clu[k][1]).sum() > min_cluster:
                    x_labels.append(
                        kmeans.cluster_centers_[clu[k][1]][1])
                    y_labels.append(
                        kmeans.cluster_centers_[clu[k][1]][0])

        # if each cluster is cosidered a vesicle
        else:
            for k in range(len(kmeans.cluster_centers_)):

                # exclude cluster smaller then min_cluster
                if (kmeans.labels_ == k).sum() > min_cluster:
                    x_labels.append(kmeans.cluster_centers_[k][1])
                    y_labels.append(kmeans.cluster_centers_[k][0])

    # if there is only one cluster
    else:
        for k in range(len(kmeans.cluster_centers_)):

            # if each cluster is cosidered a vesicle
            if (kmeans.labels_ == k).sum() > min_cluster:
                x_labels.append(kmeans.cluster_centers_[k][1])
                y_labels.append(kmeans.cluster_centers_[k][0])

for i in range(len(x_labels)):
    imagen_res[int(y_labels[i]),int(x_labels[i])]=255
    imagen_res[int(y_labels[i])+1,int(x_labels[i])]=255
    imagen_res[int(y_labels[i])-1,int(x_labels[i])]=255
    imagen_res[int(y_labels[i]),int(x_labels[i])+1]=255
    imagen_res[int(y_labels[i]),int(x_labels[i])-1]=255

cv2.imwrite("/Users/Milagros/Desktop/center_zt2_slice98_2.jpg", imagen_res)


#print(x_labels)
#print(y_labels)
