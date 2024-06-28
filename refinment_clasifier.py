#Second_clisifier
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
master_directory = "/Users/Milagros/Downloads/model_dir2" #poner path
#PATH = master_directory + '/' + 'epoch-24.pth'
PATH_post = master_directory + '/' + 'epoch-54.pth'

if torch.cuda.is_available():
    #model = MultiClass(out=2).to(device)
    #model.load_state_dict(torch.load(PATH))
    model_post = MultiClassPost(out=2).to(device)
    model_post.load_state_dict(torch.load(PATH_post))
else:
    #model = MultiClass(out=2)
    #model.load_state_dict(torch.load(PATH, map_location=device))
    model_post = MultiClassPost(out=2)
    model_post.load_state_dict(torch.load(PATH_post,map_location=device))

#model.eval()
model_post.eval()


x_labels=[4032.72602739726, 4264.490066225166, 2300.774548311076, 2498.69186949767, 2629.0845546786923, 2579.967122275582, 2513.4865013774106, 768.5501285347044, 2664.6295913088466, 2556.8789107763614, 1661.1205673758866, 2596.923197492163, 2540.284810126582, 1721.4642857142858, 1942.048611111111, 1141.3477812177503, 1502.3910149750416, 1739.8970331588132, 2398.3927710843373, 2319.495695514273, 667.1614123581336, 2002.93802262666, 765.640977443609, 2566.0542635658917, 2210.33984375, 2798.375939849624, 1835.8894736842105, 1750.4107933950866, 1464.2977011494254, 2482.499840916322, 870.0325138778746, 2194.5, 2270.177024482109, 2168.3144712430426, 1601.417225950783, 2330.798165137615, 2225.7045454545455, 1848.7947269303202, 1554.5, 1541.4904051172707, 1826.8690744920993, 2419.201451905626, 2241.061751732829, 1525.497308934338, 2047.4693333333332, 2385.2298969072167, 1961.7417142857144, 1857.9749121926743, 2275.8334612432845, 2074.7395833333335]
y_labels=[533.2465753424658, 604.5747398297068, 828.5659858601728, 833.0010357327809, 907.5552423900789, 962.0081270779459, 979.425344352617, 987.3393316195373, 1019.6125193998965, 1037.5706836616455, 1032.1583924349882, 1111.1543887147336, 1124.5, 1138.204081632653, 1138.4097222222222, 1185.7925696594427, 1184.324459234609, 1179.1029668411868, 1190.9307228915663, 1200.4748527412778, 1215.1443883984869, 1205.383669454009, 1212.951127819549, 1246.6298449612402, 1279.8919270833333, 1271.9473684210527, 1311.2006578947369, 1351.9146194120015, 1362.9586206896552, 1382.0299077314667, 1393.0471847739889, 1390.2379182156133, 1429.9566854990583, 1445.7291280148424, 1458.3238255033557, 1497.1077981651376, 1504.5, 1519.1939736346517, 1523.8457943925234, 1597.9211087420042, 1610.8306997742663, 1610.1833030852995, 1643.843100189036, 1650.945102260495, 1657.65, 1661.4268041237115, 1691.9291428571428, 1701.8976417461113, 1724.7820414428243, 1734.5]

print(len(x_labels))


x_labels_semifinal = []
y_labels_semifinal = []


window_size_post=200

# put padding on image
img=cv2.imread("/Users/Milagros/Desktop/zt14_slice50.jpg")
np_img = np.array(img)
if len(np_img.shape) > 2:
    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)

imagen_res=np.zeros_like(np_img)

np_img_padded = np.zeros((np_img.shape[0] + 200, np_img.shape[1] + 200))
np_img_padded[100:np_img.shape[0] + 100,
                100:np_img.shape[1] + 100] = np_img

# iterate over the detected vesicles
for det_ves in range(len(x_labels)):
    snapshot = np_img_padded[int(y_labels[det_ves]):
                                int(y_labels[det_ves]) + 200,
                                int(x_labels[det_ves]):
                                    int(x_labels[det_ves]) + 200]
    if (snapshot.shape[0] != window_size_post) or (
            snapshot.shape[1] != window_size_post):
        continue

    snapshot=cv2.resize(snapshot, (80, 80))
    snapshot = snapshot.reshape(1, snapshot.shape[0],
                                snapshot.shape[1])
    if np.max(snapshot) != np.min(snapshot):
        snapshot = (snapshot - np.min(snapshot)) / (
            np.max(snapshot) - np.min(snapshot))
    snapshot = (snapshot - 0.5) / 0.5
    snapshot = torch.from_numpy(snapshot)
    snapshot = snapshot.unsqueeze(0)

    # feed image patches into the second (refinement) classifier
    if torch.cuda.is_available():
        output = model_post.forward(snapshot.float().cuda())
        valuemax, preds = torch.max(output, 1)
        preds = preds.cpu()

    else:
        output = model_post.forward(snapshot.float())
        valuemax, preds = torch.max(output, 1)

    if preds == 1:
        x_labels_semifinal.append(x_labels[det_ves])
        y_labels_semifinal.append(y_labels[det_ves])

print(len(x_labels_semifinal))

for i in range(len(x_labels_semifinal)):
    imagen_res[int(y_labels_semifinal[i]),int(x_labels_semifinal[i])]=255
    imagen_res[int(y_labels_semifinal[i])+1,int(x_labels_semifinal[i])]=255
    imagen_res[int(y_labels_semifinal[i])-1,int(x_labels_semifinal[i])]=255
    imagen_res[int(y_labels_semifinal[i]),int(x_labels_semifinal[i])+1]=255
    imagen_res[int(y_labels_semifinal[i]),int(x_labels_semifinal[i])-1]=255

cv2.imwrite("/Users/Milagros/Desktop/second_class_zt14_slice50.jpg", imagen_res)
