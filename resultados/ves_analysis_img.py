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
master_directory = "/Users/Milagros/Downloads/model_dir" #poner path
PATH = master_directory + '/' + 'epoch-24.pth'
#PATH_post = self.master.directory + '/' + 'model_post.pth'
if torch.cuda.is_available():
    model = MultiClass(out=2).to(device)
    model.load_state_dict(torch.load(PATH))
    #self.model_post = MultiClassPost(out=2).to(device)
    #self.model_post.load_state_dict(torch.load(PATH_post))
else:
    model = MultiClass(out=2)
    model.load_state_dict(torch.load(PATH, map_location=device))
    #self.model_post = MultiClassPost(out=2)
    #self.model_post.load_state_dict(torch.load(PATH_post,map_location=device))

model.eval()
#self.model_post.eval()

#Cargar imagen
#resizear *10 y fg

sliding_size = 10
window_size = 100
img=cv2.imread("/Users/Milagros/Desktop/zt2_slice118.jpg")
np_img = np.array(img)
if len(np_img.shape) > 2:
    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
print(np_img.shape)

p_map = np.zeros((int(np_img.shape[0] / sliding_size),
                          int(np_img.shape[1] / sliding_size)))
#p_map = np.zeros((int((np_img.shape[0]- window_size) / sliding_size)+1,int((np_img.shape[1]-window_size)/ sliding_size)+1))
#20*50=451


#imagen de 200x500 
#El procesamiento de abajo me va a dar 451 patches = (((500-100)/10)+1)*(((200-100)/10)+1)!
#patches_folder = os.path.join(master_directory, "patches")
#if not os.path.exists(patches_folder):
#    os.makedirs(patches_folder)

patch_counter = 0

for x in range(0, np_img.shape[1], sliding_size):
    #print('Processing image number: 1')

    # iterate over image.shape[0] in steps of size == sliding_size
    for y in range(0, np_img.shape[0], sliding_size):
        snapshot = np_img[y :y + window_size,
                                    x :x + window_size]
        patch=snapshot

        if (snapshot.shape[0] != window_size) or (
                snapshot.shape[1] != window_size):
            continue
        snapshot=cv2.resize(snapshot, (40, 40))
        snapshot = snapshot.reshape(1, 40, 40)
        if np.max(snapshot) != np.min(snapshot):
            snapshot = (snapshot - np.min(snapshot)) / (
                np.max(snapshot) - np.min(snapshot))
        snapshot = (snapshot - 0.5) / 0.5
        snapshot = torch.from_numpy(snapshot)
        snapshot = snapshot.unsqueeze(0)

        if torch.cuda.is_available():
            output = model.forward(snapshot.float().cuda())
            valuemax, preds = torch.max(output, 1)
            valuemin, _ = torch.min(output, 1)
            valuemax = valuemax.cpu()
            valuemin = valuemin.cpu()
            preds = preds.cpu()
        else:
            output = model.forward(snapshot.float())
            valuemax, preds = torch.max(output, 1)
            valuemin, _ = torch.min(output, 1)
        
        if preds == 1:
            valuemax = valuemax.data.numpy()
            valuemin = valuemin.data.numpy()
            pvalue = np.exp(valuemax) / (np.exp(valuemax) + np.exp(
                valuemin))
            p_map[int((y + 50) / sliding_size),
                          int((x + 50) / sliding_size)] = pvalue

        #print("Clase predicha:", preds.item())
        #patch_filename = f"pred{preds.item()}_snapshot{patch_counter}.jpg"
        #patch_path = os.path.join(patches_folder, patch_filename)
        #cv2.imwrite(patch_path, patch)
        patch_counter += 1

#cv2.imwrite("/Users/Milagros/Downloads/map/pmap1.jpg", p_map)
print(patch_counter)
proc_pmap = cv2.resize(p_map, (np_img.shape[1], np_img.shape[0])) #que quede del mismo tamaño de la imagen
proc_pmap = cv2.blur(proc_pmap, (3, 3))

if np.max(proc_pmap) > 0:
    proc_pmap = (proc_pmap / (np.max(proc_pmap))) * 255

# set a threshold for proc_map (below 20% of 255, pixel=0)
for xx in range(proc_pmap.shape[0]):
    for yy in range(proc_pmap.shape[1]):
        if proc_pmap[xx, yy] < 255 / 100 * 20:
            proc_pmap[xx, yy] = 0

cv2.imwrite("/Users/Milagros/Desktop/pmap_zt2_slice118.jpg", proc_pmap)