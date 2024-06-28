# https://youtu.be/q-p8v1Bxvac

"""
Author: Dr. Sreenivas Bhattiprolu

Multiclass semantic segmentation using U-Net - prediction on large images
and 3D volumes (slice by slice)

To annotate images and generate labels, you can use APEER (for free):
www.apeer.com 
"""

from simple_multi_unet_model import multi_unet_model #Uses softmax (From video 208)

from keras.utils import normalize
import os
import glob
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
import tensorflow as tf


n_classes=3 #Number of classes for segmentation
IMG_HEIGHT = 256
IMG_WIDTH  = 256
IMG_CHANNELS = 1
patch_size=256

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

#model = get_model()
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()


#model.load_weights('multiclass_epoch20.hdf5')  
#model.load_weights('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')  

#model = tf.keras.models.load_model('/Users/catalinainsussarry/Library/Mobile Documents/com~apple~CloudDocs/Documents/PFC/Codigos seg 3d/multiclass_UNet_50epochs_B_focal.hdf5', compile=False)
from models import Attention_ResUNet, UNet, Attention_UNet, dice_coef, dice_coef_loss, jacard_coef
from focal_loss import BinaryFocalLoss

model = UNet((256,256,1), 3)
model.compile(optimizer='adam', loss=BinaryFocalLoss(gamma=2), metrics=['accuracy', jacard_coef])
model.load_weights('/Users/catalinainsussarry/Downloads/multiclass_UNet_50epochs_B_focal.hdf5')
model.summary()

segm_images = []
#path = "all_images/*.tif"
file_path_volumen='/Users/catalinainsussarry/Documents/zt22_TRIMVOL.tif'
volumen_tiff = Image.open(file_path_volumen)
print(volumen_tiff.n_frames)
'''
def prediction(model, image, patch_size):
    segm_img = np.zeros(image.shape[:2])  #Array with zeros to be filled with segmented values
    patch_num=1
    for i in range(0, image.shape[0], 256):   #Steps of 256
        for j in range(0, image.shape[1], 256):  #Steps of 256
            #print(i, j)
            single_patch = image[i:i+patch_size, j:j+patch_size]
            if single_patch.shape==(patch_size,patch_size):
                single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
                single_patch_shape = single_patch_norm.shape[:2]
                single_patch_input = np.expand_dims(single_patch_norm, 0)
                single_patch_prediction = (model.predict(single_patch_input)).astype(np.uint8)
                single_patch_predicted_img=np.argmax(single_patch_prediction, axis=3)[0,:,:]
                segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(single_patch_predicted_img, single_patch_shape[::-1])
                
            #print("Finished processing patch number ", patch_num, " at position ", i,j)
            patch_num+=1
    return segm_img
'''
import cv2
import numpy as np

def cut_image(image):
    # Get the dimensions of the image
    height, width = image.shape[:2]
    # Calculate the new dimensions that are divisible by 256
    new_height = height - (height % 256)
    new_width = width - (width % 256)
    # Cut the image to the new dimensions
    cut_image = image[:new_height, :new_width]
    return cut_image, height - new_height, width - new_width

def attach_remaining(image, remaining_height, remaining_width):
    # Get the dimensions of the image
    height, width = image.shape[:2]
    # Create an empty canvas with dimensions equal to the original image
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # Place the processed image onto the canvas
    canvas[:height - remaining_height, :width - remaining_width] = image
    return canvas

from pathlib import Path
for k in range(volumen_tiff.n_frames):
    #print(file)     #just stop here to see all file names printed
    #name = Path(file).stem #Get the original file name
    #print(name)
    #large_image = cv2.imread(file, 0)
    volumen_tiff.seek(k)
    large_image=volumen_tiff.copy()
    large_image=np.array(large_image)
    large_image_cut, remaining_height, remaining_width = cut_image(large_image)
    patches = patchify(large_image_cut, (256, 256), step=256)  #Step=256 for 256 patches means no overlap
    
    predicted_patches = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            #print(i,j)
            
            single_patch = patches[i,j,:,:]
            if single_patch.shape==(256,256):
                
                single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
                single_patch_input=np.expand_dims(single_patch_norm, 0)
        
                single_patch_prediction = (model.predict(single_patch_input))
                single_patch_predicted_img=np.argmax(single_patch_prediction, axis=3)[0,:,:]
        
                predicted_patches.append(single_patch_predicted_img)
    
    predicted_patches = np.array(predicted_patches)
    
    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 256,256) )
    
    reconstructed_image = unpatchify(predicted_patches_reshaped, large_image_cut.shape)
    reconstructed_whole_image=np.zeros(large_image.shape)
    reconstructed_whole_image[:large_image_cut.shape[0], :large_image_cut.shape[1]] = reconstructed_image
    #aplico prediction al borde de abajo
    patches_borde_abajo = patchify(large_image[large_image.shape[0]-256:,int(remaining_width/2):int(large_image.shape[1]-remaining_width/2)], (256, 256), step=256)  #Step=256 for 256 patches means no overlap
    predicted_patches_borde_abajo = []
    for i in range(patches_borde_abajo.shape[0]):
        for j in range(patches_borde_abajo.shape[1]):
            #print(i,j)
            
            single_patch = patches_borde_abajo[i,j,:,:]
            if single_patch.shape==(256,256):
                
                single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
                single_patch_input=np.expand_dims(single_patch_norm, 0)
        
                single_patch_prediction = (model.predict(single_patch_input))
                single_patch_predicted_img=np.argmax(single_patch_prediction, axis=3)[0,:,:]
        
                predicted_patches_borde_abajo.append(single_patch_predicted_img)

    predicted_patches_borde_abajo = np.array(predicted_patches_borde_abajo)

    predicted_patches_borde_abajo_reshaped = np.reshape(predicted_patches_borde_abajo, (patches_borde_abajo.shape[0], patches_borde_abajo.shape[1], 256,256) )

    reconstructed_image_borde_abajo = unpatchify(predicted_patches_borde_abajo_reshaped, (256,large_image_cut.shape[1]))
    reconstructed_whole_image[large_image.shape[0]-256:, int(remaining_width/2):int(large_image.shape[1]-remaining_width/2)] = reconstructed_image_borde_abajo

    #aplico prediction al borde lateral
    patches_borde_lateral = patchify(large_image[int(remaining_height/2):int(large_image.shape[0]-remaining_height/2),large_image.shape[1]-256:], (256, 256), step=256)  #Step=256 for 256 patches means no overlap
    predicted_patches_borde_lateral = []
    for i in range(patches_borde_lateral.shape[0]):
        for j in range(patches_borde_lateral.shape[1]):
            #print(i,j)
            
            single_patch = patches_borde_lateral[i,j,:,:]
            if single_patch.shape==(256,256):
                
                single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
                single_patch_input=np.expand_dims(single_patch_norm, 0)
        
                single_patch_prediction = (model.predict(single_patch_input))
                single_patch_predicted_img=np.argmax(single_patch_prediction, axis=3)[0,:,:]
        
                predicted_patches_borde_lateral.append(single_patch_predicted_img)

    predicted_patches_borde_lateral = np.array(predicted_patches_borde_lateral)

    predicted_patches_borde_lateral_reshaped = np.reshape(predicted_patches_borde_lateral, (patches_borde_lateral.shape[0], patches_borde_lateral.shape[1], 256,256) )

    reconstructed_image_borde_lateral = unpatchify(predicted_patches_borde_lateral_reshaped, (large_image_cut.shape[0],256))
    reconstructed_whole_image[int(remaining_height/2):int(large_image.shape[0]-remaining_height/2),large_image.shape[1]-256:] = reconstructed_image_borde_lateral

    segm_images.append(reconstructed_whole_image)
    print("Finished segmenting image: ", k)
    
    
final_segm_image = np.array(segm_images).astype(np.uint8)   

from tifffile import imsave
imsave('/Users/catalinainsussarry/Downloads/multiclass_test_zt22_unet_epoch50.tif', final_segm_image)
    

    
    
    
    
    
    
    
    
    
    
    