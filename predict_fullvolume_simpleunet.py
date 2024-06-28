# https://youtu.be/q-p8v1Bxvac

"""
Author: Dr. Sreenivas Bhattiprolu

Multiclass semantic segmentation using U-Net - prediction on large images
and 3D volumes (slice by slice)

To annotate images and generate labels, you can use APEER (for free):
www.apeer.com 
"""
from simple_unet_model import simple_unet_model 

from keras.utils import normalize
import os
import glob
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify


n_classes=1 #Number of classes for segmentation
IMG_HEIGHT = 256
IMG_WIDTH  = 256
IMG_CHANNELS = 1
patch_size=256

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()

model.load_weights('/Users/catalinainsussarry/Downloads/simple_multiplied_unet_epoch50.hdf5')

segm_images = []
#path = "all_images/*.tif"
file_path_volumen='/Users/catalinainsussarry/Documents/multiplied_zt2.tif'
file_path_mask='/Users/catalinainsussarry/Documents/mitocondrias_segmentadas_zt2.tif'
volumen_tiff = Image.open(file_path_volumen)
print(volumen_tiff.n_frames)

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
    volumen_tiff.seek(k)
    large_image=volumen_tiff.copy()
    large_image=np.array(large_image)
    large_image_cut, remaining_height, remaining_width = cut_image(large_image)
    patches = patchify(large_image_cut, (256, 256), step=256)  #Step=256 for 256 patches means no overlap
    #patches = np.expand_dims(normalize(np.array(patches), axis=1),3)
    predicted_patches = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            #print(i,j)
            
            single_patch = patches[i,j,:,:]
            if single_patch.shape==(256,256):

                single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
                print(single_patch_norm.shape)
                single_patch_shape = single_patch_norm.shape[:2]
                single_patch_norm=single_patch_norm[:,:,0][:,:,None]
                single_patch_input=np.expand_dims(single_patch_norm, 0)
                single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.2).astype(np.uint8)
                #single_patch_input = np.expand_dims(single_patch_norm, 0)
                #single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
                #segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])        
                predicted_patches.append(cv2.resize(single_patch_prediction, single_patch_shape[::-1]))
        
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
                single_patch_shape = single_patch_norm.shape[:2]
                single_patch_norm=single_patch_norm[:,:,0][:,:,None]
                single_patch_input=np.expand_dims(single_patch_norm, 0)
                single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.2).astype(np.uint8)
                #single_patch_input = np.expand_dims(single_patch_norm, 0)
                #single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
                #segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])        
                predicted_patches_borde_abajo.append(cv2.resize(single_patch_prediction, single_patch_shape[::-1]))

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
                single_patch_shape = single_patch_norm.shape[:2]
                single_patch_norm=single_patch_norm[:,:,0][:,:,None]
                single_patch_input=np.expand_dims(single_patch_norm, 0)
                single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.2).astype(np.uint8)
                #single_patch_input = np.expand_dims(single_patch_norm, 0)
                #single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
                #segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])        
                predicted_patches_borde_lateral.append(cv2.resize(single_patch_prediction, single_patch_shape[::-1]))

    predicted_patches_borde_lateral = np.array(predicted_patches_borde_lateral)

    predicted_patches_borde_lateral_reshaped = np.reshape(predicted_patches_borde_lateral, (patches_borde_lateral.shape[0], patches_borde_lateral.shape[1], 256,256) )

    reconstructed_image_borde_lateral = unpatchify(predicted_patches_borde_lateral_reshaped, (large_image_cut.shape[0],256))
    reconstructed_whole_image[int(remaining_height/2):int(large_image.shape[0]-remaining_height/2),large_image.shape[1]-256:] = reconstructed_image_borde_lateral
    reconstructed_whole_image[reconstructed_whole_image == 1] = 255  # Cambiar los píxeles con valor 1 a 120
    #reconstructed_whole_image[reconstructed_whole_image == 2] = 255
    kernel_morf = np.ones((5,5), np.uint8)
    eroded_image = cv2.erode(reconstructed_whole_image, kernel_morf, iterations=1)
    reconstructed_whole_image = cv2.dilate(eroded_image, kernel_morf, iterations=1)
    segm_images.append(reconstructed_whole_image)
    
    
final_segm_image = np.array(segm_images).astype(np.uint8)   

from tifffile import imsave
imsave('/Users/catalinainsussarry/Downloads/simple_multiplied_test_zt14.tif', final_segm_image)
    

    
    
    
    
    
    
    
    
    
    
    