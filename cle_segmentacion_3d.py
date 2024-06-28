
#from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import numpy as np
#from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
import tifffile
import stackview
import io

file_path_volumen=r'\Users\ulabceriani\Documents\ITBA\ITBA_zt2\mitocondrias_segmentadas_zt2.tif'
input_image = tifffile.imread(file_path_volumen)

# Print the shape of the resulting 3D array
print(f"Shape of the volume: {input_image.shape}")

import pyclesperanto_prototype as cle

# List all available devices
all_devices = cle.available_device_names()
print("Available devices:", all_devices)

# Select the best device (example: an RTX GPU)
selected_device = cle.select_device('RTX')
print("Selected device:", selected_device)
# select a specific OpenCL / GPU device and see which one was chosen
cle.select_device('RTX')

input_gpu = cle.push(input_image[:,:,:])
print("Image size in GPU: " + str(input_gpu.shape))

def show(image_to_show, labels=False):
    """
    This function generates three projections: in X-, Y- and Z-direction and shows them.
    """
    projection_x = cle.maximum_x_projection(image_to_show)
    projection_y = cle.maximum_y_projection(image_to_show)
    projection_z = cle.maximum_z_projection(image_to_show)

    fig, axs = plt.subplots(1, 3, figsize=(15, 15))
    cle.imshow(projection_x, plot=axs[0], labels=labels)
    cle.imshow(projection_y, plot=axs[1], labels=labels)
    cle.imshow(projection_z, plot=axs[2], labels=labels)
    plt.show()

show(input_gpu)
print(input_gpu.shape)

###########################
#If you do not have isotropic pixels or need to perform background corrections
#follow the tutorials from here...
# https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/segmentation/Segmentation_3D.ipynb
###########################

#Segmentation
segmented = cle.voronoi_otsu_labeling(input_gpu, spot_sigma=13, 
                                      outline_sigma=1)
show(segmented, labels=True)

segmented_array = cle.pull(segmented)

#Wrtie image as tif. Ue imageJ for visualization
from skimage.io import imsave
imsave(r'\Users\ulabceriani\Documents\ITBA\ITBA_ZT2\segmented_3d_mito_fullvolume.tif', segmented_array) 
'''

# Write dataset as multi-dimensional OMETIFF *image*
#Use ZEN or any other scientific image visualization s/w
from apeer_ometiff_library import io

# Expand image array to 5D of order (T, Z, C, X, Y)
# This is the convention for OMETIFF format as written by APEER library
final = np.expand_dims(segmented_array, axis=0)
final = np.expand_dims(final, axis=0)

final=np.swapaxes(final, 2, 1)

final = final.astype(np.int8)

print("Shape of the segmented volume is: T, Z, C, X, Y ", final.shape)
print(final.dtype)

# Write dataset as multi-dimensional OMETIFF *image*
io.write_ometiff("segmented_multi_channel.ome.tiff", final)
'''