#!/usr/bin/env python
# coding: utf-8

########## USAGE:
# cd
# source venv_jupyter/bin/activate
# cd Desktop/calibration
# python3 calibration.py
##########

# # CALIBRATION

# To analyze the level of ignition of each single LED it is first of all necessary to calibrate the software in order to identify the area to be controlled.
# 
# Each LED must be correctly identified and catalogued to determine the maximum light intensity and the image area it occupies.
# 
# Eventualities to take into account:
#     - all 16 LEDs must be found in the calibration file
#     - there must not be overlapping areas of separate LEDs
#     - check that the image is not saturated

# ### 1 Importing libraries

import numpy as np
import os
import glob
import cv2 as cv

from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter

from LEDMinMax import compute_zero, compute_roi_stats
from ROIDetection import split_leds, find_and_refine_rois, relabel_rois_grid
from common import show_images, show_hist, show_stats, show_saturated, save_calib_data, save_image


# ### 2 Loading images file .npy

# saturation values
SAT_VALUE     = 4095
SAT_THRESHOLD = 0.01                       # warning se > 1 % dei pixel è saturo

# erosion of ROIs to achieve the number we want
TARGET_ROIS    = 16
MAX_ITERATIONS = 20


#imgs = np.load("CalibImagesF2_Float62.npy")                     # no saturation, LEDs overlay
#imgs = np.load("CalibImagesFloat62.npy")                        # saturation, no LEDs overlay
directory = "data"
percorsi_file = sorted(glob.glob(os.path.join(directory, "*.npy")))

imgs_dict = {}
for f in percorsi_file:
    nome = os.path.splitext(os.path.basename(f))[0]  # nome file senza estensione
    imgs_dict[nome] = np.load(f)

#imgs = imgs_dict["CalibImagesFloat62"]
imgs = imgs_dict["LEDs_ON"]
imgs = imgs/16                                                  # se immagine è UINT16
print(f"Loaded: dtype={imgs.dtype}, shape={imgs.shape}")


# ### 3 Images Check

img_list = [imgs[i].astype(np.uint16) for i in range(imgs.shape[0])]
#show_images(img_list, title='Immagini originali')


# Histograms of each single image to identify the distribution of pixel values, mean and standard deviation are also indicated

#show_hist(img_list)


# Stampa per ogni immagine:
# - i -> indice immagine
# - p.min() -> valore minimo pixel
# - p.max() -> valore massimo pixel
# - p.mean() -> media intensità pixel
# - p.std() -> deviazione standard (variabilità)


show_stats(img_list)


# If we have both LEDs on and LEDs off images in the same set, we can see that while the average remains almost stable in all photos, the standard deviation is much higher in photos containing the LEDs turned on.

# ### 4 Filtering and Warnings

# Take only images with std higher than 300
# 
# K-Means Clustering is an unsupervised clustering algorithm that:
# - takes the data (here: the standard deviations of the images)
# - decides how many groups you want (here: n_clusters=2)
# - splits the data into 2 groups by minimizing the within-cluster distance
# 
# The algorithm groups similar std by starting from randomly initialized centroids and iteratively refining them by assigning each std to the nearest centroid and recomputing the centroid of each cluster. This process continues until the within-cluster sum of squared distances is minimized and the clustering converges.


#img_list = [img for img in img_list if img.ravel().astype(np.float64).std() > 300]

print(list(imgs_dict.keys())[:5])  # nomi reali delle chiavi

if "LEDs_OFF" in imgs_dict:
    # dividi per 16 subito, sia ON che OFF, una sola volta
    stack_on  = imgs_dict["LEDs_ON"]  / 16
    stack_off = imgs_dict["LEDs_OFF"] / 16

    img_no_LEDs = [stack_off[i] for i in range(stack_off.shape[0])] if stack_off.ndim == 3 else [stack_off]
    img_LEDs    = [stack_on[i]  for i in range(stack_on.shape[0])]

    print("Trovato 'LEDs_OFF.npy', uso quello come riferimento senza LED.")
else:
    img_LEDs, img_no_LEDs = split_leds(img_list)
    img_LEDs    = [img / 16 for img in img_LEDs]
    img_no_LEDs = [img / 16 for img in img_no_LEDs]

print(f"Numero frame LEDs_ON: {len(img_LEDs)}, shape singolo frame: {img_LEDs[0].shape}, dtype: {img_LEDs[0].dtype}")


# ##### (De-activated)
# Images may contain sensor noise and small intensity fluctuations. A median filter is applied in order to remove mpulsive noise while preserving edges.

#images_filtered = [median_filter(img, size=7).astype(np.uint16) for img in img_list]
#show_images(images_filtered, title='Immagini filtrate (median 7x7)')


# #### SATURATION WARNING

# Controllare la percentuale di pixel saturi nelle immagini ottenute dal KMeans. Se la saturazione è sopra il 10% del totale dei pixel, visualizzare un warning

#print(f"{'Photo':<10} {'Sat. pixels':>14} {'Tot. pixels':>14} {'Sat. %':>10}  Status")
#print("─" * 60)

for i, img in enumerate(img_LEDs):
    total_pixels     = img.size
    saturated_pixels = int(np.sum(img >= SAT_VALUE))
    sat_pct          = saturated_pixels / total_pixels * 100

    status = "WARNING - saturazione elevata!" if sat_pct > SAT_THRESHOLD * 100 else "OK"

    #print(f"{i:<10} {saturated_pixels:>14d} {total_pixels:>14d} {sat_pct:>9.3f}%  {status}")



#sat_images = show_saturated(img_LEDs, SAT_VALUE, SAT_THRESHOLD)

images_filtered = [
    cv.bilateralFilter(img.astype(np.float32), d=9, sigmaColor=150, sigmaSpace=75).astype(np.uint16)
    for img in img_LEDs
]

#show_images(images_filtered, title='Immagini filtrate (bilateral)')



# Apply Gaussian blur to the already median-filtered images
images_filtered_gaussian = [gaussian_filter(img.astype(np.uint16), sigma=2) for img in images_filtered]

# N.B.: più è alto il sigma, più grande è l'area del kernel

#show_images(images_filtered_gaussian, title='Immagini filtrate (median 7x7 + gaussian blur σ=2)')

show_stats(images_filtered_gaussian)


# ### 5 Otsu Thresholding

# To separate the LEDs from the background, the images are binarized using Otsu’s method.  
# Otsu’s method is preferable to manual thresholding because it automatically selects a threshold that maximizes the separation between foreground and background.
# 
# The correct threshold is identified as follows.
# 
# 
# Given:
# - i = 1,...,L gray levels
# - N number of pixels
# - h(i): $i^{th}$ entry of image histogram
# - p(i) = $\frac{i}{N}$ probability of gray lvl i
# 
# And also mean and variance:
# $$
# \mu = \sum_{i=1}^{L} i*p(i)
# $$
# $$
# \sigma^2 = \sum_{i=1}^{L} (i - \mu)^2*p(i)
# $$
# 
# The mean can be thought of as the expectation of i, while the variance as the expectation of square error from mean.
# 
# The threshold will split two regions: a "bright" region and a "dark" region.
# 
# Mean of dark region:
# $$
# \mu_1 = \sum_{i=1}^{t} i \cdot \frac{h(i)}{N_1},
# \qquad
# \text{given t = threshold, and } N_1 = \sum_{i=1}^{t} h(i), \text{tot dark pixels}
# $$
# $$
# \mu_1 = \sum_{i=1}^{t} i \cdot \frac{h(i)}{N_1} * \frac{N}{N} =
# $$
# $$
# = \sum_{i=1}^{t} i \cdot \frac{h(i)}{N} * \frac{N}{N_1} =
# \qquad
# $$
# $$
# \text{we take } \frac{N_1}{N} = q_1 \text{ ---> } \frac{\sum_{i=1}^{t} h(i)}{N} = \sum_{i=1}^{t} p(i)
# $$
# $$
# = \sum_{i=1}^{t} i \cdot \frac{i*p(i)}{q_1}
# $$
# 
# Variance of dark region:
# $$
# \sigma_1^2= \sum_{i=1}^{t} \frac{(i - \mu)^2*p(i)}{q_1}
# $$
# 
# We need to do this for each possible threshold:
# $$
# \mu_1(t)= \sum_{i=1}^{t} i \cdot \frac{i*p(i)}{q_1(t)}, 
# \sigma_1(t)^2= \sum_{i=1}^{t} \frac{(i - \mu)^2*p(i)}{q_1(t)}
# $$
# $$
# \mu_2(t)= \sum_{i=1}^{t} i \cdot \frac{i*p(i)}{q_2(t)}, 
# \sigma_2(t)^2= \sum_{i=1}^{t} \frac{(i - \mu)^2*p(i)}{q_2(t)}
# $$
# 
# Within-group variance of two regions is defined as:
# $$
# \sigma_w(t)^2 = q_1(t)*\sigma_1(t)^2 + q_2(t)*\sigma_2(t)^2
# $$
# It is the minimum when all pixel in dark/bright regions looks similar between pixels of same region
# 
# This produces an initial binary mask of the LEDs.

images_otsu = []
original_thresholds = []

for i, img in enumerate(images_filtered_gaussian):
    threshold, otsu_img = cv.threshold(img, 0, 4095, cv.THRESH_BINARY + cv.THRESH_OTSU)
    images_otsu.append(otsu_img)
    original_thresholds.append(threshold)
    print(f'Photo {i} — threshold: {threshold:.1f}')

#show_images(images_otsu, title="Otsu's thresholding")

# Convert Otsu images to 8-bit masks (bitwise_and requires uint8 mask)
binary_masks = [(img > 0).astype(np.uint8) * 255 for img in images_otsu]

# Apply mask to each original image
masked_images = [cv.bitwise_and(img_LEDs[i], img_LEDs[i], mask=binary_masks[i]) 
                 for i in range(len(img_LEDs))]

#show_images(masked_images, title='Immagini con maschera Otsu applicate')

show_stats(masked_images)
# naturalmente le foto risulteranno ancora sature perchè la maschera è stata applicata alle immagini iniziali (non filtrate)


# ### 6 Flood Fill

# To identify the different regions in the same image we assign a label to each pixel in order to define to which object it is from


roi_list_final, binary_masks_final = find_and_refine_rois(masked_images, images_filtered_gaussian, MAX_ITERATIONS)

roi_list_final, label_maps = relabel_rois_grid(roi_list_final, n_rows=4, n_cols=4)


# ##### N.B.
# Con kernel (3,3) e iterations=1, ogni chiamata a cv.erode rimuove 1 pixel dal bordo di ogni ROI in tutte le direzioni (su, giù, sinistra, destra, e diagonali con connectivity=8).
# 
# Quindi dopo it iterazioni ogni ROI è rimpicciolita di it pixel su ogni lato, e la distanza tra due ROI adiacenti è aumentata di 2 * it pixel totali (un pixel per lato da ciascuna delle due ROI).

# Apply mask to each original image
masked_images = [cv.bitwise_and(img_LEDs[i], img_LEDs[i], mask=binary_masks_final[i]) 
                 for i in range(len(img_LEDs))]


# Visualizza le immagini con le ROI evidenziate
roi_images = []
for i, (img, rois) in enumerate(zip(masked_images, roi_list_final)):
    img_display = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    img_bgr     = cv.cvtColor(img_display, cv.COLOR_GRAY2BGR)

    for roi in rois:
        cv.rectangle(img_bgr, (roi['x'], roi['y']), (roi['x'] + roi['w'], roi['y'] + roi['h']), (0, 255, 0), 2)
        cv.putText(img_bgr, str(roi['label']), (int(roi['cx']), int(roi['cy'])),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    roi_images.append(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))

#show_images(roi_images, title='Flood Fill — ROI rilevate')


# 2) indicare quanto si sovrappongono
# 5) visualizzare pixel che sono in comune nelle zone led in diverse maschere

# ### 7 Saturazione in ogni ROI

sat_images_roi = show_saturated(img_LEDs, SAT_VALUE, SAT_THRESHOLD, binary_masks_final, roi_list_final, label_maps)


# ### 8 Zero di riferimento

# Dalle immagini a led spenti ottenere uno stack medio dei frame e trovare uno zero di riferimento per ogni ROI.
# Applicare quindi la maschera ottenuta dalle immagini con i LED accesi (per semplicità viene usata la prima maschera individuata) e appliccarla alle immagini senza LED.
# Media dei valori dei pixel in ogni ROI per ottenere lo zero.


#ref_zero = compute_zero(img_no_LEDs, binary_masks_final, roi_list_final)

ref_zero, std_zero_roi = compute_roi_stats(img_no_LEDs, binary_masks_final, roi_list_final, tipo='zero')


# CALCOLO ZERO AGGIORNATO
# 1) per ogni immagine -> per ogni roi somma zeri
# 2) media di stesso roi in diverse immagini
# 3) calcolo varianza per ogni roi (valore da usare come soglia zero)
# 
# (stessa cosa per masssimo)

# ### 9 Valore Massimo

segnale_max, std_segnale_roi = compute_roi_stats(img_LEDs, binary_masks_final, roi_list_final, tipo='segnale')


save_calib_data(ref_zero, std_zero_roi, segnale_max, std_segnale_roi)


save_image(binary_masks_final[0], 'reference_mask.png')
save_image(sat_images_roi[0], 'calibrated_frame.png')


# calcolo valore led:
# 1) minimo e massimo calcolati sulle immagini originali sommando pixel in stesso roi (considera eventuale deviazione)
# 2) (valore pixel attuale - min)/(max-min)
# 3) bisogna trovare l'incertezza sul calcolo, non dovrebbe essere superiore a 1/50

# Per convertire da terminale un notebook Jupyter a file .py
# 
#     file -> Save and export Notebook As -> Executable script
# 
# Genera calibration.py dove ogni cella diventa un blocco di codice separato da commenti # In[N]:. È il modo più veloce