#!/usr/bin/env python
# coding: utf-8

# # TESTING

# #### 1 Importing libraries

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
import os

from common import show_images
from ROIDetection import find_rois, relabel_rois_grid
from LEDtest import compute_val, sequence_position

# #### 1 Extracting testing images

directory = "test"
percorsi_file = sorted(glob.glob(os.path.join(directory, "*.npy")))

if len(percorsi_file) == 0:
    raise FileNotFoundError(f"Nessun file .npy trovato nella cartella '{directory}'")
elif len(percorsi_file) > 1:
    print(f"Attenzione: trovati {len(percorsi_file)} file .npy, uso il primo: {os.path.basename(percorsi_file[0])}")

f = percorsi_file[0]
nome = os.path.splitext(os.path.basename(f))[0]
imgs = np.load(f)
imgs = imgs / 16   # se immagine è UINT16

print(f"Caricato: '{nome}', dtype={imgs.dtype}, shape={imgs.shape}")
#show_images(imgs, title='Immagini originali')


# #### 2 Extracting mask and file with min and max values

# --- Caricamento maschera di riferimento ---
mask_path = os.path.join("reference_mask.png")
reference_mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

if reference_mask is None:
    raise FileNotFoundError(f"Maschera non trovata in: {mask_path}")

print(f"Maschera caricata: dtype={reference_mask.dtype}, shape={reference_mask.shape}")

# --- Caricamento dati di calibrazione ---
calib_path = os.path.join("calib_data.txt")
if not os.path.isfile(calib_path):
    raise FileNotFoundError(f"File di calibrazione non trovato in: {calib_path}")
calib_data = np.loadtxt(calib_path)
# se il file ha una sola riga, np.loadtxt restituisce un array 1D: lo forzo a 2D
if calib_data.ndim == 1:
    calib_data = calib_data.reshape(1, -1)

ref_zero        = calib_data[:, 0].tolist()
std_zero_roi    = calib_data[:, 1].tolist()
segnale_max     = calib_data[:, 2].tolist()
std_segnale_roi = calib_data[:, 3].tolist()

print(f"Dati di calibrazione caricati: {calib_data.shape[0]} righe (ROI)")
print(f"{'ROI':<6} {'ref_zero':>15} {'std_zero':>15} {'segnale_max':>15} {'std_segnale':>15}")
print("─" * 70)
for i in range(len(calib_data)):
    print(f"{i:<6} {ref_zero[i]:>15.3f} {std_zero_roi[i]:>15.3f} {segnale_max[i]:>15.3f} {std_segnale_roi[i]:>15.3f}")


# #### 3 Applying mask to testing images

# Apply the same mask to each original image
masked_images = [cv.bitwise_and(img, img, mask=reference_mask) 
                 for img in imgs]

show_images(masked_images, title='Immagini con maschera Otsu applicate')


# #### 4 Flood fill

roi_list_final, binary_masks_final = find_rois(masked_images, imgs)

roi_list_final, label_maps = relabel_rois_grid(roi_list_final, n_rows=4, n_cols=4)

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

show_images(roi_images, title='Flood Fill — ROI rilevate')


# #### 5 Compute value for each LED

####################################################################################################################
# METODO 1: calcolo del valore normalizzato e arrotondato a 0.02, con etichettatura dello stato (ZERO, MAX, intermedio)
# L'obiettivo è quello di avere un'incertezza non superiore a 1/50 (0.02)
# for j, img in enumerate(imgs):
#     results = compute_val(imgs, binary_masks_final, roi_list_final, j)
    
#     step = 0.01
    
#     print(f"\n{'='*60}")
#     print(f"Photo {j}")
#     print(f"{'='*60}")
#     print(f"{'ROI':<6} {'Norm':>10} {'Arrotondato':>14}  Stato")
#     print("─" * 60)
    
#     for i, r in enumerate(results):
#         valore_attuale = r['segnale']
        
#         min_val = ref_zero[i]
#         max_val = segnale_max[i]
        
#         norm_value = (valore_attuale - min_val) / (max_val - min_val)
        
#         # arrotonda al multiplo di 0.02 più vicino
#         norm_rounded = round(norm_value / step) * step
#         norm_rounded = round(norm_rounded, 2)  # arrotonda valore a 2 cifre decimali
        
#         if norm_rounded <= 0.00:
#             stato = "ZERO"
#         elif norm_rounded >= 1.00:
#             stato = "MAX"
#         else:
#             stato = "intermedio"
        
#         r['norm_value']    = norm_value
#         r['norm_rounded']  = norm_rounded
#         r['stato']         = stato
        
#         print(f"{r['label']:<6} {norm_value:>10.4f} {norm_rounded:>14.2f}  {stato}")

####################################################################################################################
# METODO 2: calcolo del valore normalizzato e arrotondato a 0.02, con etichettatura dello stato (ZERO, MAX, intermedio) e calcolo della posizione temporale nella sequenza
# N_SIGMA = 15
# reference_mask = binary_masks_final[0]

# for j, img in enumerate(imgs):

#     print(f"\n{'='*60}")
#     print(f"Photo {j}")
#     print(f"{'='*60}")
#     print(f"{'ROI':<6} {'Val':>10} Stato")
#     print("─" * 60)
    
#     results = compute_val(imgs, binary_masks_final, roi_list_final,j)

#     states = []
#     normalized_values = []
#     for i, r in enumerate(results):
#         valore_attuale = r['segnale']
        
#         min_val = ref_zero[i]
#         max_val = segnale_max[i]
        
#         norm_value = (valore_attuale - min_val) / (max_val - min_val)
        
#         r['norm_value'] = norm_value
#         normalized_values.append(norm_value)
        
#         #print(f"ROI {r['label']}: {norm_value:.4f}")
            
#         zero_lower = ref_zero[i] - N_SIGMA * std_zero_roi[i]
#         zero_upper = ref_zero[i] + N_SIGMA * std_zero_roi[i]
        
#         max_lower = segnale_max[i] - N_SIGMA * std_segnale_roi[i]
#         max_upper = segnale_max[i] + N_SIGMA * std_segnale_roi[i]
        
#         in_zero_range = zero_lower <= valore_attuale <= zero_upper
#         in_max_range  = max_lower <= valore_attuale <= max_upper
        
#         if in_zero_range:
#             stato = "ZERO"
#         elif in_max_range:
#             stato = "MAX"
#         else:
#             stato = "intermedio"

#         states.append(stato)
#         roi['stato'] = stato
#         print(f"{r['label']:<6} {norm_value:>10.4f} {stato} [ {std_zero_roi[i]} , {std_segnale_roi[i]} ]")
#         #print(f"ROI {i+1}: valore={valore_attuale:.3f} -> {stato}")
#         #print(f"ROI {i}: valore={valore_attuale:.4f}  "
#         #      f"zero_band=({zero_lower:.4f},{zero_upper:.4f})  "
#         #      f"max_band=({max_lower:.4f},{max_upper:.4f})  "
#         #      f"std_zero={std_zero_roi[i]:.5f}  std_max={std_segnale_roi[i]:.5f}")

#     #position = sequence_position(normalized_values, states)
#     #print(f"Posizione nella sequenza in microsecondi: {position}")
    
   
# 0.2 : 2.2 = x : 1 -> x = 0.090

####################################################################################################################À
# METODO 3
# Questo codice serve a ricalcolare la deviazione standard (std) del "rumore" del segnale in due situazioni: quando il segnale è a ZERO e quando è al MASSIMO (MAX), usando però solo i dati "veri" osservati nelle foto reali.
# 
# Per etichettare in modo approssimativo quali foto sono chiaramente ZERO e quali chiaramente MAX, raccoglie i valori reali corrispondenti, e da questi ricalcola std_zero_roi_new e std_segnale_roi_new — cioè quanto "rumore" (variabilità) c'è realmente nei dati, misurato sul campo invece che stimato a priori.

# n_roi = len(ref_zero)
# zero_samples_per_roi = [[] for _ in range(n_roi)]
# max_samples_per_roi  = [[] for _ in range(n_roi)]

# # passata preliminare "grezza": usa il criterio del primo metodo (normalizzato)
# # per etichettare in modo affidabile quali frame sono ZERO/MAX per ogni ROI
# step = 0.02
# for j, img in enumerate(imgs):
#     results = compute_val(imgs, binary_masks_final, roi_list_final, j)       # valore grezzo del segnale
#     for i, r in enumerate(results):
#         val = r['segnale']
#         norm = (val - ref_zero[i]) / (segnale_max[i] - ref_zero[i])          # normalizzazione
#         norm_rounded = round(round(norm / step) * step, 2)                   # arrotondamento
#         if norm_rounded <= 0.00:
#             zero_samples_per_roi[i].append(val)
#         elif norm_rounded >= 1.00:
#             max_samples_per_roi[i].append(val)

# # ricalcola std usando SOLO i campioni "veri" osservati nella sequenza reale
# std_zero_roi_new = [
#     np.std(s) if len(s) > 1 else std_zero_roi[i]
#     for i, s in enumerate(zero_samples_per_roi)
# ]
# std_segnale_roi_new = [
#     np.std(s) if len(s) > 1 else std_segnale_roi[i]
#     for i, s in enumerate(max_samples_per_roi)
# ]

# N_SIGMA = 10  # ora puoi anche provare a scendere, es. 8-10, dato che gli std riflettono il rumore reale

# reference_mask = binary_masks_final[0]

# for j, img in enumerate(imgs):
#     print(f"\n{'='*60}")
#     print(f"Photo {j}")
#     print(f"{'='*60}")
#     print(f"{'ROI':<6} {'Val':>10} Stato")
#     print(f"Intervallo")
#     print("─" * 60)
    
#     results = compute_val(imgs, binary_masks_final, roi_list_final, j)
#     states = []
#     normalized_values = []
    
#     for i, r in enumerate(results):
#         valore_attuale = r['segnale']  # fix: non più r.get('zero', r.get('segnale'))
        
#         min_val = ref_zero[i]
#         max_val = segnale_max[i]
        
#         norm_value = (valore_attuale - min_val) / (max_val - min_val)
        
#         r['norm_value'] = norm_value
#         normalized_values.append(norm_value)
        
#         zero_lower = ref_zero[i] - N_SIGMA * std_zero_roi_new[i]
#         zero_upper = ref_zero[i] + N_SIGMA * std_zero_roi_new[i]
        
#         max_lower = segnale_max[i] - N_SIGMA * std_segnale_roi_new[i]
#         max_upper = segnale_max[i] + N_SIGMA * std_segnale_roi_new[i]
        
#         in_zero_range = zero_lower <= valore_attuale <= zero_upper
#         in_max_range  = max_lower <= valore_attuale <= max_upper
        
#         if in_zero_range:
#             stato = "ZERO"
#         elif in_max_range:
#             stato = "MAX"
#         else:
#             stato = "intermedio"
        
#         states.append(stato)
#         r['stato'] = stato
        
#         print(f"{r['label']:<6} {norm_value:>10.4f} {stato} [ {std_zero_roi_new[i]} , {std_segnale_roi_new[i]} ]")
    
#     position = sequence_position(normalized_values, states)
#     print(f"Posizione nella sequenza in microsecondi: {position}")

####################################################################################################################
# METODO 4
# Sostituita la deviazione standard con una % di errore che consideriamo far rientrare il valore in massimo o zero

err = 0.03
reference_mask = binary_masks_final[0]

cartella = os.getcwd()
filepath = os.path.join(cartella, "timestamps.txt")

with open(filepath, 'w') as f:
    for j, img in enumerate(imgs):

        print(f"\n{'='*60}")
        print(f"Photo {j}")
        print(f"{'='*60}")
        print(f"{'ROI':<6} {'Val':>10} Stato Errore sul massimo")
        print("─" * 60)

        results = compute_val(imgs, binary_masks_final, roi_list_final,j)

        states = []
        normalized_values = []
        for i, r in enumerate(results):
            valore_attuale = r['segnale']

            min_val = ref_zero[i]
            max_val = segnale_max[i]

            norm_value = (valore_attuale - min_val) / (max_val - min_val)

            r['norm_value'] = norm_value
            normalized_values.append(norm_value)

            #print(f"ROI {r['label']}: {norm_value:.4f}")

            interval = err * segnale_max[i]

            zero_upper = ref_zero[i] + interval

            max_lower = segnale_max[i] - interval

            in_zero_range = valore_attuale <= zero_upper
            in_max_range  = max_lower <= valore_attuale

            if in_zero_range:
                stato = "ZERO"
            elif in_max_range:
                stato = "MAX"
            else:
                stato = "intermedio"

            states.append(stato)
            roi['stato'] = stato
            print(f"{r['label']:<6} {norm_value:>10.4f} {stato} {interval}")

        position = sequence_position(normalized_values, states)
        print(f"Posizione nella sequenza in microsecondi: {position}")

        f.write(f"{j+1} {position}\n")


# due intermedi sono possibili solo se tra 0.03 - 0.1 uno e 0.9 - 0.97 l'altro circa