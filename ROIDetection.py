####### ROIDetection.py #######
## Library to find all the ROIs inside each image and to erode them in order to have 16 ROI,
## to relabel rois to respect the correct functioning order of the LEDs and to remap labels map.


import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans

# KMeans to separate clusters of img_LEDs and img_no_LEDs photos (to use only if they are in the same .npy file)
def split_leds(img_list):
    # Calcola le std di tutte le immagini
    stds = np.array([img.ravel().astype(np.float64).std() for img in img_list])
    # Trova la soglia come punto di minimo nella distribuzione delle std (kmeans con 2 cluster)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(stds.reshape(-1, 1))
    # La soglia è il punto medio tra i due centroidi
    c1, c2 = sorted(kmeans.cluster_centers_.flatten())
    threshold_auto = (c1 + c2) / 2
    print(f"Std per immagine: {np.round(stds, 1)}")
    print(f"Centroidi: {c1:.1f} (spente), {c2:.1f} (accese)")
    print(f"Soglia automatica: {threshold_auto:.1f}")
    # immagini che verranno processate (solo quelle con i LED accesi)
    img_LEDs = [img for img, s in zip(img_list, stds) if s > threshold_auto]
    print(f"Immagini con LED: {len(img_LEDs)}")
    # immagini per lo zero di riferimento
    img_no_LEDs = [img for img, s in zip(img_list, stds) if s <= threshold_auto]
    print(f"Immagini senza LED: {len(img_no_LEDs)}")

    return img_LEDs, img_no_LEDs


# Function to find all the ROIs inside each image and to erode them in order to have 16 ROI
def find_and_refine_rois(masked_images, images_filtered_gaussian, MAX_ITERATIONS, TARGET_ROIS=16):
    roi_list = []          # lista di liste: per ogni immagine, la lista delle ROI trovate
    binary_masks = []      # maschere binarie iniziali (servono per l'eventuale riduzione)

    for i, img in enumerate(masked_images):
        # Converti in uint8 per connectedComponents
        img_8bit = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        # Binarizza (i pixel già mascherati sono 0, gli altri > 0)
        _, binary = cv.threshold(img_8bit, 1, 255, cv.THRESH_BINARY)
        binary_masks.append(binary)

        # Flood fill tramite connectedComponentsWithStats
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary, connectivity=8)

        # Label 0 è il background, le ROI partono da 1
        num_rois = num_labels - 1
        print(f"Photo {i} — ROI trovate: {num_rois} {'Correct calibration frame' if num_rois == TARGET_ROIS else 'WRONG NUMBER OF LEDs'}")

        # Salva le ROI (stats escludendo il background)
        rois = []
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            cx, cy = centroids[label]
            rois.append({'label': label, 'x': x, 'y': y, 'w': w, 'h': h, 'area': area, 'cx': cx, 'cy': cy})

        roi_list.append(rois)

    # RIDUZIONE MASCHERE ROI (solo dove non sono state trovate 16 ROI)
    binary_masks_final = []
    roi_list_final = []

    for i, (mask, img) in enumerate(zip(binary_masks, images_filtered_gaussian)):
        # se questa immagine ha già 16 ROI, non serve erosione: mantieni i valori originali
        if len(roi_list[i]) == TARGET_ROIS:
            binary_masks_final.append(mask)
            roi_list_final.append(roi_list[i])
            continue

        current_mask = mask.copy()
        kernel       = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        prev_mask    = current_mask.copy()
        for it in range(MAX_ITERATIONS):
            prev_mask    = current_mask.copy()
            current_mask = cv.erode(current_mask, kernel, iterations=1)
            num_labels, _, stats, centroids = cv.connectedComponentsWithStats(current_mask, connectivity=8)
            n_rois = num_labels - 1
            print(f"  Photo {i}: ciclo {it+1:3d} -> {n_rois} ROI")
            if n_rois == TARGET_ROIS:
                print(f"Photo {i}: 16 ROI raggiunte")
                break
        else:
            print(f"Photo {i}: Limite iterazioni raggiunto, ROI finali: {n_rois}")

        rois = []
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            cx, cy = centroids[label]
            rois.append({'label': label, 'x': x, 'y': y, 'w': w, 'h': h, 'area': area, 'cx': cx, 'cy': cy})

        binary_masks_final.append(current_mask)
        roi_list_final.append(rois)

    return roi_list_final, binary_masks_final


# Function to relabel rois to respect the correct functioning order of the LEDs:
# 
# 13  9   5  1
# 
# 14  10  6  2
# 
# 15  11  7  3
# 
# 16  12  8  4

def relabel_rois_grid(roi_list_final, n_rows=4, n_cols=4):
    new_roi_list = []
    label_maps = []  # una mappatura {vecchio_id: nuovo_id} per ogni immagine

    for rois in roi_list_final:
        rois = list(rois)
        if len(rois) != n_rows * n_cols:
            print(f"Attenzione: trovate {len(rois)} ROI invece di {n_rows*n_cols}")

        cx = np.array([r['cx'] for r in rois])
        cy = np.array([r['cy'] for r in rois])

        x_sorted_idx = np.argsort(cx)
        col_bins = np.array_split(x_sorted_idx, n_cols)
        col_of = np.zeros(len(rois), dtype=int)
        for col_index, idxs in enumerate(col_bins):
            col_of[idxs] = col_index

        y_sorted_idx = np.argsort(cy)
        row_bins = np.array_split(y_sorted_idx, n_rows)
        row_of = np.zeros(len(rois), dtype=int)
        for row_index, idxs in enumerate(row_bins):
            row_of[idxs] = row_index

        mapping = {}
        for i, roi in enumerate(rois):
            col_index = col_of[i]
            row_index = row_of[i]
            old_id = roi['label']
            new_id = (n_cols - 1 - col_index) * n_rows + row_index + 1
            mapping[old_id] = new_id
            roi['label'] = new_id

        rois.sort(key=lambda r: r['label'])
        new_roi_list.append(rois)
        label_maps.append(mapping)

    return new_roi_list, label_maps


def remap_labels_map(labels_map, mapping):
    """Sostituisce i vecchi id dei componenti connessi con i nuovi, pixel per pixel."""
    new_labels_map = np.zeros_like(labels_map)
    for old_id, new_id in mapping.items():
        new_labels_map[labels_map == old_id] = new_id
    return new_labels_map