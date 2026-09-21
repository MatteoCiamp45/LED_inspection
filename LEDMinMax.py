####### LEDMinMax.py #######
## Library with functions to compute zero and max values for each ROI, using the mean of all images
## with LEDs off for zero and with LEDs on for max, and to compute std for each ROI in different images. 

import numpy as np

# Function to compute zero (sum of all pixels inside each ROI) from the image achieved with the mean of all LEDs_ON images
def compute_zero(img_no_LEDs, binary_masks_final, roi_list_final):
    # 1) Immagine media dalle foto a LED spenti
    mean_dark_img = np.mean(np.stack(img_no_LEDs, axis=0), axis=0)  # shape (H, W), float64
    # 2) Maschera e ROI di riferimento
    reference_mask = binary_masks_final[0]   # uint8, 0/255
    reference_rois = roi_list_final[0]
    print(f"Immagine media calcolata su {len(img_no_LEDs)} foto a LED spenti")
    print(f"ROI di riferimento: {len(reference_rois)} (da Photo 0)\n")
    # 3) Zero di riferimento per ogni ROI = SOMMA dei pixel dentro la maschera
    print(f"{'ROI':<6} {'Zero di riferimento':>15} {'N pixel':>10}  Posizione (cx, cy)")
    print("─" * 60)
    ref_zero = []
    for roi in reference_rois:
        # estrae coordinate e dimensioni del bounding box della ROI
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
        # ritaglia dall'immagine media (LED spenti) il rettangolo corrispondente alla ROI
        patch      = mean_dark_img[y:y+h, x:x+w]
        # ritaglia la stessa regione dalla maschera binaria di riferimento
        mask_patch = reference_mask[y:y+h, x:x+w]
        # converte la maschera in booleano: True dove il pixel appartiene effettivamente alla ROI (non solo al bounding box rettangolare)
        pixel_mask = mask_patch > 0
        n_pixel    = np.sum(pixel_mask)
        # somma i valori dei soli pixel mascherati, "zero di riferimento" per la ROI
        sum_zero   = np.sum(patch[pixel_mask])

        ref_zero.append({
            'label': roi['label'],
            'cx': roi['cx'], 'cy': roi['cy'],
            'zero': sum_zero,
            'n_pixel': n_pixel
        })

        print(f"{roi['label']:<6} {sum_zero:>15.3f} {n_pixel:>10d}  ({roi['cx']:.1f}, {roi['cy']:.1f})")

    return ref_zero


# Function to compute zero or max value, differently from the previous function it computes a sum inside each ROI and then a mean and std for each ROI in different images
def compute_roi_stats(images, binary_masks_final, roi_list_final, tipo='zero'):

    if tipo == 'zero':
        label_foto   = "Numero foto a LED spenti"
        chiave_val   = 'zero'
        header_val   = 'Zero di riferimento'
    elif tipo == 'segnale':
        label_foto   = "Numero foto a LED accesi"
        chiave_val   = 'segnale'
        header_val   = 'Valore Massimo'
    else:
        raise ValueError("tipo deve essere 'zero' o 'segnale'")

    # 1) Maschera e ROI di riferimento (stesse ROI, posizione fisica dei LED invariata)
    reference_mask = binary_masks_final[0]   # uint8, 0/255
    reference_rois = roi_list_final[0]
    print(f"{label_foto}: {len(images)}")
    print(f"ROI di riferimento: {len(reference_rois)} (da Photo 0)\n")

    # 2) Somma dei pixel per ogni ROI, per ogni singola immagine
    somma_per_roi = {roi['label']: [] for roi in reference_rois}
    for img_idx, img in enumerate(images):
        for roi in reference_rois:
            x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
            label = roi['label']
            # ritaglia dall'immagine corrente il rettangolo della ROI
            patch      = img[y:y+h, x:x+w].astype(np.float64)
            # ritaglia la stessa regione dalla maschera binaria di riferimento
            mask_patch = reference_mask[y:y+h, x:x+w]
            pixel_mask = mask_patch > 0
            sum_val = np.sum(patch[pixel_mask])
            somma_per_roi[label].append(sum_val)

    # 3) Media e std per ogni ROI (calcolate sulle N immagini)
    risultati = []
    std_roi   = []
    print(f"{'ROI':<6} {header_val:>15} {'Std':>15} {'N pixel':>10}  Posizione (cx, cy)")
    print("─" * 70)
    for roi in reference_rois:
        label      = roi['label']
        valori     = np.array(somma_per_roi[label])   # valori della ROI su tutte le immagini
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
        mask_patch = reference_mask[y:y+h, x:x+w]
        n_pixel    = int(np.sum(mask_patch > 0))
        media_val  = np.mean(valori)
        standard_dev = np.std(valori)

        risultati.append({
            'label': label,
            'cx': roi['cx'], 'cy': roi['cy'],
            chiave_val: media_val,
            'n_pixel': n_pixel,
            'n_immagini': len(valori)
        })
        std_roi.append({
            'label': label,
            'std': standard_dev
        })

        print(f"{label:<6} {media_val:>15.3f} {standard_dev:>15.3f} {n_pixel:>10d}  ({roi['cx']:.1f}, {roi['cy']:.1f})")

    return risultati, std_roi