import numpy as np

# Per una singola foto (indicata da idx), calcola quanto segnale (luminosità/intensità) c'è in ogni ROI
def compute_val(images, binary_masks_final, roi_list_final, idx=0):

    label_foto   = "Numero foto test"
    chiave_val   = 'segnale'
    header_val   = 'Valore attuale'
    # 1) Maschera e ROI di riferimento (stesse ROI, posizione fisica dei LED invariata)
    reference_mask = binary_masks_final[0]   # uint8, 0/255
    reference_rois = roi_list_final[0]
    #print(f"{label_foto}: {len(images)}")
    #print(f"ROI di riferimento: {len(reference_rois)} (da Photo 0)\n")

    # 2) Somma dei pixel per ogni ROI
    img = images[idx]  # singola immagine di testing
    somma_per_roi = {}
    #print(f"{'ROI':<6} {header_val:>15} {'N pixel':>10}  Posizione (cx, cy)")
    #print("─" * 70)
    for roi in reference_rois:
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
        label = roi['label']
        # ritaglia dall'immagine il rettangolo della ROI
        patch      = img[y:y+h, x:x+w].astype(np.float64)
        # ritaglia la stessa regione dalla maschera binaria di riferimento
        mask_patch = reference_mask[y:y+h, x:x+w]
        pixel_mask = mask_patch > 0
        sum_val = np.sum(patch[pixel_mask])
        n_pixel = int(np.sum(pixel_mask))
        somma_per_roi[label] = sum_val
        #print(f"{label:<6} {sum_val:>15.3f} {n_pixel:>10d}  ({roi['cx']:.1f}, {roi['cy']:.1f})")

    # 3) Lista dei risultati, un elemento per ogni ROI
    risultati = [
        {
            'label': roi['label'],
            'cx': roi['cx'], 'cy': roi['cy'],
            chiave_val: somma_per_roi[roi['label']],
            'n_pixel': int(np.sum(reference_mask[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']] > 0))
        }
        for roi in reference_rois
    ]

    return risultati


# Funzione per calcolare la posizione temporale del segnale in base ai valori normalizzati e agli stati del LED a
# partire dall'inizio della sequenza. Se è l'ultimo LED la sequenza dura solo 1.2 microsecondi
def sequence_position(normalized_values, states, interval=2.2, last_interval=1.2, overlap=0.2):
    step = interval - overlap  # 2.0 µs: distanza tra l'accensione di due led successivi

    intermedi = [i for i, st in enumerate(states) if st == 'intermedio']
    last_int = intermedi[-1]+1 if intermedi else None
    #print(f"Ultimo led {last_int}")

    if not intermedi:
        return 0

    if len(intermedi) == 1:
        for i, st in enumerate(states):
            if st == 'intermedio' and states[i-1] == 'ZERO' and last_int != 16:
                res = float(i * step + (1 - normalized_values[i]) * interval)
                continue
            elif st == 'intermedio' and states[i-1] == 'ZERO' and last_int == 16:
                res = float(i * step + (1 - normalized_values[i]) * last_interval)
                continue
            elif st == 'intermedio' and states[i-1] == 'MAX' and last_int != 16:
                res = float(i * step + normalized_values[i] * interval)
                continue
            elif st == 'intermedio' and states[i-1] == 'MAX' and last_int == 16:
                res = float(i * step + normalized_values[i] * last_interval)
                continue

        return res

    for i, st in enumerate(states):
        if st == 'intermedio':
            if states[i+1] == 'intermedio' and normalized_values[i+1] > normalized_values[i] and last_int != 16:
                res = (i+1) * step + (1 - normalized_values[i+1]) * interval
                res2 = i * step + (1 - normalized_values[i]) * interval
                continue
            elif states[i+1] == 'intermedio' and normalized_values[i+1] > normalized_values[i] and last_int == 16:
                res = (i+1) * step + (1 - normalized_values[i+1]) * last_interval
                res2 = i * step + (1 - normalized_values[i]) * interval
                continue
            elif states[i+1] == 'intermedio' and normalized_values[i+1] < normalized_values[i] and last_int != 16:
                res = (i+1) * step + normalized_values[i+1] * interval
                res2 = i * step + normalized_values[i] * interval
                continue
            elif states[i+1] == 'intermedio' and normalized_values[i+1] < normalized_values[i] and last_int == 16:
                res = (i+1) * step + normalized_values[i+1] * last_interval
                res2 = i * step + normalized_values[i] * interval
                continue
    print(f"Differenza calcolo tra primo e secondo led intermedi: {res - res2}")

    return res


#    intermedi = [i for i, st in enumerate(states) if st == 'intermedio']

#    if not intermedi:
#        return None  # nessun led intermedio: snapshot non valido

    # ogni led intermedio dà una stima indipendente della stessa posizione t
#    stime = [i * step + normalized_values[i] * interval for i in intermedi]
#    stime = [i * step + interval * (1 - normalized_values[i]) for i in intermedi]

#    return sum(stime) / len(stime)  # media (1 valore se isolato, 2 se in overlap)
