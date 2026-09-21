####### common.py #######
## Library with common functions used to show images, histograms, stats,
## saturated pixels, save calibration data and save images.

import numpy as np
import os
import cv2 as cv
from matplotlib import pyplot as plt

from ROIDetection import remap_labels_map

# Function to show the images contained in the list passed as an argument
def show_images_notused(img_list, title='PHOTOS', vmax=None):
    n = len(img_list)
    fig, axes = plt.subplots(n, 1, figsize=(10, n * 8))
    axes = np.array(axes).flatten()
    plt.suptitle(title, fontsize=20)
    for i, img in enumerate(img_list):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Photo {i}', fontsize=12)
        axes[i].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def show_images(img_list, title='PHOTOS', vmax=None):
    # Prefer a scrollable column using Tkinter + Pillow when available.
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except Exception:
        # Fallback to matplotlib grid if tkinter/Pillow not available
        n = len(img_list)
        cols = 4
        rows = int(np.ceil(n / cols)) if n > 0 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        plt.suptitle(title, fontsize=20)
        axes = np.array(axes).flatten()
        imshow_kwargs = {'cmap': 'gray'}
        if vmax is not None:
            imshow_kwargs['vmax'] = vmax
        for i in range(rows * cols):
            ax = axes[i]
            if i < n:
                img = img_list[i]
                ax.set_title(f'Photo {i}', fontsize=12)
                ax.imshow(img, **imshow_kwargs)
                ax.axis('off')
            else:
                ax.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
        return

    # Create Tkinter window with vertical scrollbar
    root = tk.Tk()
    root.title(title)
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    win_w = int(screen_w * 0.6)
    win_h = int(screen_h * 0.8)
    root.geometry(f"{win_w}x{win_h}")

    container = tk.Frame(root)
    canvas = tk.Canvas(container, width=win_w, height=win_h)
    vsb = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)

    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    container.pack(fill="both", expand=True)

    images_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=images_frame, anchor='nw')

    photo_tk_refs = []

    # Determine target width for images (account for scrollbar)
    target_width = win_w - 30

    for i, img in enumerate(img_list):
        # Convert image to uint8 RGB PIL Image
        if isinstance(img, np.ndarray):
            arr = img.copy()
            if arr.dtype == np.float32 or arr.dtype == np.float64:
                # normalize floats to 0-255
                mn, mx = arr.min(), arr.max()
                if mx > mn:
                    arr = ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
            elif arr.dtype != np.uint8:
                arr = cv.normalize(arr.astype(np.float32), None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

            if arr.ndim == 2:
                pil_img = Image.fromarray(arr).convert('RGB')
            elif arr.shape[2] == 3:
                # assume RGB or BGR? OpenCV often uses BGR; our code usually passes RGB
                pil_img = Image.fromarray(arr)
            else:
                pil_img = Image.fromarray(arr[:, :, :3])
        else:
            # if already a PIL image
            pil_img = img

        # Resize to fit target width while keeping aspect ratio
        w, h = pil_img.size
        if w > target_width:
            new_h = int(h * (target_width / w))
            pil_resized = pil_img.resize((target_width, new_h), Image.LANCZOS)
        else:
            pil_resized = pil_img

        photo = ImageTk.PhotoImage(pil_resized)
        lbl = tk.Label(images_frame, image=photo)
        lbl.pack(padx=5, pady=5)
        title_lbl = tk.Label(images_frame, text=f'Photo {i}', font=('Arial', 12))
        title_lbl.pack()

        photo_tk_refs.append(photo)

    def _on_configure(event):
        canvas.configure(scrollregion=canvas.bbox('all'))

    images_frame.bind('<Configure>', _on_configure)

    # Allow mousewheel scrolling on most platforms
    def _on_mousewheel(event):
        if event.num == 5 or event.delta < 0:
            canvas.yview_scroll(1, 'unit')
        elif event.num == 4 or event.delta > 0:
            canvas.yview_scroll(-1, 'unit')

    canvas.bind_all('<MouseWheel>', _on_mousewheel)
    canvas.bind_all('<Button-4>', _on_mousewheel)
    canvas.bind_all('<Button-5>', _on_mousewheel)

    root.mainloop()


# Function to show images' histograms
def show_hist(img_list):
    n = len(img_list)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    plt.suptitle('Istogrammi', fontsize=20)
    axes = axes.flatten()
    for i, img in enumerate(img_list):
        ax = axes[i]
        # appiattisce l’immagine in un vettore di pixel
        pixels = img.ravel().astype(np.float64)
        # calcolo istogramma (divide il range tra min e max in 256 intervalli)
        counts, bin_edges = np.histogram(pixels, bins=256, range=(pixels.min(), pixels.max()))
        counts_norm = counts / counts.sum() # normalizzazione frequenze
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 # centro dei bin
        ax.plot(bin_centers, counts_norm, color='steelblue', linewidth=1)
        ax.fill_between(bin_centers, counts_norm, alpha=0.3, color='steelblue')
        # media e deviazione standard
        mean, std = pixels.mean(), pixels.std()

        ax.axvline(mean, color='red', linewidth=1.5, linestyle='--', label=f'μ={mean:.1f}')
        ax.axvline(mean + std, color='orange', linewidth=1, linestyle=':')
        ax.axvline(mean - std, color='orange', linewidth=1, linestyle=':', label=f'σ={std:.1f}')
        ax.set_title(f'Photo {i}', fontsize=11)
        ax.set_xlabel('Valore pixel', fontsize=9)
        ax.set_ylabel('Frequenza relativa', fontsize=9)
        ax.set_xlim(pixels.min(), pixels.max())
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=9)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# Function to show Min, Max and Mean
def show_stats(img_list):
    print(f"{'Photo':<10} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 52)
    for i, img in enumerate(img_list):
        p = img.ravel().astype(np.float64)
        print(f"{i:<10} {p.min():>10.2f} {p.max():>10.2f} {p.mean():>10.2f} {p.std():>10.2f}")
        #formattazione:
        # <10 = allineato a sinistra (colonna Photo)
        # >10.2f = float con 2 decimali, allineato a destra stessa cosa


# Function to show saturated pixelsshow_s
def show_saturated(img_LEDs, SAT_VALUE, SAT_THRESHOLD, binary_masks_final=None, roi_list_final=None, label_maps=None):
    sat_images = []
    for i, img in enumerate(img_LEDs):
        img_norm = cv.normalize(img.astype(np.float32), None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        img_rgb  = cv.cvtColor(img_norm, cv.COLOR_GRAY2RGB)

        if binary_masks_final is not None and roi_list_final is not None:
            mask = binary_masks_final[i]
            rois = roi_list_final[i]
            num_labels, labels_map, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)

            if label_maps is not None:
                labels_map = remap_labels_map(labels_map, label_maps[i])

            print(f"\n=== Photo {i} ===")
            for roi in rois:
                label = roi['label']
                roi_pixel_mask   = (labels_map == label)
                saturated_mask   = roi_pixel_mask & (img >= SAT_VALUE)
                img_rgb[saturated_mask, 0] = 255
                img_rgb[saturated_mask, 1] = 0
                img_rgb[saturated_mask, 2] = 0
                total_pixels = int(np.sum(roi_pixel_mask))
                sat_pct = saturated_mask.sum() / total_pixels * 100 if total_pixels > 0 else 0.0
                status  = "WARNING" if sat_pct > SAT_THRESHOLD * 100 else "OK"
                print(f"  ROI {label} — {sat_pct:.3f}% saturi  {status}")
                cv.rectangle(img_rgb, (roi['x'], roi['y']), (roi['x'] + roi['w'], roi['y'] + roi['h']), (0, 255, 0), 2)
                cv.putText(img_rgb, str(roi['label']), (int(roi['x']), int(roi['y'] - 3)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            saturated_mask = img >= SAT_VALUE
            img_rgb[saturated_mask, 0] = 255
            img_rgb[saturated_mask, 1] = 0
            img_rgb[saturated_mask, 2] = 0
            sat_pct = saturated_mask.sum() / img.size * 100
            status  = "WARNING" if sat_pct > SAT_THRESHOLD * 100 else "OK"
            print(f"Photo {i} — {sat_pct:.3f}% saturi  {status}")

        sat_images.append(img_rgb)

    title = 'Pixel saturi per ROI (rosso) con bounding box' if roi_list_final is not None else 'Pixel saturi evidenziati in rosso'
    show_images(sat_images, title=title)
    return sat_images

# Function to save calibration data inside a .txt file. Each row of the file will contain zero, std_zero, max, std_max
def save_calib_data(res_zero, std_zero, res_max, std_max,
                     filename='calib_data.txt'):
    """
    Salva su file (sovrascrivendo se esiste) una riga per ogni ROI con:
    ref_zero  std_zero_roi  segnale_max  std_segnale_roi
    """
    # cartella corrente del notebook
    cartella = os.getcwd()
    filepath = os.path.join(cartella, filename)

    # indicizzo std per label, per essere sicuro dell'abbinamento corretto
    std_zero_map     = {d['label']: d['std'] for d in std_zero}
    std_max_map  = {d['label']: d['std'] for d in std_max}
    max_map      = {d['label']: d['segnale'] for d in res_max}

    with open(filepath, 'w') as f:
        for r in res_zero:
            label = r['label']
            ref_zero        = r['zero']
            std_zero_roi    = std_zero_map[label]
            segnale_max     = max_map[label]
            std_segnale_roi = std_max_map[label]

            f.write(f"{ref_zero} {std_zero_roi} {segnale_max} {std_segnale_roi}\n")

    print(f"Dati salvati in: {filepath}")


def save_image(image, filename):
    cartella = os.getcwd()
    filepath = os.path.join(cartella, filename)

    if image.ndim == 3 and image.shape[2] == 3:
        image_to_save = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    else:
        image_to_save = image

    cv.imwrite(filepath, image_to_save)
    print(f"Immagine salvata in: {filepath}")