import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from pathlib import Path
from collections import defaultdict

# ==========================================
# KONFIGURASI
# ==========================================
SOURCE_DIR = "dataset_original"
OUTPUT_TRAIN_NPZ = "dataset_abjad_custom_train.npz"
OUTPUT_TEST_NPZ = "dataset_abjad_custom_test.npz"

VALID_CLASSES = list("abcdefghijklmnopqrstuvwxyz")
LABEL_MAP = {char: idx for idx, char in enumerate(VALID_CLASSES)}
JUMLAH_AUGMENTASI = 5

# Pipeline Albumentations Berbasis Rekomendasi Ilmiah LION Dataset
AUG_PIPELINE = A.Compose([
    # Opsi 1: Kombinasi Top 3 Eksperimen (Rotasi Konservatif, Shift Alami, & Downscaling)
    A.ShiftScaleRotate(
        shift_limit_x=(0.0, 0.05),    # Pergeseran horizontal ringan (~max 15px dari resolusi standar)
        shift_limit_y=(-0.02, 0.02),  # Pergeseran vertikal sangat tipis agar tidak keluar kanvas
        scale_limit=(-0.25, 0.0),     # HANYA Downscaling (0.75 hingga 1.0), tidak ada upscaling
        rotate_limit=(-1.5, 1.5),     # WAJIB KECIL: Batas aman agar tidak merusak skala karakter
        p=0.8,
        mode=cv2.BORDER_CONSTANT,
        cval=0                        # Latar belakang hitam biner
    ),
    
    # Opsi 2: Kemiringan Karakter (Shearing) yang Terbukti Bagus
    A.Affine(
        shear=(-15, 15),              # Menangani variasi tulisan miring/italic secara aman
        p=0.5,
        mode=cv2.BORDER_CONSTANT,
        cval=0
    ),
    
    # Opsi 3: Distorsi Fisik Elastis (Garis Tangan Bergelombang)
    A.ElasticTransform(
        alpha=20, 
        sigma=5,                      # Dioptimalkan ke angka 5 sesuai standar paper
        alpha_affine=10, 
        p=0.5,
        border_mode=cv2.BORDER_CONSTANT, 
        value=0
    )
])


# ==========================================
# FUNGSI PEMBANTU (Fase 3 - Fase 5)
# ==========================================
def extract_roi_and_resize(blurred_image, capture_debug=False):
    """
    Menangani Smart Crop, Square Canvas, dan Resizing.
    Dipisahkan agar bisa dipanggil berulang untuk gambar asli maupun hasil augmentasi.
    """
    debug_imgs = {}
    
    # Fase 3: Smart Crop (Pusatkan gambar sesuai kontur tulisan putih di atas latar hitam)
    contours, _ = cv2.findContours(blurred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    x_min, y_min = blurred_image.shape[1], blurred_image.shape[0]
    x_max, y_max = 0, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x + w), max(y_max, y + h)

    roi = blurred_image[y_min:y_max, x_min:x_max]
    h_roi, w_roi = roi.shape

    if h_roi == 0 or w_roi == 0:
        return None, None
        
    if capture_debug: debug_imgs['3_Cropped'] = roi.copy()

    # Fase 4: Centered Square Canvas
    max_dim = max(h_roi, w_roi)
    square_canvas = np.zeros((max_dim, max_dim), dtype=np.uint8)

    y_offset = (max_dim - h_roi) // 2
    x_offset = (max_dim - w_roi) // 2
    square_canvas[y_offset:y_offset+h_roi, x_offset:x_offset+w_roi] = roi
    
    if capture_debug: debug_imgs['4_Squared'] = square_canvas.copy()

    # Fase 5: Resize 24x24 + Padding -> 28x28
    resized_24 = cv2.resize(square_canvas, (24, 24), interpolation=cv2.INTER_CUBIC)
    final_28 = cv2.copyMakeBorder(resized_24, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    final_28 = cv2.normalize(final_28, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if capture_debug: debug_imgs['5_Final_28x28'] = final_28.copy()

    return final_28.flatten(), debug_imgs

# ==========================================
# FUNGSI UTAMA (Fase 0 - Fase 2 + Augmentasi Loop)
# ==========================================
def process_image_to_abjad(image_path, capture_debug=False):
    """
    Memproses 1 gambar mentah menjadi kumpulan array 1D (Asli + Augmentasi).
    """
    debug_images = {}
    features_list = [] # Penampung hasil (asli + augmentasi)
    aug_visuals = []   # Penampung visual khusus untuk debug hasil augmentasi
    
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [], None

    # Fase 0: Zoom In
    zoom_percent = 0.24  
    h, w = img.shape
    y_crop1, y_crop2 = int(h * zoom_percent), int(h * (1 - zoom_percent))
    x_crop1, x_crop2 = int(w * zoom_percent), int(w * (1 - zoom_percent))
    img = img[y_crop1:y_crop2, x_crop1:x_crop2]

    # Fase 1: Binarisasi & WAJIB Invert Latar Sejak Awal (Hitamkan Latar)
    # Ini memastikan crop dan centered canvas tidak kacau
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if capture_debug: debug_images['1_Binarized'] = binary.copy()

    # Fase 2: Gaussian Blur
    blurred = cv2.GaussianBlur(binary, (0, 0), sigmaX=1.3, sigmaY=1.3) 
    if capture_debug: debug_images['2_Blurred'] = blurred.copy()

    # --- EKSEKUSI DATA ASLI ---
    orig_features, orig_debug = extract_roi_and_resize(blurred, capture_debug)
    if orig_features is not None:
        features_list.append(orig_features)
        if capture_debug and orig_debug:
            debug_images.update(orig_debug)

    # --- FASE 2.5: EKSEKUSI AUGMENTASI ---
    for _ in range(JUMLAH_AUGMENTASI):
        # Terapkan augmentasi pada gambar biner/blur yang masih berukuran penuh
        augmented_blurred = AUG_PIPELINE(image=blurred)['image']
        
        # Simpan visual hasil augmentasi yang sudah di-resize ke 28x28
        aug_features, aug_debug = extract_roi_and_resize(augmented_blurred, capture_debug=capture_debug)
        
        if aug_features is not None:
            features_list.append(aug_features)
            # Simpan sampel variasi augmentasi untuk ditampilkan
            if capture_debug and aug_debug:
                aug_visuals.append(aug_debug['5_Final_28x28'])

    if capture_debug:
        debug_images['Augmentations_List'] = aug_visuals

    return features_list, debug_images

# ==========================================
# FUNGSI VISUALISASI
# ==========================================
def generate_step_visualizations(captured_samples):
    print("\nMembangun visualisasi untuk setiap fase...")
    for step_name, class_dict in captured_samples.items():
        fig, axes = plt.subplots(2, 13, figsize=(20, 4))
        fig.suptitle(f'Sampel Gambar Dataset — Fase: {step_name}', fontsize=14)

        for idx, huruf in enumerate(VALID_CLASSES):
            row = idx // 13
            col = idx % 13
            
            if huruf in class_dict:
                axes[row, col].imshow(class_dict[huruf], cmap='gray')
            else:
                axes[row, col].imshow(np.zeros((28,28)), cmap='gray')
                
            axes[row, col].set_title(huruf, fontsize=10)
            axes[row, col].axis('off')

        plt.tight_layout()
        nama_file = f'sampel_fase_{step_name}.png'
        plt.savefig(nama_file, dpi=150, bbox_inches='tight')
        print(f" [✓] Tersimpan: {nama_file}")
        plt.close(fig)

def generate_augmentation_comparison(sample_original, list_of_augmentations, class_name):
    """
    Menampilkan visualisasi perbandingan 1 Gambar Asli dengan semua variasi Augmentasinya.
    """
    print(f"\nMembangun visualisasi komparasi augmentasi untuk kelas '{class_name}'...")
    total_plots = 1 + len(list_of_augmentations)
    
    fig, axes = plt.subplots(1, total_plots, figsize=(3 * total_plots, 3))
    fig.suptitle(f"Perbandingan Geometri Spasial Kelas '{class_name.upper()}' (Asli vs Augmentasi)", fontsize=12, y=1.05)
    
    # Plot Gambar Asli
    axes[0].imshow(sample_original, cmap='gray')
    axes[0].set_title("Asli (1x)", fontsize=10, color='cyan', bbox=dict(facecolor='black', alpha=0.8))
    axes[0].axis('off')
    
    # Plot Gambar Hasil Augmentasi Loop
    for idx, aug_img in enumerate(list_of_augmentations):
        axes[idx + 1].imshow(aug_img, cmap='gray')
        axes[idx + 1].set_title(f"Aug {idx + 1}", fontsize=10)
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    nama_file = f'komparasi_augmentasi_{class_name}.png'
    plt.savefig(nama_file, dpi=150, bbox_inches='tight')
    print(f" [✓] Tersimpan: {nama_file}")
    plt.close(fig)

def plot_distributions(original_counts, train_counts, test_counts, total_augmented):
    """
    Membuat visualisasi komparasi Before-After distribusi data per kelas.
    
    Parameters:
    - original_counts: dict, jumlah gambar original per kelas
    - train_counts: dict, jumlah data train (original + augmented) per kelas
    - test_counts: dict, jumlah data test per kelas
    - total_augmented: dict, jumlah data augmented per kelas
    """
    print("\nMembangun visualisasi distribusi data (Before vs After)...")
    
    classes = sorted(original_counts.keys())
    x = np.arange(len(classes))
    width = 0.6
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [1, 1.5]})
    
    # ==========================================
    # GRAFIK ATAS: BEFORE AUGMENTATION
    # ==========================================
    original_values = [original_counts[c] for c in classes]
    
    bars_before = ax1.bar(x, original_values, width, 
                          label='Total Original Images', 
                          color='#3498db', 
                          edgecolor='#2c3e50', 
                          linewidth=0.8,
                          alpha=0.85)
    
    # Tambahkan anotasi nilai di atas setiap batang
    for i, (bar, value) in enumerate(zip(bars_before, original_values)):
        if value > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(original_values)*0.01,
                    f'{value}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax1.set_title('DISTRIBUSI DATA SEBELUM AUGMENTASI (Original Dataset)', 
                  fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
    ax1.set_xlabel('Kelas Abjad', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Jumlah Gambar', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, fontsize=10, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(original_values) * 1.15 if original_values else 1)
    
    # ==========================================
    # GRAFIK BAWAH: AFTER AUGMENTATION (STACKED)
    # ==========================================
    train_original_values = [original_counts[c] - test_counts.get(c, 0) for c in classes]
    train_augmented_values = [total_augmented.get(c, 0) for c in classes]
    test_values = [test_counts.get(c, 0) for c in classes]
    
    # Plot batang Train Original (biru)
    bars_train_ori = ax2.bar(x, train_original_values, width,
                             label='Train Original',
                             color='#3498db',
                             edgecolor='#2c3e50',
                             linewidth=0.8,
                             alpha=0.85)
    
    # Plot batang Train Augmented (biru muda dengan arsiran) - ditumpuk di atas Train Original
    bars_train_aug = ax2.bar(x, train_augmented_values, width,
                             bottom=train_original_values,
                             label='Train Augmented',
                             color='#85c1e9',
                             edgecolor='#2471a3',
                             linewidth=0.8,
                             alpha=0.7,
                             hatch='///')
    
    # Plot batang Test (hijau) - ditumpuk di atas Train
    bottom_test = [train_original_values[i] + train_augmented_values[i] for i in range(len(classes))]
    bars_test = ax2.bar(x, test_values, width,
                        bottom=bottom_test,
                        label='Test (Original Only)',
                        color='#2ecc71',
                        edgecolor='#27ae60',
                        linewidth=0.8,
                        alpha=0.75,
                        hatch='...')
    
    # Tambahkan anotasi total di atas setiap batang
    max_total = max([train_original_values[i] + train_augmented_values[i] + test_values[i] 
                     for i in range(len(classes))])
    
    for i in range(len(classes)):
        total = train_original_values[i] + train_augmented_values[i] + test_values[i]
        if total > 0:
            ax2.text(i, total + max_total*0.01, f'{total}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#2c3e50')
            
            # Tambahkan breakdown kecil di atas untuk kelas dengan augmentasi
            if train_augmented_values[i] > 0:
                ax2.text(i, total + max_total*0.06, 
                        f'(Tr:{train_original_values[i]}+{train_augmented_values[i]} | Ts:{test_values[i]})',
                        ha='center', va='bottom', fontsize=6.5, color='#7f8c8d', style='italic')
    
    ax2.set_title('DISTRIBUSI DATA SETELAH AUGMENTASI & TRAIN-TEST SPLIT (80:20)', 
                  fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
    ax2.set_xlabel('Kelas Abjad', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Jumlah Data (Train + Test)', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, fontsize=10, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9, ncol=3)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, max_total * 1.25 if max_total > 0 else 1)
    
    # Tambahkan ringkasan statistik sebagai teks
    total_original = sum(original_values)
    total_final_train = sum(train_original_values) + sum(train_augmented_values)
    total_final_test = sum(test_values)
    
    stats_text = (f'📊 Ringkasan Pipeline:\n'
                  f'• Total Original: {total_original} gambar\n'
                  f'• Final Train: {total_final_train} data (Original + Augmented)\n'
                  f'• Final Test: {total_final_test} data\n'
                  f'• Rasio Augmentasi: {JUMLAH_AUGMENTASI}x per Train Image\n'
                  f'• Split Ratio: 80:20 (Train:Test)')
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9, edgecolor='#bdc3c7'),
             verticalalignment='bottom')
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    nama_file = 'distribusi_data_before_after.png'
    plt.savefig(nama_file, dpi=150, bbox_inches='tight')
    print(f" [✓] Tersimpan: {nama_file}")
    plt.close(fig)

# ==========================================
# EKSEKUSI PIPELINE
# ==========================================
def main():
    source_path = Path(SOURCE_DIR)
    if not source_path.exists():
        print(f"[ERROR] Folder '{SOURCE_DIR}' tidak ditemukan!")
        return

    # List untuk menampung fitur dan label sebelum dikonversi ke Numpy Array
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    total_train_ori = 0
    total_train_aug = 0
    total_test_ori = 0

    # Dictionary untuk tracking distribusi
    original_counts = defaultdict(int)  # Jumlah gambar original per kelas
    train_original_counts = defaultdict(int)  # Jumlah train original per kelas
    train_augmented_counts = defaultdict(int)  # Jumlah train augmented per kelas
    test_counts = defaultdict(int)  # Jumlah test per kelas

    # Penampung untuk Visualisasi
    captured_samples = {
        '1_Binarized': {}, '2_Blurred': {}, 
        '3_Cropped': {}, '4_Squared': {}, '5_Final_28x28': {}
    }
    sample_aug_data = None 

    folders = sorted([d for d in source_path.iterdir() if d.is_dir() and d.name.lower() in VALID_CLASSES])

    print("Memulai pemrosesan dengan Split 80:20 (Augmentasi HANYA pada Train)...")
    for folder in folders:
        class_name = folder.name.lower()
        label_idx = LABEL_MAP[class_name]
        
        images = sorted([f for f in folder.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        
        # Hitung titik potong 80% (Train) dan 20% (Test)
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Catat jumlah original
        original_counts[class_name] = len(images)
        test_counts[class_name] = len(test_images)
        
        sample_captured = False # Reset pemicu visualisasi per kelas
        
        # 1. PROSES DATA TRAIN (Ambil Asli + Augmentasi + Simpan Visualisasi)
        for img_path in train_images:
            try:
                # Hanya tangkap debug untuk 1 gambar pertama per kelas
                needs_capture = not sample_captured
                list_features, debug_imgs = process_image_to_abjad(img_path, capture_debug=needs_capture)
                
                if list_features:
                    for i, features in enumerate(list_features):
                        X_train_list.append(features)
                        y_train_list.append(label_idx)
                        
                        if i == 0:
                            total_train_ori += 1
                            train_original_counts[class_name] += 1
                        else:
                            total_train_aug += 1
                            train_augmented_counts[class_name] += 1
                            
                    # Simpan gambar ke kamus visualisasi jika mode capture aktif
                    if needs_capture and debug_imgs:
                        for step_name, img_array in debug_imgs.items():
                            if step_name != 'Augmentations_List':
                                captured_samples[step_name][class_name] = img_array
                        
                        # Simpan satu contoh kelas pertama yang diproses untuk gambar perbandingan augmentasi
                        if sample_aug_data is None and 'Augmentations_List' in debug_imgs:
                            sample_aug_data = (debug_imgs['5_Final_28x28'], debug_imgs['Augmentations_List'], class_name)
                            
                        sample_captured = True
            except Exception:
                pass
                
        # 2. PROSES DATA TEST (HANYA Ambil Asli, abaikan augmentasi & visualisasi)
        for img_path in test_images:
            try:
                list_features, _ = process_image_to_abjad(img_path, capture_debug=False)
                if list_features and len(list_features) > 0:
                    # Ambil indeks ke-0 karena hanya mengambil gambar asli
                    X_test_list.append(list_features[0])
                    y_test_list.append(label_idx)
                    total_test_ori += 1
            except Exception:
                pass

        print(f" [✓] Kelas '{class_name}' -> Latih: {len(train_images)} img ({train_original_counts[class_name]} ori + {train_augmented_counts[class_name]} aug) | Uji: {len(test_images)} img")

    # Memanggil pembuat plot visualisasi setelah seluruh iterasi folder selesai
    if any(captured_samples['1_Binarized']):
        generate_step_visualizations(captured_samples)
        
    if sample_aug_data:
        generate_augmentation_comparison(sample_aug_data[0], sample_aug_data[1], sample_aug_data[2])

    # Panggil visualisasi distribusi Before-After
    plot_distributions(original_counts, train_original_counts, test_counts, train_augmented_counts)

    print("\nMenyimpan ke NPZ...")
    # Konversi list ke Numpy Arrays
    # Pixel dikonversi ke uint8 untuk menghemat memori
    X_train = np.array(X_train_list, dtype=np.uint8)
    y_train = np.array(y_train_list, dtype=np.int32)
    
    X_test = np.array(X_test_list, dtype=np.uint8)
    y_test = np.array(y_test_list, dtype=np.int32)

    # Simpan ke dalam file Numpy Terkompresi
    np.savez_compressed(OUTPUT_TRAIN_NPZ, images=X_train, labels=y_train)
    np.savez_compressed(OUTPUT_TEST_NPZ, images=X_test, labels=y_test)
    
    print("=" * 60)
    print(" PIPELINE DATASET KUSTOM SELESAI!")
    print("=" * 60)
    print(f" Total Train (Asli + Aug) : {len(X_train)} data -> {OUTPUT_TRAIN_NPZ}")
    print(f" Total Test  (Hanya Asli) : {len(X_test)} data -> {OUTPUT_TEST_NPZ}")
    print(f" Breakdown Train: {total_train_ori} original + {total_train_aug} augmented")
    print(f" Breakdown Test : {total_test_ori} original")
    print("=" * 60)

if __name__ == "__main__":
    main()