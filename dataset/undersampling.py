import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# KONFIGURASI FILE
# ==========================================
INPUT_TRAIN_CSV = 'emnist_lowercase_train.csv'
INPUT_TEST_CSV = 'emnist_lowercase_test.csv'
OUTPUT_TRAIN_NPZ = 'emnist_lowercase_train_balanced.npz'
OUTPUT_TEST_NPZ = 'emnist_lowercase_test_balanced.npz'
OUTPUT_CHART_IMG = 'distribusi_sebelum_sesudah.png'

# ==========================================
# 1. FUNGSI LOAD DATA AMAN
# ==========================================
def load_clean_data(file_path):
    print(f"Memuat dan membersihkan {file_path}...")
    chunks = []
    
    # Baca tanpa memaksakan dtype di awal agar tidak crash saat bertemu header ganda
    chunk_iterator = pd.read_csv(file_path, chunksize=50000, low_memory=False)
    
    for chunk in chunk_iterator:
        chunk_clean = chunk[chunk['label'] != 'label'].copy()
        chunk_clean = chunk_clean.astype(np.uint8) # Kompresi ke uint8
        chunks.append(chunk_clean)
        
    return pd.concat(chunks, ignore_index=True)

# ==========================================
# 2. FUNGSI UNDERSAMPLING & SHUFFLING
# ==========================================
def balance_and_shuffle(df, dataset_name="Dataset", offset_char=0):
    print(f"\nMengevaluasi distribusi kelas pada {dataset_name}...")
    class_counts = df['label'].value_counts()
    min_class = class_counts.idxmin()
    min_count = class_counts.min()

    huruf_minoritas = chr(min_class + offset_char)
    print(f"   -> Kelas paling sedikit adalah '{huruf_minoritas}' (label {min_class}) dengan {min_count:,} sampel.")
    print(f"   -> Melakukan Random Undersampling menjadi {min_count:,} sampel per kelas...")
    
    # Random Undersampling berdasarkan kelas terkecil
    df_balanced = df.groupby('label').sample(n=min_count, random_state=42)
    
    # Mengacak ulang (shuffling) urutan baris
    print(f"   -> Mengacak ulang (shuffling) urutan dataset {dataset_name}...")
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_balanced, min_count

# ==========================================
# 3. FUNGSI SIMPAN KE NPZ
# ==========================================
def save_to_npz(df, output_filename):
    print(f"   -> Menyimpan ke {output_filename}...")
    y = df['label'].to_numpy(dtype=np.int32)
    X = df.drop('label', axis=1).to_numpy(dtype=np.uint8)
    
    np.savez_compressed(output_filename, images=X, labels=y)
    print(f"   [✓] Selesai disimpan!")

# ==========================================
# 4. FUNGSI VISUALISASI
# ==========================================
def plot_distributions(train_sebelum, test_sebelum, train_sesudah, test_sesudah, min_label, offset_char):
    print(f"\nMembuat visualisasi komparasi sebelum dan sesudah undersampling...")
    
    # Dinamis mengikuti apakah EMNIST 0-indexed atau 1-indexed
    classes = np.arange(min_label, min_label + 26)
    abjad_labels = [chr(i + offset_char) for i in classes]

    data_skenario = [
        {
            "judul": "Before Undersampling (Class Imbalance)",
            "train_counts": train_sebelum,
            "test_counts": test_sebelum
        },
        {
            "judul": "After Undersampling (Balanced Dataset)",
            "train_counts": train_sesudah,
            "test_counts": test_sesudah
        }
    ]

    plt.style.use('default') 
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 12))

    for ax, skenario in zip(axes, data_skenario):
        # Ambil value dari dictionary, set 0 jika kelas tidak ditemukan
        train_values = [skenario["train_counts"].get(i, 0) for i in classes]
        test_values = [skenario["test_counts"].get(i, 0) for i in classes]
        
        total_train_samples = sum(train_values)
        total_test_samples = sum(test_values)
        total_population = total_train_samples + total_test_samples
        
        train_pct = (total_train_samples / total_population * 100) if total_population > 0 else 0
        test_pct = (total_test_samples / total_population * 100) if total_population > 0 else 0
        
        # Plot Data Train
        ax.bar(abjad_labels, train_values, color='#4C72B0', edgecolor='black', 
               label=f'Train Split ({train_pct:.1f}%)')
        
        # Plot Data Test (Ditumpuk dengan parameter bottom)
        ax.bar(abjad_labels, test_values, bottom=train_values, color='#55A868', 
               edgecolor='black', hatch='///', alpha=0.9, label=f'Test Split ({test_pct:.1f}%)')
        
        # Formatting Plot
        ax.set_title(f"Visual Breakdown of Lowercase Character Dataset\n{skenario['judul']}", 
                     fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("Classes (a-z)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Number of Samples", fontsize=11, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True) 
        
        # Set X-ticks
        ax.set_xticks(range(26))
        ax.set_xticklabels(abjad_labels)
        ax.legend(loc='upper right', fontsize=11, framealpha=1)

    plt.tight_layout(pad=3.0)
    plt.savefig(OUTPUT_CHART_IMG, dpi=150, bbox_inches='tight')
    print(f" [✓] Grafik tersimpan sebagai: {OUTPUT_CHART_IMG}")

# ==========================================
# EKSEKUSI UTAMA
# ==========================================
def main():
    print("=== MEMULAI PIPELINE DATASET EMNIST ===")
    
    # 1. Load Data
    df_train = load_clean_data(INPUT_TRAIN_CSV)
    df_test = load_clean_data(INPUT_TEST_CSV)
    print(f" -> Data Train awal : {len(df_train):,} baris")
    print(f" -> Data Test awal  : {len(df_test):,} baris")

    # Ambil metrik untuk mapping ASCII dinamis (antisipasi 0-indexed vs 1-indexed)
    min_label = df_train['label'].min()
    offset_char = ord('a') if min_label == 0 else ord('a') - 1

    # Rekam distribusi SEBELUM undersampling
    train_counts_sebelum = df_train['label'].value_counts().to_dict()
    test_counts_sebelum = df_test['label'].value_counts().to_dict()

    # 2. Proses Data Latih (Train)
    df_train_balanced, min_train_count = balance_and_shuffle(df_train, "Data Train", offset_char)
    save_to_npz(df_train_balanced, OUTPUT_TRAIN_NPZ)

    # 3. Proses Data Uji (Test)
    df_test_balanced, min_test_count = balance_and_shuffle(df_test, "Data Test", offset_char)
    save_to_npz(df_test_balanced, OUTPUT_TEST_NPZ)

    # Rekam distribusi SESUDAH undersampling
    train_counts_sesudah = df_train_balanced['label'].value_counts().to_dict()
    test_counts_sesudah = df_test_balanced['label'].value_counts().to_dict()

    # 4. Bangun dan Simpan Visualisasi
    plot_distributions(train_counts_sebelum, test_counts_sebelum, 
                       train_counts_sesudah, test_counts_sesudah, 
                       min_label, offset_char)

    # 5. Laporan Akhir
    print("\n" + "="*50)
    print("STATISTIK DATASET AKHIR (BALANCED NPZ):")
    print("="*50)
    print("[DATA TRAIN]")
    print(f"Total Baris         : {len(df_train_balanced):,}")
    print(f"Isi Tiap Kelas      : {min_train_count:,} sampel/kelas")
    
    print("\n[DATA TEST]")
    print(f"Total Baris         : {len(df_test_balanced):,}")
    print(f"Isi Tiap Kelas      : {min_test_count:,} sampel/kelas")
    print("="*50)

if __name__ == "__main__":
    main()