## Lowercase Latin Character Classification Model

Repositori ini berisi kode dan pipeline untuk melakukan klasifikasi karakter huruf kecil Latin (a-z). Proyek ini membandingkan dua pendekatan utama: *Deep Learning* menggunakan arsitektur **AlexNet** dan *Machine Learning* konvensional menggunakan **Support Vector Machine (SVM)**. 

Kedua model tersebut dioptimasi menggunakan teknik **Bayesian Optimization (BO)** untuk mendapatkan konfigurasi *hyperparameter* terbaik secara efisien.


## Struktur Repositori

```text
README.md has been generated.

```text
├── dataset/
│   ├── pipeline_preprocessing+augment.md   # Dokumentasi pipeline preprocessing & augmentasi
│   ├── preprocessing+augment.py            # Skrip untuk preprocessing dan augmentasi citra
│   ├── undersampling.py                    # Skrip untuk undersampling (menyeimbangkan data)
│   └── undersampling_pipeline.md           # Dokumentasi pipeline undersampling
├── Notebooks/
│   ├── AlexNet_pipeline.md                 # Dokumentasi alur kerja model AlexNet
│   ├── Klasifikasi_huruf_kecil_latin_BO+AlexNet.ipynb  # Notebook training & evaluasi AlexNet
│   ├── Klasifikasi_huruf_kecil_latin_BO+SVM.ipynb      # Notebook training & evaluasi SVM
│   └── SVM_pipeline.md                     # Dokumentasi alur kerja model SVM
└── Result/
    ├── AlexNet Output/                     # Hasil evaluasi, metrik JSON, grafik kurva, dan confusion matrix AlexNet
    └── SVM Output/                         # Hasil evaluasi, metrik JSON, precision-recall, dan confusion matrix SVM

```

## ⚙️ Alur Kerja (Pipeline)

1. **Penyeimbangan Data (Undersampling)**: Mengatasi ketidakseimbangan jumlah sampel antar kelas huruf agar model tidak bias (diatur melalui `dataset/undersampling.py`).
2. **Preprocessing & Augmentasi**: Mengubah format dan ukuran citra agar seragam, dilanjutkan dengan teknik augmentasi (seperti rotasi, zoom, translasi) untuk memperkaya variabilitas data dan mencegah *overfitting* (`dataset/preprocessing+augment.py`).
3. **Hyperparameter Tuning**:
* Menggunakan **Bayesian Optimization** pada *Jupyter Notebook* untuk mencari titik optimal.
* **AlexNet**: Mengoptimalkan parameter seperti *learning rate*, *batch size*, dll.
* **SVM**: Mengoptimalkan parameter `C`, `gamma`, dan pemilihan `kernel`.


4. **Evaluasi**: Mengukur performa akhir model menggunakan metrik *Accuracy*, *Precision*, *Recall*, *F1-Score*, serta visualisasi performa melalui *Confusion Matrix* dan grafik kurva pembelajaran.

## Cara Penggunaan

1. **Persiapan Data**
Jalankan skrip di direktori `dataset/` untuk mempersiapkan gambar sebelum di-training:
```bash
python dataset/undersampling.py
python dataset/preprocessing+augment.py

```


2. **Training & Evaluasi Model**
Buka direktori `Notebooks/` menggunakan Jupyter Notebook atau JupyterLab.
* Untuk menjalankan klasifikasi berbasis Deep Learning, jalankan sel pada `Klasifikasi_huruf_kecil_latin_BO+AlexNet.ipynb`.
* Untuk menjalankan klasifikasi berbasis Machine Learning, jalankan sel pada `Klasifikasi_huruf_kecil_latin_BO+SVM.ipynb`.


3. **Analisis Hasil**
Setelah notebook selesai dijalankan, hasil (berupa metrik `.json` dan grafik `.png`) akan tersimpan secara otomatis di dalam direktori `Result/`.

## Hasil dan Metrik

Hasil dari eksperimen dicatat secara detail pada folder `Result/`. Beberapa visualisasi yang dihasilkan meliputi:

* **Confusion Matrix**: Untuk melihat detail salah klasifikasi antar huruf yang mirip (misalnya 'l' dan 'i' atau 'p' dan 'q').
* **Per-Class Accuracy**: Akurasi klasifikasi untuk masing-masing kelas huruf.
* **Training Curves (AlexNet)**: Grafik *loss* dan akurasi per epoch.
* **Precision-Recall Curve (SVM)**: Evaluasi trade-off antara *precision* dan *recall*.

## Dependencies

Pastikan pustaka berikut telah terinstal sebelum menjalankan skrip dan notebook:

* Python 3.8+
* Jupyter Notebook / Lab
* PyTorch (untuk implementasi AlexNet)
* scikit-learn (untuk implementasi SVM dan evaluasi)
* scikit-optimize (`skopt`) (untuk Bayesian Optimization)
* OpenCV / PIL (untuk pemrosesan citra)
* NumPy & Pandas
* Matplotlib & Seaborn (untuk visualisasi)

## Sumber Dataset

### 1. Dataset Custom
Dataset buatan sendiri yang dikumpulkan dan diproses secara manual.
 **Link Dataset:**
[https://drive.google.com/drive/folders/1Y5TmtzUryBgSkSTT4I74l_KhlKbBPsuU?usp=sharing](https://drive.google.com/drive/folders/1Y5TmtzUryBgSkSTT4I74l_KhlKbBPsuU?usp=sharing)

### 2. Dataset EMNIST (Extended MNIST)
Dataset publik berisi karakter tulisan tangan yang merupakan ekstensi dari MNIST.
 **Link Dataset:**
[https://www.kaggle.com/datasets/crawford/emnist](https://www.kaggle.com/datasets/crawford/emnist)

