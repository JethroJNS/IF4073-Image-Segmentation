# Laporan Evaluasi Final CamVid Test Set (11 Kelas) — DeepLabV3 MobileNetV3-Large

## 1. Informasi Checkpoint & Konfigurasi Evaluasi
- **Checkpoint**: `C:\Users\LENOVO\Documents\Semester 7\IF4073-Image-Segmentation\models\camvid11_full_25epochs_best.pth`
- **Epoch Terbaik**: Epoch `22` (`stage = finetune`)
- **Arsitektur**: `deeplabv3_mobilenet_v3_large` (`num_classes = 11`, `ignore_index = 255`)
- **Split Dataset**: `test` (**168 citra** dari `data/CamVid/test` & `data/CamVid/test_labels`)
- **Resolusi Evaluasi**: `360 × 480` ($H \times W$), normalisasi ImageNet konsisten dengan training
- **Total Piksel Valid (excl. `ignore_index=255`)**: `28,164,248` piksel (`866,152` piksel `Void/255` atau `2.98%` diabaikan)
- **Waktu Evaluasi 168 Citra (cpu)**: `58.06` detik (`345.6` ms/citra)

## 2. Ringkasan Metrik Utama (Validation vs Test Set)

| Metrik | Validation Set (112 Citra) | Test Set (168 Citra) | Selisih (Test - Val) |
| :--- | :---: | :---: | :---: |
| **Pixel Accuracy (excl. 255)** | `90.70%` | **`90.46%`** | `-0.24%` |
| **Mean IoU (mIoU, 11 Kelas)** | `65.43%` | **`65.46%`** | `+0.03%` |
| **Frequency-Weighted IoU** | — | **`84.36%`** | — |
| **Mean Precision (11 Kelas)** | — | **`73.03%`** | — |
| **Mean Recall (11 Kelas)** | — | **`81.65%`** | — |
| **Mean F1-Score / Dice** | — | **`76.42%`** | — |
| **Cross-Entropy Loss** | `0.7827` | **`0.7571`** | `-0.0257` |

## 3. Tabel Metrik Per-Kelas pada Test Set (11 Kelas)

| ID | Nama Kelas | Val IoU (%) | Test IoU (%) | Selisih IoU | Precision (%) | Recall (%) | F1-Score (%) | GT Pixels | Pred Pixels |
| :-: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | **Sky** | `89.80%` | **`89.29%`** | `-0.51%` | `96.14%` | `92.61%` | `94.34%` | `4,679,257` | `4,507,384` |
| `1` | **Building** | `83.46%` | **`82.05%`** | `-1.41%` | `94.49%` | `86.17%` | `90.14%` | `7,042,241` | `6,422,448` |
| `2` | **Pole** | `18.88%` | **`17.79%`** | `-1.09%` | `22.42%` | `46.25%` | `30.20%` | `265,881` | `548,469` |
| `3` | **Road** | `94.47%` | **`94.50%`** | `+0.03%` | `98.57%` | `95.82%` | `97.17%` | `8,460,915` | `8,224,950` |
| `4` | **Pavement** | `76.02%` | **`77.84%`** | `+1.81%` | `84.99%` | `90.24%` | `87.54%` | `2,036,715` | `2,162,484` |
| `5` | **Tree** | `77.90%` | **`77.42%`** | `-0.48%` | `84.33%` | `90.43%` | `87.27%` | `3,211,488` | `3,443,853` |
| `6` | **SignSymbol** | `40.19%` | **`39.22%`** | `-0.97%` | `46.29%` | `71.98%` | `56.34%` | `260,418` | `404,976` |
| `7` | **Fence** | `59.04%` | **`54.90%`** | `-4.14%` | `67.08%` | `75.14%` | `70.88%` | `416,182` | `466,127` |
| `8` | **Car** | `81.69%` | **`81.80%`** | `+0.12%` | `87.69%` | `92.41%` | `89.99%` | `1,427,578` | `1,504,471` |
| `9` | **Pedestrian** | `42.20%` | **`42.51%`** | `+0.30%` | `51.03%` | `71.78%` | `59.66%` | `195,897` | `275,547` |
| `10` | **Bicyclist** | `56.12%` | **`62.76%`** | `+6.64%` | `70.33%` | `85.37%` | `77.12%` | `167,676` | `203,539` |

## 4. Analisis Generalization Gap (Validation vs Test Set)
- **Selisih Global**: Validation mIoU adalah **`65.43%`** sedangkan Test mIoU adalah **`65.46%`** (selisih **`+0.03%`**), dan Validation Pixel Accuracy adalah **`90.70%`** sedangkan Test Pixel Accuracy adalah **`90.46%`** (selisih **`-0.24%`**).
- **Faktor Penyebab Perbedaan Per-Kelas**:
  1. **Pemilihan Model Berdasarkan Validation mIoU**: Checkpoint Epoch 22 dipilih ketika validation mIoU mencapai puncaknya (`65.43%`), sehingga secara statistik wajar terdapat sedikit *optimism bias* pada validation set dibanding unseen test set.
  2. **Perbedaan Komposisi Sekuens & Pencahayaan pada Test Split**: Pada split CamVid, beberapa sekuens memiliki variasi kondisi pencahayaan/bayangan dan proporsi objek kecil (`Pole`, `SignSymbol`, `Pedestrian`, `Bicyclist`, `Fence`) yang berbeda antara `val` (`112` citra) dan `test` (`168` citra).
  3. **Trade-off Class Weighting ENet pada Kelas Minor (Recall Tinggi vs Precision Lebih Rendah)**: Karena *class weights* ENet memberikan bobot tinggi pada kelas langka (`Bicyclist = 2.00`, `Pedestrian = 1.87`, `Pole = 1.67`, `SignSymbol = 1.61`), model memiliki **Recall yang tinggi** pada kelas-kelas kecil tersebut, namun menghasilkan sejumlah *false positive* di tepi objek (tercermin pada Precision yang lebih rendah dibanding Recall untuk `Pole`, `SignSymbol`, dan `Pedestrian`).

## 5. Ringkasan 15 Sampel Visualisasi Test Set

| No | Test Idx | Nama File | Pixel Acc (%) | Sample mIoU (%) | GT Void 255 (%) | Kelas Terdeteksi |
| :-: | :-: | :--- | ---: | ---: | ---: | :--- |
| `01` | `0` | `0001TP_007170.png` | `85.53%` | `47.83%` | `6.41%` | Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist |
| `02` | `12` | `0001TP_008670.png` | `86.95%` | `51.56%` | `6.41%` | Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Car, Pedestrian, Bicyclist |
| `03` | `24` | `0001TP_010170.png` | `87.01%` | `57.11%` | `5.13%` | Sky, Building, Pole, Road, Pavement, Tree, Fence, Car, Pedestrian |
| `04` | `36` | `0006R0_f02190.png` | `90.52%` | `48.67%` | `1.46%` | Sky, Building, Road, Tree, SignSymbol, Car, Pedestrian, Bicyclist |
| `05` | `48` | `0006R0_f03690.png` | `87.39%` | `49.99%` | `3.15%` | Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian |
| `06` | `60` | `0016E5_01620.png` | `90.98%` | `65.19%` | `0.89%` | Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian |
| `07` | `72` | `0016E5_05040.png` | `92.72%` | `66.98%` | `0.75%` | Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Car, Pedestrian |
| `08` | `84` | `0016E5_06540.png` | `95.15%` | `56.34%` | `8.32%` | Sky, Building, Pole, Road, Pavement, SignSymbol, Fence, Car, Pedestrian, Bicyclist |
| `09` | `95` | `0016E5_07530.png` | `80.17%` | `49.75%` | `1.45%` | Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist |
| `10` | `107` | `0016E5_08031.png` | `91.56%` | `70.68%` | `0.89%` | Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist |
| `11` | `119` | `0016E5_08131.png` | `93.90%` | `75.09%` | `0.62%` | Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist |
| `12` | `131` | `Seq05VD_f00570.png` | `89.35%` | `51.51%` | `1.16%` | Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Car, Pedestrian |
| `13` | `143` | `Seq05VD_f02070.png` | `89.29%` | `64.30%` | `0.77%` | Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Car, Pedestrian |
| `14` | `155` | `Seq05VD_f03570.png` | `88.49%` | `49.21%` | `1.86%` | Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car |
| `15` | `167` | `Seq05VD_f05070.png` | `88.06%` | `54.18%` | `10.86%` | Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Bicyclist |
