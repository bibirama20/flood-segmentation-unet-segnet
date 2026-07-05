---
title: Segmentasi Banjir U-Net vs SegNet
emoji: 🌊
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Perbandingan Segmentasi Citra Banjir: U-Net vs SegNet

GUI untuk membandingkan hasil segmentasi citra banjir antara model **U-Net** dan **SegNet**.

## Cara Deploy ke Hugging Face Spaces

1. Jalankan cell terakhir di `segmenfinal (1).ipynb` (setelah training selesai) untuk menyimpan model:
   - `unet_model.h5`
   - `segnet_model.h5`
2. Download kedua file `.h5` dari Google Drive, taruh di folder ini (sejajar dengan `app.py`).
3. Buat Space baru di https://huggingface.co/new-space dengan SDK **Gradio**.
4. Upload 4 file berikut ke Space:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `unet_model.h5`
   - `segnet_model.h5`
5. Tunggu build selesai (beberapa menit), Space otomatis jalan dan tetap online (idle sleep setelah 48 jam tanpa akses, otomatis bangun saat diakses lagi).

## Menjalankan Lokal (opsional, untuk uji coba sebelum deploy)

```bash
pip install -r requirements.txt
python app.py
```

## Dataset untuk Metrik Ground Truth (opsional)

Folder `data/` **tidak ikut di-push ke GitHub** (lihat `.gitignore`) karena ukurannya besar.
Untuk mengaktifkan fitur pencocokan otomatis ground truth mask (tab "Metrik Model"), siapkan
folder `data/` secara manual sejajar dengan `app.py`, dengan struktur:

```
data/
  metadata.csv       # kolom: Image,Mask (nama file harus berpasangan)
  Image/             # citra asli (.jpg)
  Mask/              # ground truth mask (.png)
```

Tanpa folder ini, aplikasi tetap bisa jalan normal — hanya saja gambar dari luar dataset harus
diupload bersama mask ground truth manual lewat kolom "Upload Mask Ground Truth" di tab
Segmentasi agar metrik ikut dihitung.
