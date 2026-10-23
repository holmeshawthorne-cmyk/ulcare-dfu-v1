# ULCARE Streamlit

Antarmuka Streamlit untuk prediksi Wagner Stage **0–4** memakai `model_gangren_wagner_v1.h5`.

## Jalankan secara lokal

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Model ditempatkan dalam folder proyek agar `app.py` dapat memuatnya otomatis. Hasil memuat probabilitas Stage 0–4, alasan yang mengikuti stage prediksi, dan **Grad-CAM attention map** (bukan segmentation mask).

## GitHub

Model berukuran sekitar 22 MB, masih berada di bawah batas berkas GitHub 100 MB. Jangan unggah data pasien atau foto luka ke repositori publik.
