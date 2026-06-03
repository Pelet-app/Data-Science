# PELET — Data Science

**PELET (Pencari Lowongan Efektif & Tepat): AI Semantic Matching Berbasis SKKNI**

**Coding Camp 2026 powered by DBS Foundation | Tim CC26-PSU060**

---

# Daftar Isi

1. [Problem Discovery](#1-problem-discovery)
2. [Pertanyaan Bisnis](#2-pertanyaan-bisnis)
3. [Data Wrangling](#3-data-wrangling)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis-eda)
5. [Visualisasi & Explanatory Analysis](#5-visualisasi--explanatory-analysis)
6. [Persiapan Data untuk Modeling](#6-persiapan-data-untuk-modeling)
7. [Dashboard Streamlit](#7-dashboard-streamlit)
8. [A/B Testing](#8-ab-testing)
9. [Laporan Teknis](#9-laporan-teknis)
10. [Struktur Proyek](#10-struktur-proyek)
11. [Cara Menjalankan](#11-cara-menjalankan)

---

# 1. Problem Discovery

## Permasalahan

Indonesia menghadapi masalah *skill mismatch* yang kronis antara kompetensi pencari kerja dan kebutuhan nyata industri. Berdasarkan data BPS (2024), tingkat pengangguran tertinggi justru berasal dari lulusan SMK dan pendidikan tinggi, mencapai 8,62%. Sebanyak 50% tenaga kerja Indonesia mengalami *vertical mismatch* (Mandiri Institute, 2025), dan 46% perusahaan di Indonesia kesulitan menemukan kandidat yang sesuai (Transcon Indonesia, 2025).

Sistem pencocokan kerja yang ada saat ini masih mengandalkan *keyword matching* yang kaku, sementara Standar Kompetensi Kerja Nasional Indonesia (SKKNI) belum terintegrasi secara optimal dengan sistem rekrutmen digital modern. Akibatnya, kandidat berkualitas dapat terlewat hanya karena perbedaan penulisan istilah kompetensi, bukan karena tidak kompeten.

PELET hadir sebagai solusi semantic matching berbasis AI dan SKKNI untuk menjembatani kesenjangan semantik antara dunia industri, resume pelamar, dan standar kompetensi nasional.

---

# 2. Pertanyaan Bisnis

| No | Pertanyaan Bisnis                                                                                                      | Dataset                                                      | Metrik                                                    |
| -- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------- |
| 1  | Bagaimana kesenjangan (gap) antara rumpun skill digital yang dimiliki oleh para pelamar kerja dengan kualifikasi skill yang saat ini paling banyak dibutuhkan oleh industri? | `cleaned_training_data.csv` + `cleaned_job_5.csv`            | Frekuensi skill supply vs demand per 1.000 entri          |
| 2  | Bagaimana kecocokan profil pengalaman kerja (tahun pengalaman) para pencari kerja saat ini jika disandingkan dengan ekspektasi atau persyaratan minimum dari industri? | `cleaned_training_data.csv` + `cleaned_job_5.csv`            | Distribusi `exp_bucket` supply vs demand                  |
| 3  | Seberapa luas cakupan atau kesesuaian unit kompetensi standar nasional (SKKNI) dalam memetakan skill-skill digital modern yang berkembang di pasar kerja saat ini?  | `skill_mapping_dictionary.csv` + `skkni_reference_clean.csv` | Persentase skill terpetakan, parsial, dan belum dipetakan |

---

# 3. Data Wrangling

## 3.1 Gathering Data

Dataset dikumpulkan dari berbagai sumber dan diproses secara manual melalui tahap scraping, filtering, cleaning, dan transformasi.

| Dataset                             | Sumber                     | Metode Pengumpulan                             | Output Final                   |
| ----------------------------------- | -------------------------- | ---------------------------------------------- | ------------------------------ |
| Data SKKNI                          | Website resmi SKKNI        | Web scraping manual + cleaning                 | `skkni_reference_clean.csv`    |
| Jobstreet Vacancy Dataset Indonesia | GitHub publik              | Download dataset publik + filtering + cleaning | `cleaned_job_5.csv`            |
| Resume & Job Role Dataset           | Kaggle                     | Dataset sintetis resume dan skill              | `cleaned_training_data.csv`    |
| Skill Mapping Dictionary            | Hasil pemetaan mandiri tim | Mapping skill internasional ke unit SKKNI      | `skill_mapping_dictionary.csv` |

### Penjelasan Sumber Dataset

* **SKKNI** diperoleh melalui proses scraping manual dari website resmi SKKNI dan digunakan untuk membangun referensi unit kompetensi nasional.
* **Jobstreet Vacancy Dataset Indonesia** berasal dari GitHub publik dan digunakan sebagai representasi kebutuhan industri digital Indonesia.
* **Resume & Job Role Dataset** berasal dari Kaggle dan digunakan sebagai representasi profil skill pelamar kerja.
* **Skill Mapping Dictionary** dibangun secara manual untuk menjembatani istilah skill internasional dengan unit kompetensi SKKNI.

---

## 3.2 Assessing Data

### Dataset SKKNI (`skkni_reference_clean.csv`)

* Terdiri dari 662 baris unit kompetensi.
* Ditemukan inkonsistensi penulisan jabatan kerja.
* Banyak jabatan modern seperti AI Engineer dan DevOps Engineer belum terakomodasi penuh.

### Dataset Lowongan (`cleaned_job_5.csv`)

* Terdiri dari 2.516 lowongan kerja.
* Banyak kolom salary dan job level kosong.
* Kolom requirement masih berupa teks tidak terstruktur.

### Dataset Resume (`cleaned_training_data.csv`)

* Terdiri dari 10.000 resume sintetis.
* Kolom skill masih berbentuk string list mentah.
* Banyak variasi penulisan skill internasional.

### Skill Mapping Dictionary (`skill_mapping_dictionary.csv`)

* Berisi 114 skill digital modern.
* Sebagian besar skill modern belum memiliki padanan langsung di SKKNI.

---

## 3.3 Cleaning Data

### Cleaning Dataset Jobstreet

* Imputasi nilai null pada `type_of_work` dan `currency`.
* Membuat fitur `salary_mid`.
* Normalisasi lokasi menjadi level kota.
* Membersihkan karakter newline dan simbol.

### Cleaning Dataset Resume Kaggle

* Parsing string list skill menjadi Python list.
* Membuat fitur `skill_count`.
* Membuat kategorisasi pengalaman kerja (`exp_bucket`).

### Cleaning Dataset SKKNI

* Normalisasi teks kompetensi.
* Menghapus karakter tidak standar.
* Standarisasi nama jabatan.

---

## 3.4 Integrasi & Feature Engineering

Proses integrasi dilakukan pada `data_loader.py` sebagai pipeline utama dashboard dan modeling.

### Feature Engineering

* `kota`
* `salary_mid`
* `has_salary`
* `req_skills`
* `req_skill_count`
* `skill_count`
* `exp_bucket`

### Teknik yang Digunakan

* Regex matching skill
* Parsing object list
* City normalization
* Experience bucketing
* Dynamic path handling

---

# 4. Exploratory Data Analysis (EDA)

EDA dilakukan dalam dua tahap utama:

## EDA Tahap Awal (`EDA1.ipynb`)

Tahap awal dilakukan terhadap dataset gabungan untuk memahami karakteristik data serta menghasilkan dataset hasil refinement yang digunakan pada tahap modeling.

### Output Utama EDA Awal

EDA awal menghasilkan dua dataset penting:

* `skill_mapping_dictionary.csv`
* `skkni_reference_clean.csv`

Kedua dataset tersebut kemudian digunakan kembali pada proses modeling dan dashboard analytics.

### Temuan Utama

* Resume menggunakan istilah skill global seperti Python, SQL, Machine Learning.
* SKKNI menggunakan bahasa formal berbasis aktivitas kerja.
* Terjadi *semantic gap* besar antara bahasa industri dan bahasa regulasi nasional.

### Dampak

Jika dilakukan exact keyword matching biasa, maka kecocokan resume dan SKKNI akan sangat rendah meskipun sebenarnya kompetensinya relevan.

---

## EDA Tahap Final (`EDA2.ipynb`)

EDA final dilakukan terhadap empat dataset utama hasil cleaning dan refinement final.

### Dataset Final untuk Modeling

* `cleaned_training_data.csv`
* `cleaned_job_5.csv`
* `skill_mapping_dictionary.csv`
* `skkni_reference_clean.csv`

### Analisis Supply

* Technology mendominasi kategori resume.
* Soft skill seperti communication sangat dominan.
* Python dan SQL menjadi hard skill utama.

### Analisis Demand

* Jakarta mendominasi lowongan kerja digital.
* Industri membutuhkan skill PHP, MySQL, dan JavaScript lebih tinggi dibanding supply.

### Analisis SKKNI

* Coverage skill modern masih rendah.
* 80% lebih skill modern belum terpetakan secara langsung ke SKKNI.

---

# 5. Visualisasi & Explanatory Analysis

Dilakukan di `Visualisasi.ipynb`.

## Pertanyaan Bisnis 1 — Skill Gap

Visualisasi perbandingan skill supply vs demand menggunakan bar chart interaktif.

### Temuan

* PHP, MySQL, dan JavaScript mengalami shortage.
* Communication dan problem solving mengalami oversupply.

---

## Pertanyaan Bisnis 2 — Experience Mismatch

Visualisasi distribusi pengalaman pelamar dan kebutuhan industri.

### Temuan

* Fresh graduate mendominasi supply.
* Industri lebih banyak membutuhkan level Junior hingga Senior.

---

## Pertanyaan Bisnis 3 — Coverage SKKNI

Analisis cakupan kompetensi modern terhadap unit SKKNI.

### Temuan

* Banyak framework dan tools modern belum memiliki unit kompetensi nasional.

---

# 6. Persiapan Data untuk Modeling

## Dataset Final yang Digunakan

| Dataset                        | Fungsi                        |
| ------------------------------ | ----------------------------- |
| `cleaned_training_data.csv`    | Data supply resume pelamar    |
| `cleaned_job_5.csv`            | Data demand industri          |
| `skill_mapping_dictionary.csv` | Semantic bridge skill         |
| `skkni_reference_clean.csv`    | Referensi kompetensi nasional |

Dataset final merupakan dataset hasil cleaning dan refinement yang digunakan secara langsung dalam dashboard dan eksperimen modeling.

---

# 7. Dashboard Streamlit

Dashboard PELET dibangun menggunakan Streamlit dan Plotly.

## Halaman Dashboard

| Halaman            | Isi                             |
| ------------------ | ------------------------------- |
| Overview           | KPI dan insight utama           |
| Job Market         | Analisis pasar kerja            |
| Skill Demand       | Analisis supply vs demand skill |
| SKKNI Gap Analysis | Analisis coverage SKKNI         |

Link dashboard : **https://data-science-jtqg8peajwjqqksibr9pji.streamlit.app/**
---

# 8. A/B Testing

Eksperimen dilakukan untuk membandingkan:

| Grup A                     | Grup B        |
| -------------------------- | ------------- |
| TF-IDF + Cosine Similarity | Sentence-BERT |

## Hasil

| Model         | Accuracy |
| ------------- | -------- |
| TF-IDF        | 73,75%   |
| Sentence-BERT | 77,25%   |

## Kesimpulan

Sentence-BERT terbukti lebih efektif dalam semantic matching dibanding pendekatan keyword matching tradisional.

---

# 9. Laporan Teknis

Laporan teknis lengkap tersedia pada:

`Laporan.pdf`

Laporan mencakup:

* Problem Discovery
* Data Wrangling
* EDA
* Explanatory Analysis
* Dashboard Architecture
* A/B Testing
* Kesimpulan

---

# 10. Struktur Proyek

```bash
Data-Science/
│
├── dashboard/                              # Modul untuk visualisasi antarmuka pengguna
│   ├── dashboard.py                        # Script utama untuk menjalankan aplikasi dashboard
│   ├── data_loader.py                      # Script untuk membaca dan memproses awal data dashboard
│   └── logopelet.png                       # Aset gambar/logo untuk kebutuhan visual dashboard
│
├── data/                                   # Penyimpanan seluruh dataset proyek
│   ├── raw/                                # Dataset mentah sebelum diproses (Data Asli)
│   │   ├── Dataset_pekerjaan.csv
│   │   ├── all_job_post.csv
│   │   ├── job_roles.csv
│   │   ├── skills_list.csv
│   │   └── training_data.csv
│   │
│   └── clean/                              # Dataset hasil pembersihan dan siap analisis
│       ├── cleaned_job_5.csv
│       ├── cleaned_skills_list.csv
│       ├── cleaned_training_data.csv
│       ├── data_pekerjaan.csv
│       ├── skill_mapping_dictionary.csv
│       └── skkni_reference_clean.csv
│
├── dataset/                                # Salinan dataset terpilih (deployment/keperluan khusus)
│   ├── cleaned_job_5.csv
│   ├── cleaned_training_data.csv
│   ├── skill_mapping_dictionary.csv
│   └── skkni_reference_clean.csv
│
├── notebooks/                              # Berkas eksperimen dan analisis data (Jupyter Notebook)
│   ├── EDA_1.ipynb                         # Analisis Data Eksploratif tahap awal
│   ├── EDA_2_final.ipynb                   # Analisis Data Eksploratif tahap akhir/final
│   ├── ab_testing.ipynb                    # Notebook untuk eksperimen dan pengujian A/B
│   ├── datawrangling_kaggle.ipynb          # Pembersihan data yang bersumber dari Kaggle
│   ├── datawrangling_jobstreet.ipynb       # Pembersihan data lowongan dari Jobstreet
│   └── datawrangling_skkni.ipynb           # Pembersihan datastandar kompetensi SKKNI
│
├── Laporan.pdf				                 # Berisi penjelasan analisis
├── requirements.txt                        # Daftar dependensi library Python proyek 
└── README.md                               # Dokumentasi ringkas mengenai proyek dan panduan instalasi
```

## Penjelasan Struktur Dataset

### Folder `dataset/raw`

Berisi dataset mentah yang belum melalui proses cleaning dan preprocessing.

* `Data_pekerjaan.csv` → dataset hasil scraping SKKNI.
* `jobstreet-vacancy-dataset.csv` → dataset lowongan kerja Indonesia dari GitHub publik
* `job_role_skill_dataset.csv` → dataset skill dan job role dari Kaggle.

### Folder `dataset/clean`

Berisi dataset final hasil cleaning dan refinement yang digunakan langsung pada modeling dan dashboard.

---

## Catatan Dataset GitHub

Dataset `jobstreet-vacancy-dataset.csv` tidak di-upload ke GitHub repository karena ukuran file melebihi batas upload GitHub (>25 MB).

---

# 11. Cara Menjalankan

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Menjalankan Dashboard

```bash
streamlit run dashboard/dashboard.py
```

## Menjalankan A/B Testing

```bash
python notebooks/ab_testing.ipynb
```

---

# Requirements

```txt
streamlit
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
scipy
sentence-transformers
```

---

# Tim Data Science

| Nama                 | ID             | Role           |
| -------------------- | -------------- | -------------- |
| Sukma Novianti Tulak | CDCC237D6X1337 | Data Scientist |
| Ai Irma Anjelina     | CDCC237D6X1441 | Data Scientist |

---

Proyek ini dikembangkan sebagai bagian dari **Capstone Project PELET — AI Semantic Matching Berbasis SKKNI**, Coding Camp 2026 powered by DBS Foundation.
