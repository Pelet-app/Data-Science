# PELET — Data Science
**PELET (Pencari Lowongan Efektif & Tepat): AI Semantic Matching Berbasis SKKNI**
Coding Camp 2026 powered by DBS Foundation | Tim CC26-PSU060

---

## Daftar Isi

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

## 1. Problem Discovery

### Permasalahan

Indonesia menghadapi masalah *skill mismatch* yang kronis antara kompetensi pencari kerja dan kebutuhan nyata industri. Berdasarkan data BPS (2024), tingkat pengangguran tertinggi justru berasal dari lulusan SMK dan pendidikan tinggi, mencapai 8,62%. Sebanyak 50% tenaga kerja Indonesia mengalami *vertical mismatch* (Mandiri Institute, 2025), dan 46% perusahaan di Indonesia kesulitan menemukan kandidat yang sesuai (Transcon Indonesia, 2025).

Sistem pencocokan kerja yang ada saat ini masih mengandalkan *keyword matching* yang kaku, dan Standar Kompetensi Kerja Nasional Indonesia (SKKNI) belum terintegrasi secara optimal dengan sistem rekrutmen digital. Akibatnya, kandidat berkualitas bisa terlewat hanya karena perbedaan penulisan istilah, bukan karena tidak kompeten.

### Solusi Utama

Membangun sistem analisis dan pencocokan kerja berbasis AI bernama **PELET (Pencari Lowongan Efektif & Tepat)** yang bertugas untuk:
- Mengagregasi profil resume pelamar kerja secara massal dan mengekstrak entitas skill-nya.
- Memetakan kebutuhan riil lowongan pekerjaan di Indonesia berdasarkan data pasar nyata.
- Menghubungkan secara langsung *international skills* (kebutuhan industri) terhadap kode unit kompetensi nasional (SKKNI) menggunakan pendekatan *semantic matching* berbasis Sentence-BERT.
- Mengukur persentase kecocokan kompetensi kandidat serta menampilkan visualisasi peta kesenjangan (*skill gap*) secara interaktif dalam bentuk dashboard Streamlit.
- Memvalidasi secara ilmiah keunggulan pendekatan semantik dibandingkan *keyword matching* konvensional melalui A/B Testing.

---

## 2. Pertanyaan Bisnis

Pertanyaan bisnis didefinisikan secara terukur untuk memandu seluruh proses analisis:

| No | Pertanyaan Bisnis | Dataset | Metrik |
|---|---|---|---|
| 1 | Bagaimana kesenjangan (gap) antara rumpun skill digital yang dimiliki pelamar dengan kualifikasi skill yang paling banyak dibutuhkan industri? | `cleaned_training_data.csv` + `cleaned_job_5.csv` | Frekuensi kemunculan skill di resume vs lowongan (per 1.000 entri) |
| 2 | Bagaimana kecocokan profil pengalaman kerja para pencari kerja jika disandingkan dengan ekspektasi minimum dari industri? | `cleaned_training_data.csv` + `cleaned_job_5.csv` | Distribusi `exp_bucket` supply vs demand (%) |
| 3 | Seberapa luas cakupan unit kompetensi SKKNI dalam memetakan skill digital modern yang berkembang di pasar kerja saat ini? | `skill_mapping_dictionary.csv` + `skkni_reference_clean.csv` | Persentase status Terpetakan, Parsial, dan Belum Dipetakan |

---

## 3. Data Wrangling

### 3.1 Gathering Data

Data dikumpulkan dari berbagai sumber, **tidak menggunakan dataset siap pakai tanpa proses cleaning manual**:

| Dataset | Sumber | Metode Pengumpulan | File |
|---|---|---|---|
| SKKNI | Website resmi SKKNI | Web scraping manual + filter + cleaning manual | `skkni_reference_clean.csv` |
| Lowongan Kerja Indonesia | Glints Indonesia (via GitHub publik) | Download dataset publik + filter manual + cleaning manual | `cleaned_job_5.csv` |
| Data Resume (Training) | Kaggle | Dataset profil resume dan skill teks sintetis (10.000 entri) | `cleaned_training_data.csv` |
| Kamus Pemetaan Skill | Hasil Pemetaan Mandiri | Kamus relasi skill internasional vs kode unit SKKNI | `skill_mapping_dictionary.csv` |

### 3.2 Assessing Data

Evaluasi kualitas, tipe data, dan integritas dilakukan secara ketat pada tahap awal:

**SKKNI (`skkni_reference_clean.csv`):**
- Terdiri dari 662 baris dengan kolom `Kode Unit`, `Judul Unit`, `Elemen Kompetensi`, dan `Jabatan`.
- Ditemukan inkonsistensi penulisan nama jabatan target (misalnya variasi teks seperti `'Cyber security'` dan `'Security Eng.'`).
- Hanya mencakup sekitar 40 dari 194 jabatan unik yang dibutuhkan industri digital saat ini — jabatan mutakhir seperti DevOps Engineer, Data Scientist, dan AI Engineer belum terakomodasi sepenuhnya.

**Lowongan Kerja (`cleaned_job_5.csv`):**
- Terdiri dari 2.516 baris lowongan dengan 18 kolom informasi pasar.
- Kolom `job_level` kosong pada 88,1% baris dan kolom `min_salary`/`max_salary` kosong pada 80,5% baris (*inherent sparsity* dari sumber data).
- Kolom `location` berisi nama alamat jalan atau wilayah mikro yang terlalu spesifik sehingga perlu normalisasi ke level kota.
- Kolom `requirement` berisi teks tidak berstruktur dengan banyak karakter spesial (`\r`, `\n`, lambang bullet poin).

**Data Resume (`cleaned_training_data.csv`):**
- Terdiri dari 10.000 data profil pelamar kerja hasil generator sintetis.
- Kolom `skills_clean` diekspor dalam bentuk string mentah dari representasi list Python (contoh: `"['python', 'sql']"`), sehingga tidak terbaca sebagai tipe data *Iterable List* secara langsung.

**Kamus Pemetaan Skill (`skill_mapping_dictionary.csv`):**
- Terdiri dari 114 baris relasi kompetensi internasional terhadap status nasional.
- Seluruh kolom `Kode_Unit_SKKNI` dan `Judul_Unit_SKKNI` awalnya kosong — mencerminkan kondisi riil bahwa belum ada jembatan konseptual resmi antara standar internasional dan SKKNI nasional. Pemetaan dilakukan secara mandiri oleh tim.

### 3.3 Cleaning Data

**Dataset Lowongan:**
- Kolom `type_of_work` yang null diimputasi dengan nilai `"Tidak Diketahui"`.
- Kolom `currency` yang null diimputasi dengan `"Tidak Diketahui"`.
- Dibuat kolom baru `has_salary` (boolean) untuk menandai baris yang memiliki informasi gaji, menghindari bias analisis.
- Dibuat kolom `salary_mid` sebagai rata-rata `min_salary` dan `max_salary`.
- Ditambahkan kolom `kota` hasil normalisasi dari kolom `location` menggunakan `CITY_MAP`.

**Dataset Resume:**
- Kolom `skills_clean` yang berformat string list diparse menjadi list Python menggunakan `ast.literal_eval`.
- Dibuat kolom `skill_count` untuk menghitung jumlah skill per kandidat.
- Dibuat kolom `exp_bucket` untuk mengelompokkan pengalaman kerja ke dalam 5 kategori terstruktur.

**Pembersihan Umum:**
- Menghilangkan karakter *break-line* (`\r`, `\n`) dan simbol tak standar pada teks SKKNI dan kolom kualifikasi lowongan.
- Normalisasi seluruh teks ke *lowercase* untuk menghindari duplikasi akibat perbedaan kapitalisasi.

### 3.4 Integrasi & Transformasi Lanjutan (`data_loader.py`)

`data_loader.py` berfungsi sebagai **Data Pipeline & Feature Engineering** yang mengintegrasikan 4 dataset agar terhubung secara dinamis dengan dashboard:

- **Penyelarasan Geografis Multilevel:** Fungsi `extract_city(loc)` dengan kamus `CITY_MAP` mendeteksi variasi penulisan wilayah mikro dan menyatukannya ke kota induk yang seragam.
- **Object Type Transformation:** `ast.literal_eval` di dalam `parse_skills()` mengubah string skill menjadi list Python asli agar bisa dihitung frekuensinya dengan `Counter`.
- **Ekstraksi Skill Berbasis Kamus (Regex Matching):** Fungsi `extract_skills_with_regex()` menggunakan `re.escape()` dan *word boundary* `\b` untuk mencocokkan skill dari kolom `requirement` lowongan terhadap kamus `skill_mapping_dictionary.csv` — memastikan hanya nama skill valid yang diekstrak, bukan token kata acak.
- **Kategorisasi Pengalaman (Feature Engineering):** `pd.cut` terhadap kolom `Experience Years` menghasilkan kolom `exp_bucket` dengan 5 label: `Fresh (<1)`, `Junior (1–3)`, `Mid (3–5)`, `Senior (5–10)`, `Expert (10+)`.
- **Path Dinamis:** Seluruh path dataset menggunakan `pathlib.Path(__file__).resolve().parent` sehingga tidak bergantung pada direktori tempat script dijalankan.

---

## 4. Exploratory Data Analysis (EDA)

Dilakukan di `notebooks/EDA.ipynb`. Seluruh temuan dianalisis secara tekstual dengan narasi markdown dan dipastikan selaras dengan visualisasi pada dashboard.

### Analisis Sisi Supply (Resume Pelamar)
- Sektor **Technology** menguasai pasokan terbesar dengan 2.511 resume (25,1%), diikuti Finance (12,0%) dan Healthcare (9,8%).
- Rata-rata pelamar memiliki **8,3 skill** per resume dengan distribusi mendekati normal (*slight right-skew*).
- *Soft skills* universal mendominasi: `communication` (4.821 kemunculan) dan `problem solving` (4.103 kemunculan). Di sisi *hard skills*, `SQL` (3.201) dan `Python` (2.287) memimpin.

### Analisis Sisi Demand (Pasar Lowongan)
- **Jakarta** mendominasi 73,4% dari seluruh lowongan (1.847 lowongan), diikuti Bandung (189) dan Surabaya (124) — mencerminkan sentralisasi industri digital di ibu kota.
- Median gaji IDR menyentuh **Rp 8,5 juta/bulan** dengan rentang sangat lebar (Rp 2–150 juta), mengkonfirmasi premium kompensasi untuk level Senior/Expert.
- Posisi terpopuler: Business Development Manager (143), Data Analyst (127), Fullstack Developer (98).

### Analisis Dokumen Regulasi SKKNI
- Profesi **Programmer** memiliki beban standardisasi terbanyak dengan 37 unit kompetensi, diikuti Keuangan (28 unit) dan Cyber Security (25 unit).
- Dari 114 skill internasional: **14,9% terpetakan konseptual**, **4,4% terpetakan parsial**, dan **80,7% belum terpetakan** sama sekali ke unit SKKNI.

---

## 5. Visualisasi & Explanatory Analysis

Dilakukan di `notebooks/Visualisasi.ipynb`. Dashboard **PELET** menjawab 3 pertanyaan bisnis melalui grafik interaktif berbasis Plotly:

### Pertanyaan Bisnis 1 — Skill Gap: Supply vs Demand
- **Bar Chart komparatif** yang menampilkan frekuensi skill dinormalisasi per 1.000 entri dari sisi resume (supply) dan lowongan (demand).
- **Temuan kritis:** Industri sangat membutuhkan `PHP` (gap: +222,5), `MySQL` (+159,3), dan `JavaScript` (+146,6) namun ketersediaannya di resume sangat minim. Sebaliknya, `communication` dan `problem solving` jauh oversupply di resume namun hampir tidak muncul di persyaratan lowongan.

### Pertanyaan Bisnis 2 — Experience Mismatch
- **Bar Chart distribusi** level pengalaman pelamar vs ekspektasi industri dalam persentase.
- **Temuan kritis:** Pelamar Fresh Graduate (<1 tahun) mendominasi supply, sementara industri paling banyak membutuhkan profil Junior–Mid (1–3 tahun) hingga Senior (5–10 tahun) — mengindikasikan *vertical mismatch* struktural.

### Pertanyaan Bisnis 3 — Cakupan SKKNI
- **Bar Chart** jumlah unit kompetensi dan elemen per jabatan pada 5 rumpun utama (Programmer, UI/UX, Data Analyst, Cyber Security, Keuangan).
- **Stacked bar chart** status pemetaan skill: 80,7% skill modern berstatus *Belum Dipetakan*, termasuk ekosistem DevOps (Kubernetes, Docker, Terraform) dan framework modern (React, Node.js, Flutter).

---

## 6. Persiapan Data untuk Modeling

### Data Dictionary

**`skkni_reference_clean.csv`** — 662 baris

| Kolom | Tipe Data | Keterangan |
|---|---|---|
| `Kode Unit` | string | Kode registrasi unit kompetensi resmi SKKNI |
| `Judul Unit` | string | Deskripsi unit kompetensi |
| `Elemen Kompetensi` | string | Detail elemen pekerjaan di dalam unit terkait |
| `Jabatan` | string | Nama posisi/jabatan pekerjaan yang dinormalisasi |

**`skill_mapping_dictionary.csv`** — 114 baris

| Kolom | Tipe Data | Keterangan |
|---|---|---|
| `Skill_Internasional_Kaggle` | string | Nama skill dalam bahasa Inggris |
| `Kode_Unit_SKKNI` | string | Kode unit SKKNI yang sesuai (hasil pemetaan mandiri) |
| `Judul_Unit_SKKNI` | string | Judul unit SKKNI yang sesuai |
| `Kecocokan` | string | Status pemetaan: `Cocok`, `Parsial`, atau `Belum Dipetakan` |

**`cleaned_training_data.csv`** — 10.000 baris

| Kolom | Tipe Data | Keterangan |
|---|---|---|
| `Resume ID` | string | ID unik resume |
| `Resume Text` | string | Teks resume lengkap |
| `Education` | string | Latar belakang pendidikan |
| `Experience Years` | int | Lama pengalaman kerja (tahun) |
| `Skills` | string | Daftar skill dipisah pipe |
| `Job Role` | string | Jabatan/peran pekerjaan |
| `Category` | string | Kategori bidang pekerjaan |
| `resume_text_clean` | string | Teks resume setelah cleaning |
| `skills_clean` | string | Skill dalam format list Python |

**`cleaned_job_5.csv`** — 2.516 baris

| Kolom | Tipe Data | Keterangan |
|---|---|---|
| `title` | string | Judul posisi pekerjaan dari lowongan |
| `company` | string | Nama perusahaan yang membuka lowongan |
| `location` | string | Lokasi penempatan kerja (raw) |
| `type_of_work` | string | Jenis kontrak kerja (Full Time, Contract, dll) |
| `requirement` | string | Persyaratan kompetensi dan kualifikasi dari industri |
| `currency` | string | Mata uang gaji |
| `min_salary` | float | Gaji minimum (juta) |
| `max_salary` | float | Gaji maksimum (juta) |
| `min_work_experience` | int | Minimum pengalaman kerja (tahun) |
| `max_work_experience` | int | Maksimum pengalaman kerja (tahun) |

### Fitur Hasil Feature Engineering

| Fitur Baru | Sumber | Keterangan |
|---|---|---|
| `kota` | `location` | Kota yang dinormalisasi via `CITY_MAP` |
| `has_salary` | `min_salary`, `max_salary` | Flag boolean ketersediaan info gaji |
| `salary_mid` | `min_salary`, `max_salary` | Nilai tengah rentang gaji |
| `req_skills` | `requirement` | List skill valid hasil regex matching vs kamus |
| `req_skill_count` | `req_skills` | Jumlah skill yang ditemukan per lowongan |
| `exp_mid` | `min_work_experience`, `max_work_experience` | Rata-rata pengalaman yang diminta |
| `skill_count` | `skills_clean` | Jumlah skill per resume |
| `exp_bucket` | `Experience Years` | Kategori level pengalaman (5 bucket) |

---

## 7. Dashboard Streamlit

Dashboard interaktif **PELET** dikembangkan menggunakan Streamlit + Plotly untuk menyajikan hasil analisis gap kompetensi secara visual dan real-time.

### Halaman Dashboard

| Halaman | Konten |
|---|---|
| **Overview** | KPI cards (total lowongan, resume, unit SKKNI, coverage rate), distribusi kategori resume, distribusi jabatan SKKNI, key insights |
| **Job Market** | Filter kota/tipe/mata uang, distribusi geografis lowongan, tipe pekerjaan, distribusi gaji IDR, experience mismatch supply vs demand, top 20 judul lowongan |
| **Skill Demand** | Top skills dari resume (per kategori), top skills dari requirement lowongan (regex-based), heatmap skill per kategori, perbandingan supply vs demand per 1.000 entri |
| **SKKNI Gap Analysis** | Metric cards coverage rate, pie chart status pemetaan, bar chart skill terpetakan per jabatan SKKNI, tabel detail berwarna, bar chart skill demand vs status SKKNI |

### Deployment

Dashboard sudah di-deploy ke Streamlit Cloud dan dapat diakses secara publik pada tautan berikut:

**Link:** *(isi tautan deployment Streamlit Cloud di sini)*

### Cara Menjalankan Lokal

```bash
pip install -r requirements.txt
streamlit run dashboard/dashboard1.py
```

---

## 8. A/B Testing

Dilakukan di `notebooks/ab_testing.ipynb` untuk memvalidasi secara ilmiah pemilihan Sentence-BERT sebagai *core matching engine* sistem PELET.

### Desain Eksperimen

| | Grup A (Baseline Control) | Grup B (Treatment) |
|---|---|---|
| **Model** | TF-IDF + Cosine Similarity | Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`) |
| **Pendekatan** | Keyword matching berbasis frekuensi token | Semantic matching berbasis sentence embedding (d=384) |
| **Dataset** | `cleaned_training_data.csv` — 10.000 resume | Sama |

### Hipotesis

- **H0:** Tidak terdapat perbedaan performa yang signifikan antara TF-IDF dan Sentence-BERT.
- **H1:** Sentence-BERT menghasilkan performa pencocokan yang secara signifikan lebih tinggi (α = 0,05).

### Hasil

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Grup A — TF-IDF | 73,75% | 100,00% | 47,50% | 64,41% |
| Grup B — Sentence-BERT | 77,25% | 100,00% | 54,50% | 70,55% |

### Uji Statistik

- **Uji Normalitas (Shapiro-Wilk):** p-value = 0,0000 → data tidak normal → digunakan uji non-parametrik.
- **Mann-Whitney U Test:** U = 3204,00, p-value = 0,0000 → **Tolak H0**.
- **Effect Size (Cohen's d):** +0,4774 (kategori sedang–kuat).
- **Kesimpulan:** Keunggulan Sentence-BERT terbukti valid secara statistik — bukan hasil kebetulan. Sentence-BERT dipilih sebagai model utama sistem PELET.

### Cara Menjalankan

```bash
python notebooks/ab_testing.py
```

---

## 9. Laporan Teknis

Laporan teknis komprehensif tersedia dalam format PDF di `Laporan.pdf`.

Isi laporan mencakup:
- **Bab 1** — Problem Discovery & latar belakang skill mismatch di Indonesia
- **Bab 2** — Data Wrangling end-to-end (Gathering, Assessing, Cleaning)
- **Bab 3** — Exploratory Data Analysis (EDA) dengan insight per dataset
- **Bab 4** — Explanatory Analysis per pertanyaan bisnis
- **Bab 5** — Arsitektur Dashboard Streamlit & cara kerja `data_loader.py`
- **Bab 6** — Desain eksperimen, hasil, dan interpretasi statistik A/B Testing
- **Bab 7** — Kesimpulan, jawaban pertanyaan bisnis, dan rekomendasi

**File:** `Laporan.pdf`

---

## 10. Struktur Proyek

```
capstone_project/
│
├── dataset/
│   ├── cleaned_job_5.csv               # Dataset lowongan kerja Indonesia (2.516 baris)
│   ├── cleaned_training_data.csv       # Dataset profil resume pelamar kerja (10.000 baris)
│   ├── skill_mapping_dictionary.csv    # Kamus status kecocokan skill internasional vs SKKNI (114 skill)
│   └── skkni_reference_clean.csv       # Referensi unit kompetensi resmi SKKNI (662 baris)
│
├── dashboard/
│   ├── dashboard1.py                   # File utama aplikasi dashboard web Streamlit (4 halaman)
│   ├── data_loader.py                  # Modul pipeline data: parsing, feature engineering, caching
│   └── logopelet.png                   # Aset logo untuk sidebar dashboard
│
├── notebooks/
│   ├── EDA__1_.ipynb                   # Analisis eksploratif karakteristik awal data
│   ├── Visualisasi__1_.ipynb           # Explanatory analysis & visualisasi per pertanyaan bisnis
│   └── ab_testing__1_.ipynb            # Eksperimen komparatif TF-IDF vs Sentence-BERT
│
├── Laporan.pdf                         # Laporan teknis Data Science final (PDF)
├── requirements.txt                    # Dependensi library Python
└── README.md                           # Dokumentasi utama (file ini)
```

---

## 11. Cara Menjalankan

### Install semua dependensi

```bash
pip install -r requirements.txt
```

### Jalankan Dashboard

```bash
streamlit run dashboard/dashboard1.py
```

### Jalankan A/B Testing

```bash
python notebooks/ab_testing__1_.ipynb
```

### Isi `requirements.txt`

```
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

## Tim Data Science

| Nama | ID | Peran |
|---|---|---|
| Sukma Novianti Tulak | CDCC237D6X1337 | Data Scientist |
| Ai Irma Anjelina | CDCC237D6X1441 | Data Scientist |

Proyek ini dikembangkan sebagai bagian dari **Capstone Project PELET — SkillBridge AI**, Coding Camp 2026 powered by DBS Foundation, Tim CC26-PSU060.
