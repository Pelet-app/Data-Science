# SkillBridge AI — Data Science

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
Banyak pencari kerja di Indonesia kesulitan mengetahui jabatan mana yang paling cocok dengan skill yang mereka miliki. Di sisi lain, standar kompetensi resmi (SKKNI) belum tentu selaras dengan kebutuhan industri nyata saat ini, sehingga pelamar sulit mengukur kesiapan mereka secara objektif.

### Solusi Utama
Membangun sistem rekomendasi pekerjaan berbasis AI (**SkillBridge AI**) yang:
- Menerima CV/Resume pelamar kerja.
- Menganalisis dan mengekstrak skill dari resume secara terstruktur.
- Memetakan kebutuhan skill industri nyata (Kaggle & JobStreet) terhadap standar kompetensi nasional (SKKNI).
- Mengukur persentase kecocokan skill (relevansi) dan menampilkan visualisasi gap secara interaktif.
  
---

## 2. Pertanyaan Bisnis

Pertanyaan bisnis didefinisikan secara terukur untuk memandu seluruh proses analisis:

| No | Pertanyaan Bisnis | Dataset | Metrik |
|---|---|---|---|
| 1 | Jabatan SKKNI mana yang memiliki jumlah unit kompetensi unik paling tinggi dibandingkan jabatan lainnya? | SKKNI | Jumlah unit kompetensi unik per jabatan |
| 2 | Skill apa yang paling sering muncul pada requirement lowongan kerja industri di Indonesia? | JobStreet Indonesia | Frekuensi kemunculan skill di lowongan kerja |
| 3 | Bagaimana distribusi pemetaan status kecocokan skill internasional terhadap unit kompetensi SKKNI? | Skill Mapping Dictionary | Persentase status Terpetakan, Parsial, dan Belum Dipetakan |

---

## 3. Data Wrangling

### 3.1 Gathering Data

Data dikumpulkan dari berbagai sumber, **tidak menggunakan dataset siap pakai tanpa proses cleaning manual**:

| Dataset | Sumber | Metode Pengumpulan | File |
|---|---|---|---|
| SKKNI | Website resmi SKKNI | Web scraping manual + filter + cleaning manual | `skkni_reference_clean.csv` |
| Lowongan Kerja Indonesia | JobStreet Indonesia | Download dataset publik + filter manual + cleaning manual | `cleaned_job_5.csv` |
| Data Resume (Training) | Kaggle | Dataset profil resume dan skill teks | `cleaned_training_data.csv` |
| Kamus Pemetaan Skill | Hasil Pemetaan Mandiri | Kamus relasi skill industri vs kode unit SKKNI | `skill_mapping_dictionary.csv` |

### 3.2 Assessing Data

Evaluasi kualitas dan struktur data dilakukan pada setiap dataset:

**SKKNI (`skkni_reference_clean.csv`):**
- Terdiri dari 662 baris data unit kompetensi terstruktur.
- Ditemukan: variasi penulisan nama jabatan yang sangat beragam (misal: 'Programmer', 'Web Developer', 'Cyber security', 'Data Analyst').
- Ditemukan: beberapa unit kompetensi yang berulang karena memiliki elemen kompetensi yang berbeda.

**Lowongan Kerja (`cleaned_job_5.csv`):**
- Terdiri dari 2.516 baris lowongan pekerjaan dengan 18 kolom informasi.
- Ditemukan: Kolom `requirement` memiliki format teks tidak beraturan dengan noise karakter khsusus.
- Ditemukan: Lokasi kerja masih berupa nama jalan/area spesifik, bukan nama kota besar.

**Data Resume (`cleaned_training_data.csv`):**
- Terdiri dari 10.000 data resume pelamar kerja.
- Ditemukan: Kolom `Skills` asli berupa gabungan teks dengan separator pipa (`|`), serta kolom `skills_clean` yang tersimpan dalam format string dari sebuah list (`"['skill1', 'skill2']"`).
  
### 3.3 Cleaning Data

Seluruh proses cleaning dilakukan **secara manual** tanpa menggunakan dataset yang sudah siap pakai:

Seluruh proses cleaning dilakukan **secara manual dan terprogram** lewat skrip `data_loader.py`:

**SKKNI (`skkni_reference_clean.csv`)**
- Dilakukan normalisasi teks pada kolom `Jabatan` agar seragam saat difilter.
- Standarisasi teks (case-folding) pada judul unit dan elemen kompetensi.

**Lowongan Kerja (`cleaned_job_5.csv`)**
- Dilakukan ekstraksi kota menggunakan dictionary pemetaan wilayah melalui fungsi `extract_city` (misalnya mengubah kata "jakarta selatan", "dki jakarta", "jakarta raya" menjadi satu entitas seragam yaitu `"Jakarta"`).
- Pembersihan string kosong dan pengisian nilai default `"Lainnya"` jika lokasi tidak terdeteksi.

**Data Resume (`cleaned_training_data.csv`)**
- Mengekstrak kolom `skills_clean` dari format literal string kembali menjadi tipe data List aktual di Python menggunakan fungsi `ast.literal_eval`.
- Melakukan segmentasi level pengalaman kerja pelamar (`exp_bucket`) ke dalam beberapa kategori terukur: *Fresh (<1)*, *Junior (1–3)*, *Mid (3–5)*, *Senior (5–10)*, dan *Expert (10+)* menggunakan fungsi `pd.cut`.
  
---

## 4. Exploratory Data Analysis (EDA)

EDA dilakukan pada seluruh dataset utama untuk mendapatkan insight mendalam sebelum diolah ke dalam dashboard visual.

### Insight Dataset SKKNI & Kamus Skill
- Jabatan teknis tingkat tinggi (seperti *Cyber Security Specialists*, *Software Architects*, *Data Scientists*) memiliki jumlah unit kompetensi yang lebih banyak dan kompleks.
- Berdasarkan `skill_mapping_dictionary.csv`, banyak *soft skills* utama industri (seperti *leadership*, *communication*, *problem solving*, *critical thinking*) berstatus **Belum Dipetakan** ke unit kompetensi SKKNI yang spesifik karena sifat SKKNI yang cenderung berfokus pada pekerjaan *hard skills/prosedural*.

### Insight Dataset Lowongan Kerja & Resume
- Dari 2.516 lowongan kerja, sebaran lokasi lowongan kerja di Indonesia masih didominasi secara masif oleh wilayah DKI Jakarta dan Bandung.
- Pada dataset resume (10.000 data), tren skill teknis seperti Python, SQL, dan JavaScript memiliki frekuensi kemunculan tertinggi pada kategori teknologi informasi.

---

## 5. Visualisasi & Explanatory Analysis

Visualisasi di dalam proyek ini dirancang langsung untuk menjawab seluruh pertanyaan bisnis secara komprehensif:

### Kompleksitas Jabatan SKKNI
- Menampilkan grafik jumlah unit kompetensi unik per kategori jabatan kerja untuk mengidentifikasi jabatan mana yang memiliki standar kompetensi nasional paling kompleks.

### Skill Demand & Analisis Tren Industri
- Bar Chart yang menampilkan Top N Skill yang paling dicari oleh perusahaan di Indonesia berdasarkan data JobStreet.
- Distribusi tingkat pengalaman kerja (`exp_bucket`) yang diminta pasar kerja untuk mengukur serapan tenaga kerja lulusan baru (*fresh graduate*).

### Gap Analysis (SKKNI vs Industri)
- Bar chart interaktif yang diberi kode warna tegas (*Terpetakan* = Hijau, *Parsial* = Kuning, *Belum Dipetakan* = Merah) untuk menunjukkan skill populer apa saja di industri saat ini yang belum diakomodasi secara eksplisit di dalam dokumen unit kompetensi SKKNI.

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

**`cleaned_job_5.csv`** — 2.516 baris

| Kolom | Tipe Data | Keterangan |
|---|---|---|
| `title` | string | Judul posisi pekerjaan dari lowongan kerja |
| `company` | string | Nama perusahaan yang membuka lowongan |
| `location` | string | Lokasi penempatan kerja |
| `type_of_work` | string | Jenis kontrak kerja (Penuh Waktu, Paruh Waktu, dll) |
| `requirement` | string | Persyaratan kompetensi dan kualifikasi dari industri |
| `min_work_experience` | float | Batas minimum pengalaman kerja dalam satuan tahun |
| `max_work_experience` | float | Batas maksimum pengalaman kerja dalam satuan tahun |
| `link` | string | Tautan/URL menuju lowongan asli |

---

## 7. Dashboard Streamlit

Dashboard interaktif **SkillBridge AI** dikembangkan untuk menyajikan hasil analisis gap antara kurikulum kompetensi nasional dengan kebutuhan riil pasar industri secara visual.

### Fitur Utama Dashboard
- **Sidebar Kontrol:** Meliputi pilihan filter Jabatan Pekerjaan secara dinamis, serta Slider untuk mengatur jumlah Top-N Skill yang ingin ditampilkan pada grafik analisis.
- **Metric Cards:** Menampilkan metrik ringkasan agregasi data secara real-time (Total Lowongan Teranalisis, Total Profil Resume, Jumlah Jabatan Terpetakan).
- **Interactive Graphs:** Grafik interaktif menggunakan library Plotly (Bar Chart Berwarna Status SKKNI, Pie Chart Distribusi Tipe Kerja, Scatter Plot).
- **Insight Boxes:** Penjelasan naratif otomatis di bawah grafik untuk membantu pengguna awam memahami kesimpulan dari visualisasi data yang ditampilkan.

### Deployment
Dashboard sudah di-deploy ke Streamlit Cloud dan dapat diakses secara publik pada tautan berikut:

link : https://latihan-mutand2svg2mzyvqypsmgc.streamlit.app/

### Cara Menjalankan Lokal
```bash
pip install streamlit streamlit-autorefresh pandas matplotlib seaborn
streamlit run dashboard.py
```

---

## 8. A/B Testing

Eksperimen untuk membuktikan secara statistik metode matching mana yang lebih akurat dalam mencocokkan CV pelamar dengan jabatan yang tersedia.

### Desain Eksperimen

| | Group A | Group B |
|---|---|---|
| **Metode** | Keyword Matching | Semantic Matching (TF-IDF) |
| **Cara kerja** | Overlap kata kunci CV vs profil SKKNI | Cosine similarity TF-IDF |
| **Sampel** | 200 CV simulasi | 200 CV simulasi |

### Hasil

| Metrik | Group A (Keyword) | Group B (Semantic) |
|---|---|---|
| Mean Score | 0.0371 | **0.0656** |
| Match Rate | 30.0% | **51.0%** |
| Peningkatan | — | **+77%** |
| p-value | — | **0.000194** |
| Effect Size (Cohen's d) | — | 0.4842 (medium) |


### Kesimpulan
Tolak H0. Semantic Matching secara statistik **lebih baik** dari Keyword Matching (p = 0.000194 < α = 0.05). **Semantic Matching direkomendasikan sebagai metode utama SkillBridge AI.**

### Cara Menjalankan
```bash
python AB_Testing/ab_testing.py
```
![Hasil A/B Testing](AB_Testing/AB_testing.png)
---

## 9. Laporan Teknis

Laporan teknis komprehensif mencakup seluruh tahapan proyek mulai dari Problem Discovery hingga hasil akhir, tersedia dalam format PDF.

**`belum ada, nanti kami isi`**

Isi laporan mencakup:
- Problem Discovery & definisi solusi
- Pertanyaan bisnis yang terukur
- Proses Data Wrangling end-to-end
- Hasil EDA + visualisasi
- Explanatory analysis per pertanyaan bisnis
- Persiapan data untuk modeling & feature engineering
- Hasil A/B Testing + interpretasi statistik
- Kesimpulan & rekomendasi

---

## 10. Struktur Proyek

```
capstone_project/
capstone_project/
│
├── dataset/
│   ├── cleaned_job_5.csv              # Dataset lowongan kerja Indonesia (2.516 baris)
│   ├── cleaned_training_data.csv      # Dataset profil resume pelamar kerja (10.000 baris)
│   ├── skill_mapping_dictionary.csv   # Kamus status kecocokan skill internasional vs SKKNI
│   └── skkni_reference_clean.csv      # Referensi unit kompetensi resmi SKKNI (662 baris)
│
├── dashboard/
│   ├── data_loader.py                 # Script caching data, cleaning, dan preprocessing pandas
└── |── dashboard.py                   # File utama aplikasi dashboard web Streamlit
│
├── ab_tetsing                         # 
└── pdf                                # 
├── requirements.txt                   # Dependensi Python
└── README.md                          # Dokumentasi utama (file ini)

---

## 11. Cara Menjalankan

### Install semua dependensi
```bash
pip install -r requirements.txt
```

### Jalankan Dashboard
```bash
streamlit run dashboard.py
```

### Jalankan A/B Testing
```bash
python ab_testing.py
```

### Isi `requirements.txt`
```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

---

## Tim Data Science

Proyek ini dikembangkan sebagai bagian dari **Capstone Project — SkillBridge AI**.
