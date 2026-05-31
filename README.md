# Pelet — Data Science

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
Banyak pencari kerja di Indonesia kesulitan mengetahui jabatan mana yang paling cocok dengan kemampuan (*skill*) yang mereka miliki. Di sisi lain, standar kompetensi resmi nasional (SKKNI) bersifat kaku dan memerlukan waktu lama untuk diperbarui, sehingga belum tentu selaras secara *real-time* dengan kebutuhan industri nyata saat ini. Akibatnya, terjadi *missmatch* (ketidaksesuaian) yang tinggi antara suplai tenaga kerja dan kebutuhan pasar kerja (demand).

### Solusi Utama
Membangun sistem analisis kurikulum dan rekomendasi pekerjaan berbasis AI bernama **Pelet** yang bertugas untuk:
- Mengagregasi profil resume pelamar kerja secara massal dan mengekstrak entitas skill-nya.
- Memetakan kebutuhan riil lowongan pekerjaan di Indonesia berdasarkan data aggregator pasar.
- Menghubungkan secara langsung *international skills* (kebutuhan industri) terhadap kode unit kompetensi nasional (SKKNI).
- Mengukur persentase kecocokan kompetensi kandidat serta menampilkan visualisasi peta kesenjangan (*skill gap*) secara interaktif dalam bentuk dashboard.

---
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

Evaluasi kualitas, tipe data, dan integritas dilakukan secara ketat pada tahap awal:

**SKKNI (`skkni_reference_clean.csv`):**
- Terdiri dari 662 baris dengan kolom `Kode Unit`, `Judul Unit`, `Elemen Kompetensi`, dan `Jabatan`.
- Ditemukan inkonsistensi penulisan nama jabatan target (misalnya terdapat variasi teks seperti 'Cyber security' dan 'Security Eng.').

**Lowongan Kerja (`cleaned_job_5.csv`):**
- Terdiri dari 2.516 baris lowongan dengan 18 kolom informasi pasar.
- Masalah utama: Kolom `location` berisi nama alamat jalan, gedung, atau wilayah mikro yang terlalu spesifik, sehingga mustahil dikelompokkan secara langsung tanpa pembersihan makro.
- Kolom `requirement` berisi teks tidak berstruktur dengan banyak karakter spesial (`\r`, `\n`, lambang bullet poin).

**Data Resume (`cleaned_training_data.csv`):**
- Terdiri dari 10.000 data profil pelamar kerja.
- Masalah utama: Kolom `skills_clean` dieksport dalam bentuk string mentah dari representasi sebuah list (format objek literal string, contoh: `"['python', 'sql']"`), sehingga tidak terbaca sebagai tipe data *Iterable List* oleh Python.

**Kamus Pemetaan Skill (`skill_mapping_dictionary.csv`):**
- Terdiri dari 114 baris relasi kompetensi internasional terhadap status nasional.
- Ditemukan banyak nilai kosong (*Missing Values*) pada kolom `Kode_Unit_SKKNI` dan `Judul_Unit_SKKNI` khusus untuk kategori keahlian berupa *soft skills* dan alat visualisasi modern.
  
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

Seluruh temuan data dianalisis secara tekstual dan dipastikan selaras dengan visualisasi pada dashboard:

### Analisis Sisi Regulasi (SKKNI)
- Agregasi data pada berkas `skkni_reference_clean.csv` menunjukkan bahwa profesi di bawah rumpun teknologi canggih seperti *Software Architect*, *Cyber Security Specialists*, dan *Data Scientist* memiliki rata-rata elemen kompetensi yang jauh lebih banyak per jabatan. Standar kompetensinya berfokus penuh pada aspek prosedural teknis operasional (*hard skills*).

### Analisis Sisi Pasar (JobStreet & Resume)
- Dari total 2.516 lowongan kerja aktif, pusat penyerapan tenaga kerja terbesar di Indonesia terkonsentrasi sangat padat di wilayah **DKI Jakarta** dan **Bandung**, dengan sistem kerja dominan berupa **Penuh Waktu (Full-time)**.
- Dari 10.000 data profil pelamar kerja, kelompok pencari kerja terbesar didominasi oleh kelas *Fresh Graduate* (<1 tahun) dan *Junior* (1-3 tahun), mengindikasikan tingginya tingkat kompetisi di level masuk kerja (*entry-level*).

---

## 5. Visualisasi & Explanatory Analysis

Dashboard **Pelet** menjawab pertanyaan bisnis melalui sajian grafik yang interaktif:

### 1. Kompleksitas Jabatan SKKNI (Menjawab Pertanyaan 1)
- Ditampilkan dalam bentuk **Bar Chart** yang menghitung frekuensi kode unit unik per jabatan kerja. Terlihat visualisasi grafik vertikal yang menunjukkan tingkat kedalaman materi uji kompetensi pada masing-masing posisi.

### 2. Geografis & Karakter Kontrak Pasar Kerja (Menjawab Pertanyaan 2)
- Menggunakan **Pie Chart** interaktif untuk melihat porsi perbandingan status kerja industri (Penuh waktu, Kontrak, dll).
- Disandingkan dengan **Bar Chart Horizontal** hasil ekstraksi lokasi kerja untuk memperlihatkan dengan jelas ketimpangan ketersediaan lapangan kerja antar wilayah di Indonesia.

### 3. Distribusi Kesiapan Suplai Tenaga Kerja (Menjawab Pertanyaan 3)
- Menampilkan grafik distribusi kategorikal dari kolom fitur `exp_bucket`. Analisis ini membantu industri memetakan apakah stok pelamar kerja yang tersedia di pasar saat ini sesuai dengan kualifikasi pengalaman kerja yang dicari.

### 4. Analisis Kesenjangan / Skill Gap (Menjawab Pertanyaan 4)
- Berupa **Bar Chart Top 30 Skills** paling dicari di industri yang dilengkapi dengan penanda warna (*Color-coded*) berbasis data status `Kecocokan` dari kamus pemetaan skill.
- **Warna Hijau (Terpetakan):** Keahlian teknis inti seperti *Python*, *SQL*, *JavaScript* terbukti aman karena sudah diakomodasi dalam skema jabatan regulasi nasional.
- **Warna Merah (Belum Dipetakan):** Menyoroti keahlian tipe *soft skills* krusial (seperti *communication*, *leadership*, *problem solving*) serta teknologi mutakhir (*Figma*, *Git*) yang permintaannya sangat tinggi di lowongan kerja nyata namun belum tercatat secara eksplisit di dalam struktur dokumen unit kompetensi SKKNI.

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

link : __

### Cara Menjalankan Lokal
```bash
pip install streamlit streamlit-autorefresh pandas matplotlib seaborn
streamlit run dashboard.py
```

---

## 8. A/B Testing
_____belum ada____
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
```
---
### 11. Cara Menjalankan

### Install semua dependensi
```bash
pip install -r requirements.txt
```
---

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
