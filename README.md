# SkillBridge AI — Skill Gap Dashboard

Dashboard interaktif berbasis Streamlit untuk menganalisis kesenjangan kompetensi antara standar SKKNI (Standar Kompetensi Kerja Nasional Indonesia) dan kebutuhan industri nyata berdasarkan data lowongan kerja di Indonesia.

---

## Deskripsi Proyek

SkillBridge AI adalah aplikasi yang membantu pencari kerja menemukan pekerjaan yang sesuai dengan kompetensi dan kebutuhan industri.

> Dashboard ini merupakan bagian dari komponen **Data Science** proyek tersebut, yang berfokus pada analisis skill gap untuk menjawab pertanyaan berikut:
1. Bagaimana distribusi jumlah unit kompetensi dan elemen kompetensi pada setiap jabatan SKKNI yang dianalisis dalam dashboard?
2. Jabatan SKKNI mana yang memiliki jumlah unit kompetensi unik paling tinggi dibandingkan jabatan lainnya?
3. Skill apa yang paling sering muncul pada requirement lowongan kerja berdasarkan dataset JobStreet Indonesia?
4. Berapa tingkat relevansi kompetensi SKKNI terhadap kebutuhan skill industri berdasarkan kecocokan kompetensi dengan data lowongan kerja JobStreet Indonesia?

---

## Fitur Dashboard

### Overview Jabatan
Ringkasan seluruh jabatan dalam dataset SKKNI, mencakup total baris, jumlah unit kompetensi unik, dan elemen kompetensi unik per jabatan.

### Unit Kompetensi SKKNI
Visualisasi jumlah unit kompetensi tiap jabatan berdasarkan standar SKKNI. Menggambarkan kompleksitas masing-masing jabatan dan detail unit kompetensinya.

### Skill Demand Industri
Analisis skill yang paling sering diminta perusahaan di Indonesia, diekstrak dari kolom requirement dataset lowongan kerja. Dilengkapi distribusi tipe pekerjaan dan rata-rata pengalaman yang dibutuhkan.

### Gap Analysis
Perbandingan langsung antara kompetensi standar SKKNI dengan skill yang diminta industri. Menampilkan persentase relevansi per jabatan dan daftar skill yang belum terdeteksi di pasar kerja.

---

## Dataset

| Dataset | File | Sumber | Keterangan |
|---|---|---|---|
| SKKNI | `5_data_pekerjaan.csv` | Website SKKNI| 321 baris, 5 kolom, 5 jabatan |
| Lowongan Kerja | `cleaned_job_5.csv` | JobStreet Indonesia| 2.516 baris, 18 kolom |

### Kolom Dataset SKKNI (`5_data_pekerjaan.csv`)

| Kolom | Keterangan |
|---|---|
| `Jabatan` | Nama jabatan pekerjaan |
| `Kode Unit` | Kode unit kompetensi SKKNI |
| `Judul Unit` | Judul unit kompetensi |
| `Elemen Kompetensi` | Elemen kompetensi dalam unit |
| `KUK` | Kriteria Unjuk Kerja |

### Kolom Dataset Lowongan Kerja (`cleaned_job_5.csv`)

| Kolom | Keterangan |
|---|---|
| `title` | Judul/posisi pekerjaan |
| `company` | Nama perusahaan |
| `location` | Lokasi pekerjaan |
| `type_of_work` | Tipe pekerjaan (full-time, freelance, dll) |
| `requirement` | Skill dan kualifikasi yang dibutuhkan |
| `description` | Deskripsi pekerjaan |
| `min_work_experience` | Minimum pengalaman kerja (tahun) |
| `max_work_experience` | Maksimum pengalaman kerja (tahun) |
| `min_salary` / `max_salary` | Rentang gaji |
| `currency` | Mata uang (IDR, SGD, dll) |
| `job_level` | Level jabatan |
| `link` | URL lowongan |

### 5 Jabatan yang Dianalisis

1. Cyber Security
2. Teknisi Akuntansi
3. Programmer
4. Data Analyst
5. Insinyur Elektro

---

## Instalasi

### Prasyarat

- Python 
- pip

### Langkah Instalasi

1. Install dependensi:
```bash
pip install -r requirements.txt
```

2. Pastikan kedua file dataset berada di folder yang sama dengan `dashboard.py`:
```
skillbridge-dashboard/
├── dashboard.py
├── 5_data_pekerjaan.csv
├── cleaned_job_5.csv
├── requirements.txt
└── README.md
```

3. Jalankan dashboard:
```bash
streamlit run dashboard.py
```

---

## Cara Penggunaan

1. **Filter Jabatan** — Pilih jabatan spesifik atau tampilkan seluruh jabatan melalui sidebar.

2. **Jumlah Data Ditampilkan** — Atur jumlah data teratas (1–5) yang ditampilkan pada visualisasi dashboard melalui slider di sidebar.

3. **Navigasi Halaman** — Pilih halaman analisis melalui menu sidebar:
   - *Overview Jabatan* → ringkasan distribusi kompetensi tiap jabatan
   - *Unit Kompetensi SKKNI* → analisis kompleksitas jabatan berdasarkan unit kompetensi
   - *Skill Demand Industri* → analisis skill yang paling banyak dibutuhkan industri
   - *Gap Analysis* → analisis relevansi kompetensi SKKNI terhadap kebutuhan industri

---

## Deployment ke Streamlit Cloud

1. Push semua file ke repositori GitHub (termasuk kedua file CSV).
2. Buka [share.streamlit.io](https://share.streamlit.io) dan login dengan akun GitHub.
3. Klik **New app**, pilih repositori dan branch yang sesuai.
4. Isi **Main file path** dengan `dashboard.py`.
5. Klik **Deploy**.

---

## Struktur Proyek

```
skillbridge-dashboard/
├── dashboard.py           # File utama Streamlit
├── 5_data_pekerjaan.csv   # Dataset SKKNI (5 jabatan teratas)
├── cleaned_job_5.csv      # Dataset lowongan kerja Indonesia
├── requirements.txt       # Dependensi Python
└── README.md              # Dokumentasi proyek
```

---

## Metodologi Gap Analysis

Pencocokan skill SKKNI dengan kebutuhan industri dilakukan menggunakan **overlap kata kunci**:

1. Skill diekstrak dari kolom `requirement` dataset lowongan kerja.
2. Setiap judul unit kompetensi SKKNI dipecah menjadi kata-kata kunci (stop words dihapus).
3. Dihitung jumlah skill industri yang memiliki minimal 1 kata kunci yang sama dengan unit kompetensi SKKNI.
4. Unit kompetensi dengan match score > 0 dikategorikan **Relevan**, sisanya dikategorikan **Gap**.

---

##Link Hasil Deployment ke Streamlit Cloud

## Tim

Proyek ini dikembangkan sebagai bagian dari tugas Data Science - SkillBridge AI.
