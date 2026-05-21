# A/B Testing — SkillBridge AI
## Keyword Matching vs Semantic Matching

---

## Deskripsi

Eksperimen ini membandingkan dua metode matching CV ke lowongan kerja yang digunakan dalam aplikasi **SkillBridge AI**. Tujuannya adalah membuktikan secara statistik metode mana yang lebih akurat dalam mencocokkan profil pelamar dengan jabatan yang tersedia.

---

## Hipotesis

| | Keterangan |
|---|---|
| **H0** | Semantic Matching **tidak** lebih baik dari Keyword Matching |
| **H1** | Semantic Matching **lebih baik** dari Keyword Matching |
| **Alpha** | 0.05 |

---

## Desain Eksperimen

| | Group A | Group B |
|---|---|---|
| **Nama** | Keyword Matching | Semantic Matching |
| **Metode** | Overlap kata kunci antara CV dan profil jabatan SKKNI | TF-IDF Cosine Similarity (proxy semantic AI matching) |
| **Sampel** | 200 CV simulasi | 200 CV simulasi |
| **Sumber data** | `5_data_pekerjaan.csv` + `cleaned_job_5.csv` | `5_data_pekerjaan.csv` + `cleaned_job_5.csv` |

---

## Hasil

### Metrik Perbandingan

| Metrik | Group A (Keyword) | Group B (Semantic) |
|---|---|---|
| Mean Score | 0.0371 | 0.0656 |
| Std Dev | 0.0364 | 0.0750 |
| Match Rate | 30.0% | **51.0%** |
| Min Score | 0.0000 | 0.0000 |
| Max Score | 0.1702 | **0.4950** |
| **Peningkatan** | — | **+77.0%** |

### Uji Statistik (Mann-Whitney U Test)

| | Nilai |
|---|---|
| U-statistic | 24052.00 |
| p-value | 0.000194 |
| Alpha | 0.05 |
| **Hasil** | **Tolak H0** |
| Effect Size (Cohen's d) | 0.4842 (medium) |

> **p-value = 0.000194 < 0.05** → Semantic Matching secara statistik **lebih baik** dari Keyword Matching.

---

## Visualisasi Hasil

![Hasil A/B Testing](AB_testing.png)

> Grafik menampilkan: (1) distribusi matching score kedua metode, (2) box plot perbandingan, (3) perbandingan metrik utama Match Rate dan Mean Score.

---

## Kesimpulan

Berdasarkan eksperimen A/B Testing dengan 200 sampel CV simulasi dari dataset lowongan kerja Indonesia:

- Semantic Matching menghasilkan Match Rate **51%** dibanding Keyword Matching **30%**
- Peningkatan mean score sebesar **+77%**
- Hasil uji Mann-Whitney U **signifikan secara statistik** (p = 0.000194)
- Effect size Cohen's d = 0.4842 (medium effect)

**Rekomendasi: Gunakan Semantic Matching sebagai metode utama di SkillBridge AI.**

---

## Cara Menjalankan

1. Pastikan semua file berada dalam satu folder:
```
AB_Testing/
├── ab_testing.py
├── 5_data_pekerjaan.csv
└── cleaned_job_5.csv
```

2. Install dependensi:
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

3. Jalankan script:
```bash
python ab_testing.py
```

4. Output yang dihasilkan:
   - Hasil metrik & uji statistik di terminal
   - File `AB_testing.png` tersimpan di folder yang sama

---

## File

| File | Keterangan |
|---|---|
| `ab_testing.py` | Script utama A/B Testing |
| `AB_testing.png` | Visualisasi hasil eksperimen |
| `5_data_pekerjaan.csv` | Dataset SKKNI (321 baris, 5 jabatan) |
| `cleaned_job_5.csv` | Dataset lowongan kerja Indonesia (2.516 baris) |