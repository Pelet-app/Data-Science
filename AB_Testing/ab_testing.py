"""
A/B Testing: Keyword Matching vs Semantic Matching

Membandingkan dua metode matching CV ke job:
- Group A : Keyword Matching  (baseline)
- Group B : Semantic Matching (TF-IDF cosine similarity sebagai proxy AI)

Metrik yang diuji:
- Precision@K  : seberapa relevan hasil rekomendasi top-K
- Match Rate   : persentase CV yang berhasil di-match ke minimal 1 job

Uji statistik: Mann-Whitney U Test (non-parametric, cocok untuk skor 0-1)
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter

# SEED
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted")

# LOAD DATA
print("-" * 60)
print("  A/B TESTING: Keyword Matching vs Semantic Matching")
print("-" * 60)

df_skkni  = pd.read_csv("5_data_pekerjaan.csv") 
df_github = pd.read_csv("cleaned_job_5.csv")

print(f"\nDataset SKKNI  : {df_skkni.shape[0]} baris")
print(f"Dataset GitHub : {df_github.shape[0]} baris")

# PERSIAPAN DATA
profil_jabatan = (
    df_skkni.groupby("Jabatan")["Judul Unit"]
    .apply(lambda x: " ".join(x.dropna().str.lower()))
    .reset_index()
    .rename(columns={"Judul Unit": "profil_skill"})
)

# Bersihkan requirement dari GitHub sebagai "CV simulasi"
def bersihkan_teks(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df_github["req_clean"] = df_github["requirement"].apply(bersihkan_teks)

# Ambil sample 200 CV simulasi (requirement dari lowongan)
sample_cv = df_github[df_github["req_clean"].str.len() > 20]["req_clean"].sample(200).reset_index(drop=True)

print(f"\nSample CV simulasi : {len(sample_cv)} data")
print(f"Jabatan target     : {profil_jabatan['Jabatan'].tolist()}")

# KEYWORD MATCHING 
def keyword_matching_score(cv_text, job_profile):
    """
    Hitung skor matching berbasis overlap kata kunci.
    Skor = jumlah kata yang cocok / total kata unik di profil job
    """
    stop_words = {"dan", "atau", "yang", "untuk", "dengan", "dalam",
                  "pada", "ke", "di", "dari", "secara", "sesuai", "the",
                  "and", "or", "to", "in", "of", "a", "an", "is", "for"}
    cv_words  = set(cv_text.lower().split()) - stop_words
    job_words = set(job_profile.lower().split()) - stop_words
    if not job_words:
        return 0.0
    overlap = cv_words & job_words
    return len(overlap) / len(job_words)

print("\n[Group A] Menghitung Keyword Matching scores")
scores_A = []
for cv in sample_cv:
    job_scores = [
        keyword_matching_score(cv, row["profil_skill"])
        for _, row in profil_jabatan.iterrows()
    ]
    scores_A.append(max(job_scores))

scores_A = np.array(scores_A)

# SEMANTIC MATCHING (TF-IDF Cosine Similarity)
print("[Group B] Menghitung Semantic Matching scores (TF-IDF)")

corpus = list(profil_jabatan["profil_skill"]) + list(sample_cv)
vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),
    stop_words="english"
)
tfidf_matrix = vectorizer.fit_transform(corpus)

n_jobs = len(profil_jabatan)
job_vectors = tfidf_matrix[:n_jobs]
cv_vectors  = tfidf_matrix[n_jobs:]

similarity_matrix = cosine_similarity(cv_vectors, job_vectors)
scores_B = similarity_matrix.max(axis=1)  # skor tertinggi per CV

# METRIK
THRESHOLD = 0.05

match_rate_A = (scores_A >= THRESHOLD).mean() * 100
match_rate_B = (scores_B >= THRESHOLD).mean() * 100

mean_A = scores_A.mean()
mean_B = scores_B.mean()
std_A  = scores_A.std()
std_B  = scores_B.std()

print("\n" + "-" * 60)
print("  HASIL METRIK")
print("-" * 60)
print(f"\n{'Metrik':<30} {'Group A (Keyword)':>18} {'Group B (Semantic)':>18}")
print("-" * 69)
print(f"{'Mean Score':<30} {mean_A:>18.4f} {mean_B:>18.4f}")
print(f"{'Std Dev':<30} {std_A:>18.4f} {std_B:>18.4f}")
print(f"{'Match Rate (>= threshold)':<30} {match_rate_A:>17.1f}% {match_rate_B:>17.1f}%")
print(f"{'Min Score':<30} {scores_A.min():>18.4f} {scores_B.min():>18.4f}")
print(f"{'Max Score':<30} {scores_A.max():>18.4f} {scores_B.max():>18.4f}")

# UJI STATISTIK
print("\n" + "-" * 60)
print("  UJI STATISTIK")
print("-" * 60)

stat, p_value = stats.mannwhitneyu(scores_B, scores_A, alternative="greater")

alpha = 0.05
print(f"\nH0 : Semantic Matching TIDAK lebih baik dari Keyword Matching")
print(f"H1 : Semantic Matching LEBIH BAIK dari Keyword Matching")
print(f"\nU-statistic : {stat:.2f}")
print(f"p-value     : {p_value:.6f}")
print(f"Alpha       : {alpha}")

if p_value < alpha:
    print(f"\n[OK] KESIMPULAN: Tolak H0 (p={p_value:.4f} < α={alpha})")
    print("   Semantic Matching secara statistik LEBIH BAIK dari Keyword Matching.")
    print("   Metode B (Semantic) direkomendasikan untuk SkillBridge AI.")
else:
    print(f"\n[--] KESIMPULAN: Gagal tolak H0 (p={p_value:.4f} >= α={alpha})")
    print("   Tidak cukup bukti bahwa Semantic Matching lebih baik.")

# EFFECT SIZE
pooled_std = np.sqrt((std_A**2 + std_B**2) / 2)
cohens_d   = (mean_B - mean_A) / pooled_std if pooled_std > 0 else 0

effect_label = (
    "kecil (small)"   if abs(cohens_d) < 0.5 else
    "sedang (medium)" if abs(cohens_d) < 0.8 else
    "besar (large)"
)
print(f"\nEffect Size : {cohens_d:.4f} → {effect_label}")

# VISUALISASI
print("\nMembuat visualisasi : ")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("A/B Testing: Keyword Matching vs Semantic Matching\nSkillBridge AI — Hasil Analisis",
             fontsize=14, fontweight="bold", y=1.02)

# Distribusi skor
ax1 = axes[0]
ax1.hist(scores_A, bins=30, alpha=0.6, color="#3498db", label="Group A: Keyword")
ax1.hist(scores_B, bins=30, alpha=0.6, color="#e74c3c", label="Group B: Semantic")
ax1.axvline(mean_A, color="#3498db", linestyle="--", linewidth=2, label=f"Mean A: {mean_A:.3f}")
ax1.axvline(mean_B, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean B: {mean_B:.3f}")
ax1.set_title("Distribusi Matching Score", fontweight="bold")
ax1.set_xlabel("Score")
ax1.set_ylabel("Frekuensi")
ax1.legend(fontsize=9)

# Box plot perbandingan
ax2 = axes[1]
data_box = pd.DataFrame({
    "Score": np.concatenate([scores_A, scores_B]),
    "Metode": ["Keyword Matching"] * len(scores_A) + ["Semantic Matching"] * len(scores_B)
})
sns.boxplot(data=data_box, x="Metode", y="Score", hue="Metode", ax=ax2,
            palette={"Keyword Matching": "#3498db", "Semantic Matching": "#e74c3c"},
            legend=False)
ax2.set_title("Perbandingan Distribusi Score", fontweight="bold")
ax2.set_xlabel("")
ax2.set_ylabel("Matching Score")

# Match Rate & Mean Score comparison
ax3 = axes[2]
metrik_names = ["Mean Score", "Match Rate (%)"]
vals_A = [mean_A * 100, match_rate_A]
vals_B = [mean_B * 100, match_rate_B]
x = np.arange(len(metrik_names))
width = 0.35
bars_a = ax3.bar(x - width/2, vals_A, width, label="Group A: Keyword",
                  color="#3498db", alpha=0.85)
bars_b = ax3.bar(x + width/2, vals_B, width, label="Group B: Semantic",
                  color="#e74c3c", alpha=0.85)
ax3.bar_label(bars_a, fmt="%.1f", padding=3, fontsize=10)
ax3.bar_label(bars_b, fmt="%.1f", padding=3, fontsize=10)
ax3.set_title("Perbandingan Metrik Utama", fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(metrik_names)
ax3.set_ylabel("Nilai")
ax3.legend()

# Tambahkan anotasi p-value
significance = "Signifikan (p < 0.05)" if p_value < alpha else "Tidak Signifikan (p >= 0.05)"
fig.text(0.5, -0.04,
         f"Uji Statistik: Mann-Whitney U | p-value = {p_value:.4f} | {significance} | "
         f"Effect Size (Cohen's d) = {cohens_d:.3f} ({effect_label})",
         ha="center", fontsize=10, style="italic",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plt.savefig("AB_testing.png", dpi=150, bbox_inches="tight")
print("Visualisasi disimpan: AB_testing.png")

# RINGKASAN EKSEKUTIF
print("\n" + "-" * 60)
print("  RINGKASAN EKSEKUTIF")
print("-" * 60)
print(f"""
Eksperimen  : A/B Testing Metode Matching CV ke Job
Sampel      : {len(sample_cv)} CV simulasi dari dataset lowongan kerja Indonesia
Jabatan     : {', '.join(profil_jabatan['Jabatan'].tolist())}

Group A (Keyword Matching)
  - Mean Score  : {mean_A:.4f}
  - Match Rate  : {match_rate_A:.1f}%
  - Cara kerja  : Overlap kata kunci antara CV dan profil jabatan SKKNI

Group B (Semantic Matching)
  - Mean Score  : {mean_B:.4f}
  - Match Rate  : {match_rate_B:.1f}%
  - Cara kerja  : TF-IDF cosine similarity (proxy semantic AI matching)

Peningkatan   : +{((mean_B - mean_A) / mean_A * 100):.1f}% pada mean score
p-value       : {p_value:.6f} ({'Signifikan' if p_value < alpha else 'Tidak Signifikan'})
Effect Size   : {cohens_d:.4f} ({effect_label})

Rekomendasi   : {'Gunakan Semantic Matching untuk SkillBridge AI' if p_value < alpha else 'Perlu investigasi lebih lanjut'}
""")
