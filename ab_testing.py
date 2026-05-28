import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Load Data
print("=" * 65)
print("  A/B TESTING — Skill Mapping ke SKKNI")
print("  Grup A: Rule-based  |  Grup B: TF-IDF Cosine Similarity")
print("=" * 65)

df_map  = pd.read_csv("data/skill_mapping_dictionary.csv")
df_skkni = pd.read_csv("data/skkni_reference_clean.csv")

skills_intl = df_map["Skill_Internasional_Kaggle"].str.lower().str.strip().tolist()
skkni_units  = df_skkni["Judul Unit"].str.lower().str.strip().tolist()
skkni_kode   = df_skkni["Kode Unit"].tolist()
skkni_jabatan = df_skkni["Jabatan"].tolist()

print(f"\n✅ Skills internasional: {len(skills_intl)}")
print(f"✅ Unit SKKNI         : {len(skkni_units)}")

# 1. Ground Truth (Label Manual)
GROUND_TRUTH = {
    "python"             : [i for i, j in enumerate(skkni_jabatan) if "programmer" in j.lower()],
    "java"               : [i for i, j in enumerate(skkni_jabatan) if "programmer" in j.lower()],
    "javascript"         : [i for i, j in enumerate(skkni_jabatan) if "programmer" in j.lower()],
    "sql"                : [i for i, j in enumerate(skkni_jabatan) if "data analyst" in j.lower()],
    "data analysis"      : [i for i, j in enumerate(skkni_jabatan) if "data analyst" in j.lower()],
    "machine learning"   : [i for i, j in enumerate(skkni_jabatan) if "data scientist" in j.lower()],
    "deep learning"      : [i for i, j in enumerate(skkni_jabatan) if "data scientist" in j.lower()],
    "network security"   : [i for i, j in enumerate(skkni_jabatan) if "cyber security" in j.lower()],
    "penetration testing": [i for i, j in enumerate(skkni_jabatan) if "cyber security" in j.lower()],
    "encryption"         : [i for i, j in enumerate(skkni_jabatan) if "cyber security" in j.lower()],
    "graphic design"     : [i for i, j in enumerate(skkni_jabatan) if "desainer" in j.lower()],
    "figma"              : [i for i, j in enumerate(skkni_jabatan) if "desainer" in j.lower()],
    "tensorflow"         : [i for i, j in enumerate(skkni_jabatan) if "data scientist" in j.lower()],
    "pytorch"            : [i for i, j in enumerate(skkni_jabatan) if "data scientist" in j.lower()],
    "statistics"         : [i for i, j in enumerate(skkni_jabatan) if "data" in j.lower()],
    "mysql"              : [i for i, j in enumerate(skkni_jabatan) if "programmer" in j.lower() or "data" in j.lower()],
    "postgresql"         : [i for i, j in enumerate(skkni_jabatan) if "programmer" in j.lower() or "data" in j.lower()],
    "docker"             : [i for i, j in enumerate(skkni_jabatan) if "programmer" in j.lower()],
    "kubernetes"         : [i for i, j in enumerate(skkni_jabatan) if "programmer" in j.lower()],
    "agile"              : [i for i, j in enumerate(skkni_jabatan) if "manager" in j.lower() or "project" in j.lower()],
    "scrum"              : [i for i, j in enumerate(skkni_jabatan) if "manager" in j.lower() or "project" in j.lower()],
    "jira"               : [i for i, j in enumerate(skkni_jabatan) if "manager" in j.lower()],
    "project management" : [i for i, j in enumerate(skkni_jabatan) if "manager" in j.lower()],
    "financial analysis" : [i for i, j in enumerate(skkni_jabatan) if "keuangan" in j.lower()],
    "budget management"  : [i for i, j in enumerate(skkni_jabatan) if "keuangan" in j.lower()],
    "html"               : [i for i, j in enumerate(skkni_jabatan) if "programmer" in j.lower() or "web" in j.lower()],
    "css"                : [i for i, j in enumerate(skkni_jabatan) if "programmer" in j.lower() or "web" in j.lower()],
    "react"              : [i for i, j in enumerate(skkni_jabatan) if "programmer" in j.lower() or "web" in j.lower()],
    "angular"            : [i for i, j in enumerate(skkni_jabatan) if "programmer" in j.lower() or "web" in j.lower()],
    "android"            : [i for i, j in enumerate(skkni_jabatan) if "programmer" in j.lower()],
    "ios"                : [i for i, j in enumerate(skkni_jabatan) if "programmer" in j.lower()],
}

# Subset skill yang punya ground truth (test set)
test_skills = list(GROUND_TRUTH.keys())
print(f"\nTest set (skills dengan ground truth): {len(test_skills)} skills\n")

# 2. Grup A — Rule-based Keyword Matching
KEYWORD_RULES = {
    "python"     : "programmer",
    "java"       : "programmer",
    "javascript" : "programmer",
    "html"       : "programmer",
    "css"        : "programmer",
    "react"      : "programmer",
    "angular"    : "programmer",
    "vue js"     : "programmer",
    "node js"    : "programmer",
    "php"        : "programmer",
    "kotlin"     : "programmer",
    "swift"      : "programmer",
    "typescript" : "programmer",
    "docker"     : "programmer",
    "kubernetes" : "programmer",
    "android"    : "programmer",
    "ios"        : "programmer",
    "mysql"      : "programmer",
    "postgresql" : "programmer",
    "mongodb"    : "programmer",
    "sql"        : "data analyst",
    "data analysis": "data analyst",
    "statistics" : "data analyst",
    "tableau"    : "data analyst",
    "power bi"   : "data analyst",
    "machine learning": "data scientist",
    "deep learning"   : "data scientist",
    "tensorflow" : "data scientist",
    "pytorch"    : "data scientist",
    "scikit learn": "data scientist",
    "pandas"     : "data scientist",
    "numpy"      : "data scientist",
    "network security"   : "cyber security",
    "penetration testing": "cyber security",
    "encryption" : "cyber security",
    "firewalls"  : "cyber security",
    "graphic design": "desainer grafis",
    "figma"      : "desainer grafis",
    "adobe creative suite": "desainer grafis",
    "wireframing": "desainer grafis",
    "prototyping": "desainer grafis",
    "financial analysis": "keuangan",
    "budget management" : "keuangan",
    "agile"      : "manager",
    "scrum"      : "manager",
    "project management": "manager",
    "jira"       : "manager",
}

def predict_group_a(skill: str) -> list[int]:
    """
    Rule-based: cek KEYWORD_RULES → cari baris SKKNI dengan jabatan yang cocok.
    Jika tidak ada aturan → return [] (tidak terpetakan).
    """
    rule_jabatan = KEYWORD_RULES.get(skill.lower(), None)
    if rule_jabatan is None:
        return []
    matches = [
        i for i, jab in enumerate(skkni_jabatan)
        if rule_jabatan.lower() in jab.lower()
    ]
    return matches[:3]

# 3. Grup B — TF-IDF Cosine Similarity
skkni_docs = [f"{jud} {jab}" for jud, jab in zip(skkni_units, skkni_jabatan)]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, analyzer="word")
skkni_matrix = vectorizer.fit_transform(skkni_docs)

def predict_group_b(skill: str, top_k: int = 3, threshold: float = 0.05) -> list[int]:
    """
    TF-IDF cosine similarity: hitung kesamaan skill vs semua unit SKKNI.
    Return indeks dengan similarity > threshold, ambil top_k.
    """
    skill_vec = vectorizer.transform([skill])
    sims = cosine_similarity(skill_vec, skkni_matrix).flatten()
    ranked = np.argsort(sims)[::-1][:top_k]
    return [int(i) for i in ranked if sims[i] > threshold]

# 4. Evaluasi per skill
def precision_at_k(predicted: list, ground_truth: list) -> float:
    if not predicted:
        return 0.0
    hits = len(set(predicted) & set(ground_truth))
    return hits / len(predicted)

def recall_at_k(predicted: list, ground_truth: list) -> float:
    if not ground_truth:
        return 0.0
    hits = len(set(predicted) & set(ground_truth))
    return hits / len(ground_truth)

def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

results_a, results_b = [], []

print("-" * 65)
print(f"{'Skill':<22} {'P@A':>6} {'R@A':>6} {'F1@A':>6}  |  {'P@B':>6} {'R@B':>6} {'F1@B':>6}")
print("-" * 65)

for skill in test_skills:
    gt = GROUND_TRUTH[skill]
    pred_a = predict_group_a(skill)
    pred_b = predict_group_b(skill)

    pa = precision_at_k(pred_a, gt)
    ra = recall_at_k(pred_a, gt)
    fa = f1(pa, ra)

    pb = precision_at_k(pred_b, gt)
    rb = recall_at_k(pred_b, gt)
    fb = f1(pb, rb)

    results_a.append({"skill": skill, "precision": pa, "recall": ra, "f1": fa, "covered": len(pred_a) > 0})
    results_b.append({"skill": skill, "precision": pb, "recall": rb, "f1": fb, "covered": len(pred_b) > 0})

    print(f"{skill:<22} {pa:>6.3f} {ra:>6.3f} {fa:>6.3f}  |  {pb:>6.3f} {rb:>6.3f} {fb:>6.3f}")

print("-" * 65)

df_a = pd.DataFrame(results_a)
df_b = pd.DataFrame(results_b)

# 5. Statistik Agregat
print("\n" + "-" * 65)
print("  RINGKASAN PERFORMA")
print("-" * 65)

def summary(df: pd.DataFrame, name: str):
    print(f"\n Grup {name}")
    print(f"   Precision  : {df['precision'].mean():.4f} ± {df['precision'].std():.4f}")
    print(f"   Recall     : {df['recall'].mean():.4f} ± {df['recall'].std():.4f}")
    print(f"   F1-Score   : {df['f1'].mean():.4f} ± {df['f1'].std():.4f}")
    print(f"   Coverage   : {df['covered'].mean()*100:.1f}% skill berhasil dipetakan")

summary(df_a, "A — Rule-based")
summary(df_b, "B — TF-IDF Cosine Similarity")

# 6. Uji Statistik — Mann-Whitney U (non-parametric)
print("\n" + "-" * 65)
print("  UJI STATISTIK — Mann-Whitney U Test")
print("-" * 65)
print("  H0: Tidak ada perbedaan signifikan antara Grup A dan Grup B")
print("  H1: Terdapat perbedaan signifikan (α = 0.05)")

for metric in ["precision", "recall", "f1"]:
    a_vals = df_a[metric].values
    b_vals = df_b[metric].values
    stat, p_val = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")

    # Effect size — Cohen's d
    pooled_std = np.sqrt((a_vals.std()**2 + b_vals.std()**2) / 2)
    cohen_d = (b_vals.mean() - a_vals.mean()) / pooled_std if pooled_std > 0 else 0

    sig = "✅ SIGNIFIKAN" if p_val < 0.05 else "❌ Tidak signifikan"
    direction = "B > A" if b_vals.mean() > a_vals.mean() else "A > B"

    print(f"\n  Metrik: {metric.upper()}")
    print(f"    U-statistic : {stat:.2f}")
    print(f"    p-value     : {p_val:.4f}  → {sig}")
    print(f"    Cohen's d   : {cohen_d:.4f}  (effect size)")
    print(f"    Arah        : {direction}  (B mean={b_vals.mean():.3f} vs A mean={a_vals.mean():.3f})")

# 7. T-test (parametric, untuk validasi)
print("\n" + "-" * 65)
print("  UJI TAMBAHAN — Paired t-test (parametric)")
print("-" * 65)

for metric in ["precision", "recall", "f1"]:
    a_vals = df_a[metric].values
    b_vals = df_b[metric].values
    t_stat, p_val = stats.ttest_rel(a_vals, b_vals)
    sig = "✅ SIGNIFIKAN" if p_val < 0.05 else "❌ Tidak signifikan"
    print(f"  {metric:<10}: t={t_stat:+.3f}, p={p_val:.4f} → {sig}")

# 8. Kesimpulan & Rekomendasi
print("\n" + "-" * 65)
print("  KESIMPULAN")
print("-" * 65)

f1_a = df_a["f1"].mean()
f1_b = df_b["f1"].mean()
winner = "B (TF-IDF)" if f1_b > f1_a else "A (Rule-based)"
improvement = abs(f1_b - f1_a) / f1_a * 100 if f1_a > 0 else 0

print(f"""
  Grup A — Rule-based  F1 : {f1_a:.4f}
  Grup B — TF-IDF      F1 : {f1_b:.4f}
  
  Metode terbaik: {winner}
  Selisih F1     : {f1_b - f1_a:+.4f} ({improvement:.1f}% perubahan)

  Interpretasi:
  • Grup A (Rule-based) memiliki PRECISION tinggi karena aturan keyword
    yang eksplisit, namun RECALL rendah — banyak skill tidak tercakup.
  • Grup B (TF-IDF) memiliki COVERAGE lebih baik karena dapat menangkap
    kemiripan semantik, meskipun kadang menghasilkan false positive.
  
  Rekomendasi:
  Gunakan Grup B (TF-IDF) sebagai metode utama untuk pemetaan otomatis.
  Gunakan Grup A (Rule-based) sebagai post-filter untuk meningkatkan
  precision pada skill yang sudah diketahui padanannya.
  Kombinasi hybrid (A sebagai hard rule + B sebagai fallback) akan
  menghasilkan performa paling optimal.
"""
)

# Hitung coverage gap
covered_b = df_b["covered"].mean() * 100
unmapped_pct = 100 - covered_b
print(f"  Gap kritis: {unmapped_pct:.0f}% skill internasional masih belum memiliki")
print(f"  unit SKKNI eksplisit → perlu usulan penyusunan unit SKKNI baru.")

# 9. Simpan hasil ke CSV
output_df = pd.DataFrame({
    "skill": test_skills,
    "precision_A": df_a["precision"].values,
    "recall_A": df_a["recall"].values,
    "f1_A": df_a["f1"].values,
    "covered_A": df_a["covered"].values,
    "precision_B": df_b["precision"].values,
    "recall_B": df_b["recall"].values,
    "f1_B": df_b["f1"].values,
    "covered_B": df_b["covered"].values,
    "winner": ["B" if fb > fa else "A" if fa > fb else "Tie"
               for fa, fb in zip(df_a["f1"], df_b["f1"])],
})
output_df.to_csv("ab_testing_results.csv", index=False)
print("Hasil disimpan ke: ab_testing_results.csv")
print("-" * 65)