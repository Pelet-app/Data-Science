# SkillBridge AI: Skill ## 📖 Kamus Data (Data Dictionary)


---Gap Analysis System

SkillBridge AI adalah sebuah dashboard analisis data interaktif yang dirancang untuk menjembatani kesenjangan kompetensi antara tenaga kerja (berdasarkan data resume) dengan standar industri nasional (berdasarkan data SKKNI).

---

## Permasalahan & Pertanyaan Bisnis
Proyek ini mengidentifikasi adanya ketidaksesuaian antara keterampilan yang dimiliki pelamar kerja dengan kriteria yang ditetapkan industri.
**Pertanyaan Bisnis:**
1. Apa saja skill yang paling banyak dimiliki oleh pelamar saat ini (Supply)?
2. Apa saja kompetensi utama yang dibutuhkan industri berdasarkan standar SKKNI (Demand)?
3. Sejauh mana tingkat ketersediaan kompetensi spesifik SKKNI di dalam database resume (Undersupply Analysis)?

---

## Metodologi Data Wrangling
Proyek ini menggunakan data mentah yang diproses secara manual tanpa menggunakan dataset siap pakai.

1. **Gathering Data:**
   - **Data SKKNI:** Diperoleh melalui proses scraping dari situs resmi (Sknni) untuk mendapatkan daftar Jabatan, Kode Unit, Judul Unit, Elemen Kompetensi, dan KUK.
   - **Data Resume:** Diperoleh dari dataset publik (Kaggle) yang berisi teks profil pelamar kerja.

2. **Assessing Data:**
   - Ditemukan banyak nilai kosong pada kolom kategori pekerjaan.
   - Format daftar skill dalam dataset resume masih berupa string yang tidak terstruktur.
   - Adanya noise pada teks resume (simbol, angka, dan stopword).

3. **Cleaning Data:**
   - Melakukan normalisasi teks (lowercase, removing punctuation).
   - Mengonversi string list menjadi objek list Python menggunakan library `ast`.
   - Menghapus data duplikat untuk memastikan validitas analisis frekuensi.

---



## Panduan Menjalankan Dashboard
Pastikan Anda sudah menginstal Python di lingkungan lokal Anda.

1. **Instalasi Library:**
   ```bash
   pip install -r requirements.txt


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import ast
import re
from collections import Counter
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="SkillBridge AI Dashboard",
    layout="wide",
)

st_autorefresh(interval=60000, key="dashboard_refresh")
sns.set_theme(style="whitegrid", palette="muted")

@st.cache_data(ttl=60)
def load_data():
    df_skkni = pd.read_csv("5_data_pekerjaan.csv")
    df_github = pd.read_csv("cleaned_job_5.csv")

    # Normalisasi nama kolom (lowercase, strip spasi)
    df_skkni.columns  = df_skkni.columns.str.strip().str.lower().str.replace(" ", "_")
    df_github.columns = df_github.columns.str.strip().str.lower().str.replace(" ", "_")

    frekuensi_skkni = (
        df_skkni
        .groupby(['jabatan', 'judul_unit'])
        .size()
        .reset_index(name='jumlah_unit')
    )
    unit_per_job = (
        df_skkni
        .groupby('jabatan')['judul_unit']
        .count()
        .reset_index(name='total_unit')
        .sort_values('total_unit', ascending=False)
    )

    col_skill = 'recruitment' if 'recruitment' in df_github.columns else 'description'

    def extract_skills(text):
        if pd.isna(text):
            return []
        # Pisahkan berdasarkan koma, titik koma, dan newline
        parts = re.split(r'[,;\n•\-]', str(text))
        skills = [p.strip().title() for p in parts if 2 < len(p.strip()) < 50]
        return skills

    all_skills_github = [
        skill
        for text in df_github[col_skill]
        for skill in extract_skills(text)
    ]
    skill_counter = Counter(all_skills_github)
    df_skill_github = pd.DataFrame(
        skill_counter.most_common(30),
        columns=['Skill', 'Frekuensi']
    )

    skill_skkni_set = set(
        df_skkni['judul_unit'].dropna().str.strip().str.title().tolist()
    )
    # Skill dari GitHub (top 50)
    skill_github_set = set(
        [s for s, _ in skill_counter.most_common(50)]
    )

    # Normalisasi: cek overlap kata kunci
    def skill_overlap_score(skkni_skill, github_skills):
        words = set(skkni_skill.lower().split())
        match = sum(
            1 for gs in github_skills
            if len(words & set(gs.lower().split())) > 0
        )
        return match

    gap_rows = []
    for job in df_skkni['jabatan'].dropna().unique():
        skills_job = df_skkni[df_skkni['jabatan'] == job]['judul_unit'].dropna().str.title().tolist()
        for skill in skills_job:
            score = skill_overlap_score(skill, skill_github_set)
            gap_rows.append({
                'Jabatan': job,
                'Skill SKKNI': skill[:45] + ('...' if len(skill) > 45 else ''),
                'Match Score': score,
                'Status': 'Ada di industri' if score > 0 else 'Tidak ditemukan di industri'
            })

    df_gap = pd.DataFrame(gap_rows)

    return df_skkni, df_github, unit_per_job, frekuensi_skkni, df_skill_github, df_gap

# ── LOAD ─────────────────────────────────────────────────────────────────────
try:
    df_skkni, df_github, unit_per_job, frekuensi_skkni, df_skill_github, df_gap = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

# ── HEADER ───────────────────────────────────────────────────────────────────
st.title("🎯 SkillBridge AI Dashboard")
st.markdown("**Analisis Kesenjangan Kompetensi: Standar SKKNI vs Pasar Kerja Nyata Indonesia**")
st.markdown("---")

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Kontrol Dashboard")
st.sidebar.success(f"Refresh: {datetime.now().strftime('%H:%M:%S')}")

list_jabatan = sorted(df_skkni['jabatan'].dropna().unique().tolist())
selected_job = st.sidebar.selectbox(
    "Filter Jabatan",
    ["Semua Jabatan"] + list_jabatan
)

top_n = st.sidebar.slider("Jumlah Top Skill", min_value=5, max_value=20, value=10)

menu = st.sidebar.radio(
    "Pilih Halaman Analisis",
    ["Unit Kompetensi SKKNI", "Skill Demand Industri", "Gap Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: SKKNI + JobStreet Indonesia")

# ── METRIC CARDS ─────────────────────────────────────────────────────────────
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Total Jabatan SKKNI", df_skkni['jabatan'].nunique())
with col_m2:
    st.metric("Total Unit Kompetensi", len(df_skkni))
with col_m3:
    st.metric("Total Lowongan", len(df_github))
with col_m4:
    total_match = (df_gap['Status'] == 'Ada di industri').sum()
    total_all   = len(df_gap)
    pct = round(total_match / total_all * 100, 1) if total_all > 0 else 0
    st.metric("Skill SKKNI Relevan", f"{pct}%")

st.markdown("---")

# ── FILTER BERDASARKAN SIDEBAR ────────────────────────────────────────────────
if selected_job != "Semua Jabatan":
    df_gap_filtered   = df_gap[df_gap['Jabatan'] == selected_job]
    unit_filtered     = unit_per_job[unit_per_job['jabatan'] == selected_job]
    freq_filtered     = frekuensi_skkni[frekuensi_skkni['jabatan'] == selected_job]
else:
    df_gap_filtered   = df_gap.copy()
    unit_filtered     = unit_per_job.copy()
    freq_filtered     = frekuensi_skkni.copy()

# ════════════════════════════════════════════════════════════════════════════════
# Q2 · JUMLAH UNIT KOMPETENSI PER JOB (SKKNI)
# ════════════════════════════════════════════════════════════════════════════════
if menu == "Q2 · Unit Kompetensi SKKNI":
    st.subheader("Q2 · Berapa jumlah unit kompetensi masing-masing job di SKKNI?")
    st.markdown(
        "Visualisasi ini menunjukkan kompleksitas masing-masing jabatan "
        "berdasarkan jumlah unit kompetensi yang ditetapkan SKKNI. "
        "Semakin banyak unit kompetensi, semakin kompleks jabatan tersebut."
    )

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### Jumlah unit kompetensi per jabatan")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        colors = sns.color_palette("Blues_d", len(unit_per_job))
        bars = ax1.barh(
            unit_per_job['jabatan'],
            unit_per_job['total_unit'],
            color=colors
        )
        ax1.bar_label(bars, fmt='%d', padding=4, fontsize=11)
        ax1.set_xlabel("Jumlah Unit Kompetensi")
        ax1.set_title("Kompleksitas Jabatan SKKNI", fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig1)

    with col2:
        st.markdown("#### Ringkasan per jabatan")
        st.dataframe(
            unit_per_job.rename(columns={
                'jabatan': 'Jabatan',
                'total_unit': 'Total Unit Kompetensi'
            }),
            use_container_width=True,
            hide_index=True
        )
        top_job = unit_per_job.iloc[0]
        st.info(
            f"**{top_job['jabatan']}** memiliki unit kompetensi terbanyak "
            f"({int(top_job['total_unit'])} unit), menandakan jabatan "
            f"dengan standar kompetensi paling kompleks."
        )

    # Detail unit kompetensi jika filter jabatan dipilih
    if selected_job != "Semua Jabatan":
        st.markdown(f"#### Detail unit kompetensi: {selected_job}")
        detail = df_skkni[df_skkni['jabatan'] == selected_job][['judul_unit']].copy()
        detail.columns = ['Unit Kompetensi']
        detail = detail.reset_index(drop=True)
        detail.index += 1
        st.dataframe(detail, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# Q4 · SKILL PALING SERING DIMINTA INDUSTRI (GITHUB)
# ════════════════════════════════════════════════════════════════════════════════
elif menu == "Skill Demand Industri":
    st.subheader("Skill apa yang paling sering diminta perusahaan di Indonesia?")
    st.markdown(
        "Data diambil dari dataset lowongan kerja nyata di Indonesia (GitHub). "
        "Skill diekstrak dari kolom requirement dan deskripsi pekerjaan."
    )

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown(f"#### Top {top_n} skill paling dicari industri")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        data_plot = df_skill_github.head(top_n)
        palette   = sns.color_palette("viridis", top_n)
        bars2 = ax2.barh(
            data_plot['Skill'],
            data_plot['Frekuensi'],
            color=palette
        )
        ax2.bar_label(bars2, fmt='%d', padding=4, fontsize=10)
        ax2.set_xlabel("Frekuensi Kemunculan")
        ax2.set_title(f"Top {top_n} Skill Demand — Pasar Kerja Indonesia", fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig2)

    with col2:
        st.markdown("#### Distribusi tipe pekerjaan")
        # Kolom type_of_work — sesuaikan jika nama kolom berbeda
        col_type = 'type_of_work' if 'type_of_work' in df_github.columns else None
        if col_type:
            type_counts = df_github[col_type].dropna().value_counts().head(6)
            fig3, ax3 = plt.subplots(figsize=(5, 5))
            ax3.pie(
                type_counts.values,
                labels=type_counts.index,
                autopct='%1.1f%%',
                startangle=140,
                colors=sns.color_palette("pastel")
            )
            ax3.set_title("Tipe Pekerjaan", fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig3)
        else:
            st.info("Kolom tipe pekerjaan tidak ditemukan di dataset.")

        top_skill = df_skill_github.iloc[0]['Skill']
        st.success(
            f"Skill **{top_skill}** adalah yang paling banyak diminta "
            f"oleh industri di Indonesia berdasarkan dataset acuan."
        )

# ════════════════════════════════════════════════════════════════════════════════
# Q8 · GAP ANALYSIS: SKKNI VS GITHUB
# ════════════════════════════════════════════════════════════════════════════════
else:
    st.subheader("Gap Analysis: Skill SKKNI vs Kebutuhan Industri Nyata")
    st.markdown(
        "Analisis ini membandingkan unit kompetensi standar SKKNI dengan skill "
        "yang benar-benar diminta di pasar kerja Indonesia. "
        "Skill dengan match score = 0 mengindikasikan potensi kesenjangan kompetensi."
    )

    # ── Summary gap per jabatan ──
    summary_gap = (
        df_gap_filtered
        .groupby('Jabatan')
        .apply(lambda x: pd.Series({
            'Total Skill SKKNI': len(x),
            'Ada di industri':   (x['Status'] == 'Ada di industri').sum(),
            'Tidak ditemukan':   (x['Status'] == 'Tidak ditemukan di industri').sum(),
        }))
        .reset_index()
    )
    summary_gap['% Relevan'] = (
        summary_gap['Ada di industri'] / summary_gap['Total Skill SKKNI'] * 100
    ).round(1)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("#### Persentase skill SKKNI yang relevan dengan industri")
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        palette_gap = ['#2ecc71' if v >= 50 else '#e74c3c'
                       for v in summary_gap['% Relevan']]
        bars4 = ax4.barh(
            summary_gap['Jabatan'],
            summary_gap['% Relevan'],
            color=palette_gap
        )
        ax4.bar_label(bars4, fmt='%.1f%%', padding=4, fontsize=11)
        ax4.axvline(x=50, color='gray', linestyle='--', linewidth=1, label='50% threshold')
        ax4.set_xlabel("% Skill Relevan dengan Industri")
        ax4.set_title("Relevansi Kompetensi SKKNI vs Pasar Kerja", fontsize=13, fontweight='bold')
        ax4.set_xlim(0, 110)
        ax4.invert_yaxis()
        hijau = mpatches.Patch(color='#2ecc71', label='≥ 50% relevan')
        merah = mpatches.Patch(color='#e74c3c', label='< 50% relevan')
        ax4.legend(handles=[hijau, merah], loc='lower right')
        plt.tight_layout()
        st.pyplot(fig4)

    with col2:
        st.markdown("#### Ringkasan gap per jabatan")
        st.dataframe(
            summary_gap[['Jabatan', 'Total Skill SKKNI', 'Ada di industri',
                          'Tidak ditemukan', '% Relevan']],
            use_container_width=True,
            hide_index=True
        )

        best  = summary_gap.loc[summary_gap['% Relevan'].idxmax()]
        worst = summary_gap.loc[summary_gap['% Relevan'].idxmin()]
        st.success(f"**{best['Jabatan']}** paling relevan ({best['% Relevan']}% skill match).")
        st.warning(f"**{worst['Jabatan']}** perlu perhatian ({worst['% Relevan']}% skill match).")

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "SkillBridge AI Dashboard • Analisis Skill Gap: SKKNI vs Pasar Kerja Indonesia • "
    f"Data terakhir dimuat: {datetime.now().strftime('%d %B %Y, %H:%M')}"
)