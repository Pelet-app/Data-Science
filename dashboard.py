import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
from collections import Counter
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# KONFIGURASI
st.set_page_config(
    page_title="SkillBridge AI Dashboard",
    layout="wide",
)

st_autorefresh(interval=60000, key="dashboard_refresh")
sns.set_theme(style="whitegrid", palette="muted")

# LOAD DATA
@st.cache_data(ttl=60)
def load_data():

    # LOAD CSV
    df_skkni = pd.read_csv("5_data_pekerjaan.csv")
    df_github = pd.read_csv("cleaned_job_5.csv")

    # AMANKAN TIPE DATA NUMERIK
    df_github["min_work_experience"] = pd.to_numeric(
        df_github["min_work_experience"],
        errors="coerce"
    )

    df_github["max_work_experience"] = pd.to_numeric(
        df_github["max_work_experience"],
        errors="coerce"
    )

    # JUMLAH UNIT KOMPETENSI UNIK PER JABATAN
    unit_per_job = (
        df_skkni.groupby("Jabatan")["Judul Unit"]
        .nunique()
        .reset_index(name="Total Unit")
        .sort_values("Total Unit", ascending=False)
    )

    # EXTRACT SKILL
    def extract_skills(text):
        if pd.isna(text):
            return []

        parts = re.split(r"[,;\n]", str(text))

        return [
            p.strip().title()
            for p in parts
            if 2 < len(p.strip()) < 60
        ]

    all_skills = [
        skill
        for text in df_github["requirement"].fillna("")
        for skill in extract_skills(text)
    ]

    skill_counter = Counter(all_skills)

    df_skill_github = pd.DataFrame(
        skill_counter.most_common(30),
        columns=["Skill", "Frekuensi"]
    )

    # GAP ANALYSIS
    skill_github_set = set(
        [s for s, _ in skill_counter.most_common(80)]
    )

    stop_words = {
        "dan", "atau", "yang", "untuk", "dengan",
        "dalam", "pada", "ke", "di", "dari",
        "secara", "sesuai"
    }

    def match_score(skkni_skill, github_skills):

        words = (
            set(str(skkni_skill).lower().split())
            - stop_words
        )

        return sum(
            1
            for gs in github_skills
            if len(words & set(gs.lower().split())) > 0
        )

    gap_rows = []

    for job in df_skkni["Jabatan"].dropna().unique():

        skills = (
            df_skkni[df_skkni["Jabatan"] == job]["Judul Unit"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.title()
            .unique()
        )

        for skill in skills:

            score = match_score(skill, skill_github_set)

            gap_rows.append({
                "Jabatan": job,
                "Skill SKKNI": (
                    skill[:50] + "..."
                    if len(skill) > 50
                    else skill
                ),
                "Match Score": score,
                "Status": (
                    "Relevan"
                    if score > 0
                    else "Gap (tidak ditemukan)"
                )
            })

    df_gap = pd.DataFrame(gap_rows)

    return (
        df_skkni,
        df_github,
        unit_per_job,
        df_skill_github,
        df_gap
    )

# LOAD
try:
    (
        df_skkni,
        df_github,
        unit_per_job,
        df_skill_github,
        df_gap
    ) = load_data()

except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

# HEADER
st.title("SkillBridge AI — Skill Gap Dashboard")

st.markdown(
    "**Analisis Kesenjangan Kompetensi: "
    "Standar SKKNI vs JobStreet Indonesia**"
)

st.caption(
    f"Dataset SKKNI: {len(df_skkni)} baris · "
    f"Dataset JobStreet Indonesia: {len(df_github)} lowongan · "
    f"Refresh: {datetime.now().strftime('%H:%M:%S')}"
)

st.markdown("---")

# SIDEBAR
st.sidebar.header("⚙️ Kontrol Dashboard")

st.sidebar.success(
    f"Last update: {datetime.now().strftime('%H:%M:%S')}"
)

list_jabatan = sorted(
    df_skkni["Jabatan"].dropna().unique().tolist()
)

selected_job = st.sidebar.selectbox(
    "Filter Jabatan",
    ["Semua Jabatan"] + list_jabatan
)

top_n = st.sidebar.slider(
    "Jumlah Top Skill Ditampilkan",
    1,
    5,
    5
)

menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Overview Jabatan",
        "Unit Kompetensi SKKNI",
        "Skill Demand Industri",
        "Gap Analysis"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Sumber data:\n"
    "- SKKNI & JobStreet Indonesia"
)

# METRIC CARDS
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Jabatan SKKNI",
        df_skkni["Jabatan"].nunique()
    )

with c2:
    st.metric(
        "Total Baris SKKNI",
        f"{len(df_skkni):,}"
    )

with c3:
    st.metric(
        "Lowongan Kerja",
        f"{len(df_github):,}"
    )

with c4:

    total_rel = (
        df_gap["Status"] == "Relevan"
    ).sum()

    pct_rel = round(
        total_rel / len(df_gap) * 100,
        1
    )

    st.metric(
        "Skill SKKNI Relevan",
        f"{pct_rel}%"
    )

st.markdown("---")

# OVERVIEW
if menu == "Overview Jabatan":

    st.subheader("Overview Semua Jabatan SKKNI")

    overview = (
        df_skkni.groupby("Jabatan")
        .agg(
            Total_Baris=("Judul Unit", "count"),
            Unit_Unik=("Judul Unit", "nunique"),
            Elemen_Unik=("Elemen Kompetensi", "nunique"),
        )
        .reset_index()
        .sort_values("Total_Baris", ascending=False)
        .head(top_n)
        .rename(columns={
            "Total_Baris": "Total Baris",
            "Unit_Unik": "Unit Kompetensi Unik",
            "Elemen_Unik": "Elemen Kompetensi Unik",
        })
        .reset_index(drop=True)
    )

    overview.index += 1

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.dataframe(
            overview,
            use_container_width=True
        )

    with col2:

        fig, ax = plt.subplots(figsize=(6, 4))

        colors = sns.color_palette(
            "Blues_d",
            len(overview)
        )

        bars = ax.barh(
            overview["Jabatan"],
            overview["Total Baris"],
            color=colors
        )

        ax.bar_label(
            bars,
            fmt="%d",
            padding=4
        )

        ax.set_xlabel("Total Baris")

        ax.set_title(
            "Distribusi Baris per Jabatan"
        )

        ax.invert_yaxis()

        plt.tight_layout()

        st.pyplot(fig)

# FILTER
if selected_job != "Semua Jabatan":
    df_gap_f = df_gap[
        df_gap["Jabatan"] == selected_job
    ].copy()

else:
    df_gap_f = df_gap.copy()

# UNIT KOMPETENSI
if menu == "Unit Kompetensi SKKNI":

    st.subheader(
        "Jumlah unit kompetensi tiap jabatan di SKKNI"
    )

    st.markdown(
        "Jumlah unit kompetensi mencerminkan "
        "**kompleksitas standar** tiap jabatan."
    )

    unit_top = unit_per_job.head(top_n)

    col1, col2 = st.columns([1.4, 1])

    with col1:

        fig, ax = plt.subplots(figsize=(8, 5))

        colors = sns.color_palette(
            "Blues_d",
            len(unit_top)
        )

        bars = ax.barh(
            unit_top["Jabatan"],
            unit_top["Total Unit"],
            color=colors
        )

        ax.bar_label(
            bars,
            fmt="%d unit",
            padding=5
        )

        ax.set_xlabel(
            "Jumlah Unit Kompetensi Unik"
        )

        ax.set_title(
            "Kompleksitas Jabatan Berdasarkan SKKNI"
        )

        ax.invert_yaxis()

        plt.tight_layout()

        st.pyplot(fig)

    with col2:

        st.markdown("#### Tabel ringkasan")

        st.dataframe(
            unit_top.rename(
                columns={
                    "Total Unit": "Unit Kompetensi"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

# SKILL DEMAND
elif menu == "Skill Demand Industri":

    st.subheader(
        "Skill yang paling sering diminta perusahaan"
    )

    col1, col2 = st.columns([1.4, 1])

    with col1:

        data_plot = df_skill_github.head(top_n)

        palette = sns.color_palette(
            "viridis",
            top_n
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        bars = ax.barh(
            data_plot["Skill"],
            data_plot["Frekuensi"],
            color=palette
        )

        ax.bar_label(
            bars,
            fmt="%d",
            padding=4
        )

        ax.set_xlabel("Frekuensi")

        ax.set_title(
            f"Top {top_n} Skill Demand"
        )

        ax.invert_yaxis()

        plt.tight_layout()

        st.pyplot(fig)

    with col2:

        type_counts = (
            df_github["type_of_work"]
            .dropna()
            .value_counts()
            .head(5)
        )

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.pie(
            type_counts.values,
            labels=type_counts.index,
            autopct="%1.1f%%",
            startangle=140
        )

        ax.set_title("Tipe Pekerjaan")

        plt.tight_layout()

        st.pyplot(fig)

        avg_min = round(
            df_github["min_work_experience"].mean(),
            1
        )

        avg_max = round(
            df_github["max_work_experience"].mean(),
            1
        )

        st.metric(
            "Min pengalaman",
            f"{avg_min} tahun"
        )

        st.metric(
            "Max pengalaman",
            f"{avg_max} tahun"
        )

# GAP ANALYSIS
elif menu == "Gap Analysis":

    st.subheader(
        "Gap Analysis: SKKNI vs Industri"
    )

    summary = (
        df_gap_f
        .groupby("Jabatan")
        .apply(lambda x: pd.Series({
            "Total Skill SKKNI": len(x),
            "Relevan": (
                x["Status"] == "Relevan"
            ).sum(),
            "Gap": (
                x["Status"] == "Gap (tidak ditemukan)"
            ).sum(),
        }))
        .reset_index()
    )

    summary["% Relevan"] = (
        summary["Relevan"]
        / summary["Total Skill SKKNI"]
        * 100
    ).round(1)

    summary = (
        summary
        .sort_values("% Relevan", ascending=False)
        .head(top_n)
    )

    col1, col2 = st.columns([1.4, 1])

    with col1:

        fig, ax = plt.subplots(figsize=(8, 5))

        colors = [
            "#2ecc71"
            if v >= 50
            else "#e74c3c"
            for v in summary["% Relevan"]
        ]

        bars = ax.barh(
            summary["Jabatan"],
            summary["% Relevan"],
            color=colors
        )

        ax.bar_label(
            bars,
            fmt="%.1f%%",
            padding=5
        )

        ax.axvline(
            x=50,
            color="gray",
            linestyle="--",
            linewidth=1
        )

        ax.set_xlabel("% Skill Relevan")

        ax.set_title(
            "Relevansi Kompetensi SKKNI"
        )

        ax.invert_yaxis()

        hijau = mpatches.Patch(
            color="#2ecc71",
            label="≥ 50% relevan"
        )

        merah = mpatches.Patch(
            color="#e74c3c",
            label="< 50% relevan"
        )

        ax.legend(
            handles=[hijau, merah],
            loc="lower right"
        )

        plt.tight_layout()

        st.pyplot(fig)

    with col2:

        st.dataframe(
            summary[
                [
                    "Jabatan",
                    "Total Skill SKKNI",
                    "Relevan",
                    "Gap",
                    "% Relevan"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

# FOOTER
st.markdown("---")

st.caption(
    "SkillBridge AI Dashboard · "
    "SKKNI vs JobStreet Indonesia · "
    f"{datetime.now().strftime('%d %B %Y, %H:%M')}"
)
