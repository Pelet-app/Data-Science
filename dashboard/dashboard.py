import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import ast
import re
import warnings
warnings.filterwarnings("ignore")

from data_loader import (
    load_jobs, load_resumes, load_skkni, load_skill_map,
    skill_freq_by_category, top_skills_overall, job_skill_freq,
)

st.set_page_config(
    page_title="PELET (Pencari Lowongan Efektif & Tepat) - Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    [data-testid="stSidebar"] { background: #161b27; }
    .metric-card {
        background: #1e2535;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-number { font-size: 2.2rem; font-weight: 700; color: #60a5fa; }
    .metric-label  { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
    .section-title {
        font-size: 1.15rem; font-weight: 600;
        color: #e2e8f0; margin: 28px 0 12px;
        border-left: 3px solid #3b82f6; padding-left: 10px;
    }
    .insight-box {
        background: #1e2535; border: 1px solid #3b82f6;
        border-radius: 10px; padding: 14px 18px;
        font-size: 0.9rem; color: #cbd5e1; margin-top: 12px;
    }
    .gap-badge-mapped   { background:#166534; color:#bbf7d0; padding:3px 10px; border-radius:20px; font-size:0.78rem; }
    .gap-badge-unmapped { background:#7f1d1d; color:#fecaca; padding:3px 10px; border-radius:20px; font-size:0.78rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logopelet.png", width=100)
    st.markdown("## Pelet - Dashboard")
    st.markdown("**AI Semantic Matching Berbasis SKKNI**")
    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["Overview", "Job Market", "Skill Demand", "ID SKKNI Gap Analysis"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption("cleaned_job_5.csv — 2.516 lowongan")
    st.caption("cleaned_training_data.csv — 10.000 resume")
    st.caption("skkni_reference_clean.csv — 662 unit")
    st.caption("skill_mapping_dictionary.csv — 114 skills")


# ── PERBAIKAN 5: cache hanya di layer Streamlit, tanpa lru_cache di data_loader ──
@st.cache_data
def get_all_data():
    return load_jobs(), load_resumes(), load_skkni(), load_skill_map()

df_job, df_resume, df_skkni, df_map = get_all_data()


@st.cache_data
def build_skkni_status(_df_map, _df_skkni):
    col_skill = "Skill_Internasional_Kaggle"
    col_match = "Kecocokan"
    # ── PERBAIKAN 4: nama kolom disesuaikan dengan nama asli di CSV ──
    col_kode    = "Kode_Unit_SKKNI" if "Kode_Unit_SKKNI" in _df_map.columns else None
    col_jabatan = "Jabatan" if "Jabatan" in _df_map.columns else None

    rows = []
    for _, r in _df_map.iterrows():
        skill     = str(r[col_skill]).strip().lower()
        match_raw = str(r.get(col_match, "")).strip().lower()
        jabatan   = str(r.get(col_jabatan, "-")).strip() if col_jabatan else "-"
        kode      = str(r.get(col_kode, "-")).strip() if col_kode else "-"

        if "cocok" in match_raw and "parsial" not in match_raw:
            status = "Terpetakan"
        elif "parsial" in match_raw or "sebagian" in match_raw:
            status = "Parsial"
        else:
            status = "Belum Dipetakan"

        rows.append({
            "skill":   skill,
            "status":  status,
            "jabatan": jabatan,
            "kode":    kode,
        })

    return pd.DataFrame(rows)


status_df = build_skkni_status(df_map, df_skkni)

mapped_n   = len(status_df[status_df["status"] == "Terpetakan"])
partial_n  = len(status_df[status_df["status"] == "Parsial"])
unmapped_n = len(status_df[status_df["status"] == "Belum Dipetakan"])
total_intl = len(status_df)
coverage_pct = round((mapped_n + partial_n * 0.5) / total_intl * 100, 1) if total_intl > 0 else 0


# ═══════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════
if page == "Overview":
    st.title("PELET (Pencari Lowongan Efektif & Tepat): AI Semantic Matching Berbasis SKKNI")
    st.markdown("Analisis kesenjangan skill tenaga kerja Indonesia terhadap standar kompetensi nasional (SKKNI)")
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        (col1, "2.516",            "Lowongan Kerja"),
        (col2, "10.000",           "Resume Kandidat"),
        (col3, "662",              "Unit Kompetensi SKKNI"),
        (col4, str(total_intl),    "Skill Internasional"),
        (col5, f"{coverage_pct}%", "Coverage ke SKKNI (est.)"),
    ]
    for col, num, label in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{num}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Distribusi Kategori Resume</div>', unsafe_allow_html=True)
        cat_counts = df_resume["Category"].value_counts().head(12).reset_index()
        cat_counts.columns = ["Kategori", "Jumlah"]
        fig = px.bar(
            cat_counts, x="Jumlah", y="Kategori", orientation="h",
            color="Jumlah", color_continuous_scale="Blues",
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, height=380,
            margin=dict(l=0, r=10, t=10, b=10),
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Distribusi Jabatan SKKNI</div>', unsafe_allow_html=True)
        jab_counts = df_skkni["Jabatan"].value_counts().head(12).reset_index()
        jab_counts.columns = ["Jabatan", "Jumlah"]
        fig2 = px.bar(
            jab_counts, x="Jumlah", y="Jabatan", orientation="h",
            color="Jumlah", color_continuous_scale="Teal",
            template="plotly_dark",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, height=380,
            margin=dict(l=0, r=10, t=10, b=10),
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        st.markdown(f"""<div class="insight-box">
        <b>🔴 Gap Kritis</b><br>
        Dari <b>{total_intl} skill internasional</b>, baru <b>{mapped_n} terpetakan penuh</b>
        dan <b>{partial_n} parsial</b> ke unit kompetensi SKKNI.
        Coverage rate saat ini: <b>{coverage_pct}%</b>.
        </div>""", unsafe_allow_html=True)
    with ic2:
        st.markdown("""<div class="insight-box">
        <b>🟡 Dominasi Teknologi</b><br>
        Kategori <b>Technology</b> mendominasi resume (25,1%) jauh di atas kategori lain,
        namun SKKNI hanya punya 37 unit untuk Programmer.
        </div>""", unsafe_allow_html=True)
    with ic3:
        st.markdown("""<div class="insight-box">
        <b>🟢 Peluang Mapping</b><br>
        Skill seperti <b>Python, SQL, Data Analysis</b> memiliki padanan konseptual
        di SKKNI (Data Analyst, Programmer) dan sudah terpetakan.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# JOB MARKET
# ═══════════════════════════════════════════════
elif page == "Job Market":
    st.title("Analisis Job Market Indonesia")
    st.markdown("Insight dari **2.516 lowongan kerja** di platform Glints")
    st.markdown("---")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        kota_opts = ["Semua"] + sorted(df_job["kota"].value_counts().head(15).index.tolist())
        kota_sel = st.selectbox("Filter Kota", kota_opts)
    with col_f2:
        work_opts = ["Semua"] + sorted(df_job["type_of_work"].dropna().unique().tolist())
        work_sel = st.selectbox("Tipe Pekerjaan", work_opts)
    with col_f3:
        curr_opts = ["Semua", "IDR", "SGD"]
        curr_sel = st.selectbox("Mata Uang Gaji", curr_opts)

    dfjf = df_job.copy()
    if kota_sel != "Semua":
        dfjf = dfjf[dfjf["kota"] == kota_sel]
    if work_sel != "Semua":
        dfjf = dfjf[dfjf["type_of_work"] == work_sel]
    if curr_sel != "Semua":
        dfjf = dfjf[dfjf["currency"] == curr_sel]

    st.caption(f"Menampilkan **{len(dfjf):,}** lowongan")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Top 15 Kota berdasarkan Lowongan</div>', unsafe_allow_html=True)
        kota_ct = dfjf["kota"].value_counts().head(15).reset_index()
        kota_ct.columns = ["Kota", "Jumlah"]
        fig = px.bar(
            kota_ct, x="Kota", y="Jumlah",
            color="Jumlah", color_continuous_scale="Blues",
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, height=340,
            margin=dict(l=0, r=10, t=10, b=80),
            xaxis_tickangle=-35,
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Insight"):
            st.markdown("Jakarta mendominasi lowongan digital. Bandung, Surabaya, dan Yogyakarta menjadi pusat pertumbuhan di luar Jabodetabek.")

    with col2:
        st.markdown('<div class="section-title">Tipe Pekerjaan</div>', unsafe_allow_html=True)
        work_ct = dfjf["type_of_work"].value_counts().reset_index()
        work_ct.columns = ["Tipe", "Jumlah"]
        fig2 = px.pie(
            work_ct, names="Tipe", values="Jumlah",
            hole=0.45, template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=340,
            margin=dict(l=0, r=0, t=10, b=10),
            legend=dict(font=dict(size=11)),
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-title">Distribusi Gaji (IDR — juta/bulan)</div>', unsafe_allow_html=True)
        salary_df = dfjf[
            (dfjf["has_salary"]) &
            (dfjf["currency"] == "IDR") &
            (dfjf["salary_mid"] > 0) &
            (dfjf["salary_mid"] < 200)
        ]
        if len(salary_df) > 0:
            fig3 = px.histogram(
                salary_df, x="salary_mid", nbins=25,
                template="plotly_dark",
                color_discrete_sequence=["#3b82f6"],
                labels={"salary_mid": "Gaji Tengah (Juta IDR)"},
            )
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=300, margin=dict(l=0, r=10, t=10, b=10),
                bargap=0.08,
            )
            st.plotly_chart(fig3, use_container_width=True)
            med = salary_df["salary_mid"].median()
            st.caption(f"Median gaji IDR: **{med:.0f} juta/bulan** | n={len(salary_df)} lowongan dengan info gaji")
        else:
            st.info("Tidak ada data gaji IDR untuk filter ini.")

    with col4:
        st.markdown('<div class="section-title">Pengalaman Kerja yang Diminta</div>', unsafe_allow_html=True)
        exp_df = dfjf[dfjf["max_work_experience"] > 0]
        fig4 = px.box(
            exp_df, y="max_work_experience",
            template="plotly_dark",
            color_discrete_sequence=["#60a5fa"],
            labels={"max_work_experience": "Maks. Pengalaman (tahun)"},
        )
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(l=0, r=10, t=10, b=10),
        )
        st.plotly_chart(fig4, use_container_width=True)
        with st.expander("Insight"):
            st.markdown("Mayoritas lowongan meminta 0–3 tahun pengalaman, menunjukkan peluang besar bagi fresh graduate dan junior talent.")

    st.markdown('<div class="section-title">Experience Mismatch: Pelamar vs Industri</div>', unsafe_allow_html=True)
    bins     = [0, 1, 3, 5, 10, 100]
    labels_b = ["Fresh (<1)", "Junior (1–3)", "Mid (3–5)", "Senior (5–10)", "Expert (10+)"]
    df_resume["exp_bucket"]    = pd.cut(df_resume["Experience Years"], bins=bins, labels=labels_b, right=True)
    dfjf["exp_bucket_job"]     = pd.cut(dfjf["exp_mid"], bins=bins, labels=labels_b, right=True)

    supply_exp = df_resume["exp_bucket"].value_counts().reindex(labels_b).fillna(0)
    demand_exp = dfjf["exp_bucket_job"].value_counts().reindex(labels_b).fillna(0)
    supply_pct = (supply_exp / supply_exp.sum() * 100).round(1)
    demand_pct = (demand_exp / demand_exp.sum() * 100).round(1)

    fig_mm = go.Figure()
    fig_mm.add_trace(go.Bar(name="Supply: Pelamar (%)",   x=labels_b, y=supply_pct.values, marker_color="#3b82f6"))
    fig_mm.add_trace(go.Bar(name="Demand: Industri (%)",  x=labels_b, y=demand_pct.values, marker_color="#ef4444"))
    fig_mm.update_layout(
        barmode="group", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=360, margin=dict(l=0, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.05),
        yaxis_title="Persentase (%)",
    )
    st.plotly_chart(fig_mm, use_container_width=True)
    st.markdown("""<div class="insight-box">
    <b>Insight — Experience Mismatch:</b> Pelamar Junior (1–3 tahun) adalah segmen terbesar di sisi supply,
    namun industri juga banyak membutuhkan profil Senior (5–10 tahun). Ini mengindikasikan
    <b>vertical mismatch</b> yang menjadi salah satu alasan utama skill gap di Indonesia.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Top 20 Judul Lowongan</div>', unsafe_allow_html=True)
    title_ct = dfjf["title"].value_counts().head(20).reset_index()
    title_ct.columns = ["Judul", "Jumlah"]
    fig5 = px.bar(
        title_ct, x="Jumlah", y="Judul", orientation="h",
        color="Jumlah", color_continuous_scale="Viridis",
        template="plotly_dark",
    )
    fig5.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False, height=480,
        margin=dict(l=0, r=10, t=10, b=10),
        yaxis=dict(tickfont=dict(size=10)),
    )
    st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════
# SKILL DEMAND
# ═══════════════════════════════════════════════
elif page == "Skill Demand":
    st.title("Analisis Skill Demand")
    st.markdown("Skill apa yang paling banyak dibutuhkan dari **10.000 resume** dan **2.516 lowongan**?")
    st.markdown("---")

    # ── PERBAIKAN 1 & 2: req_skill_counts didefinisikan di luar tab
    #    menggunakan req_skills dari data_loader (regex-based), bukan split whitespace ──
    req_skill_counts = Counter()
    for lst in df_job["req_skills"]:
        req_skill_counts.update(lst)

    tab1, tab2, tab3 = st.tabs(["Dari Resume", "Dari Lowongan", "Perbandingan"])

    with tab1:
        col_f, col_g = st.columns([1, 3])
        with col_f:
            n_skills = st.slider("Jumlah skill ditampilkan", 10, 50, 25)
            cat_opts = ["Semua"] + sorted(df_resume["Category"].unique().tolist())
            cat_sel  = st.selectbox("Filter Kategori", cat_opts)

        dfr_filt  = df_resume if cat_sel == "Semua" else df_resume[df_resume["Category"] == cat_sel]
        all_skills = []
        for lst in dfr_filt["skills_list"]:
            all_skills.extend(lst)
        top_sk = pd.DataFrame(Counter(all_skills).most_common(n_skills), columns=["skill", "count"])

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="section-title">Top Skills dari Resume</div>', unsafe_allow_html=True)
            fig = px.bar(
                top_sk.sort_values("count"), x="count", y="skill",
                orientation="h", color="count",
                color_continuous_scale="Blues",
                template="plotly_dark",
                labels={"count": "Frekuensi", "skill": "Skill"},
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False, height=max(400, n_skills * 18),
                margin=dict(l=0, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">Distribusi Jumlah Skill</div>', unsafe_allow_html=True)
            fig2 = px.histogram(
                dfr_filt, x="skill_count", nbins=20,
                template="plotly_dark",
                color_discrete_sequence=["#3b82f6"],
                labels={"skill_count": "Jumlah skill per resume"},
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=300, margin=dict(l=0, r=10, t=10, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown('<div class="section-title">Pengalaman vs Skill Count</div>', unsafe_allow_html=True)
            exp_skill = dfr_filt.groupby("Experience Years")["skill_count"].mean().reset_index()
            fig3 = px.line(
                exp_skill, x="Experience Years", y="skill_count",
                template="plotly_dark",
                color_discrete_sequence=["#60a5fa"],
                labels={"skill_count": "Rata-rata skill"},
            )
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=260, margin=dict(l=0, r=10, t=10, b=10),
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<div class="section-title">Heatmap: Top 15 Skill per Kategori (Top 8 Kategori)</div>', unsafe_allow_html=True)
        top_cats         = df_resume["Category"].value_counts().head(8).index.tolist()
        top_global_skills = [s for s, _ in Counter(all_skills).most_common(15)]
        heat_data = {}
        for cat in top_cats:
            sub        = df_resume[df_resume["Category"] == cat]
            cat_skills = []
            for lst in sub["skills_list"]:
                cat_skills.extend(lst)
            counts = Counter(cat_skills)
            heat_data[cat] = [counts.get(sk, 0) for sk in top_global_skills]
        heat_df = pd.DataFrame(heat_data, index=top_global_skills)
        fig_heat = px.imshow(
            heat_df, aspect="auto", color_continuous_scale="Blues",
            template="plotly_dark",
            labels=dict(x="Kategori", y="Skill", color="Frekuensi"),
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=420,
            margin=dict(l=0, r=10, t=10, b=80),
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab2:
        # ── PERBAIKAN 1: grafik demand sekarang pakai req_skill_counts
        #    yang sudah bersih dari regex matching, bukan split whitespace ──
        st.markdown('<div class="section-title">Top Skills dari Requirement Lowongan</div>', unsafe_allow_html=True)
        top_req = pd.DataFrame(req_skill_counts.most_common(30), columns=["skill", "count"])

        col_j1, col_j2 = st.columns(2)
        with col_j1:
            fig_jsk = px.bar(
                top_req.sort_values("count"), x="count", y="skill",
                orientation="h", color="count",
                color_continuous_scale="Teal",
                template="plotly_dark",
                labels={"count": "Frekuensi", "skill": "Skill"},
            )
            fig_jsk.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False, height=500,
                margin=dict(l=0, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_jsk, use_container_width=True)

        with col_j2:
            st.markdown('<div class="section-title">Requirement Skill per Kota (Top 8)</div>', unsafe_allow_html=True)
            top_kota       = df_job["kota"].value_counts().head(8).index.tolist()
            kota_skill_data = []
            for kota in top_kota:
                sub       = df_job[df_job["kota"] == kota]
                avg_skill = sub["req_skill_count"].mean()
                kota_skill_data.append({
                    "Kota": kota,
                    "Avg skill count": round(avg_skill, 1),
                    "Jumlah lowongan": len(sub),
                })
            ksd = pd.DataFrame(kota_skill_data)
            fig_ksk = px.scatter(
                ksd, x="Jumlah lowongan", y="Avg skill count",
                size="Jumlah lowongan", text="Kota",
                template="plotly_dark",
                color="Avg skill count",
                color_continuous_scale="Blues",
            )
            fig_ksk.update_traces(textposition="top center")
            fig_ksk.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=400, margin=dict(l=0, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_ksk, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-title">Perbandingan Skill: Resume vs Lowongan Kerja</div>', unsafe_allow_html=True)
        st.markdown("Frekuensi dinormalisasi per **1.000 entri** agar perbandingan adil antara dataset resume (10.000) dan lowongan (2.516).")

        resume_counter = Counter()
        for lst in df_resume["skills_list"]:
            resume_counter.update(lst)

        common_skills = [s for s, _ in resume_counter.most_common(15)]

        comp_rows = []
        for sk in common_skills:
            r_norm = resume_counter.get(sk, 0) / len(df_resume) * 1000
            # ── PERBAIKAN 2: lookup langsung pakai nama skill, tidak perlu workaround
            #    karena req_skill_counts sudah pakai nama skill resmi ──
            j_norm = req_skill_counts.get(sk, 0) / len(df_job) * 1000
            comp_rows.append({
                "skill": sk,
                "Resume (per 1k)":  round(r_norm, 1),
                "Lowongan (per 1k)": round(j_norm, 1),
            })

        comp_df = pd.DataFrame(comp_rows)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Resume (kandidat) — per 1k",
            y=comp_df["skill"], x=comp_df["Resume (per 1k)"],
            orientation="h", marker_color="#3b82f6",
        ))
        fig_comp.add_trace(go.Bar(
            name="Lowongan (perusahaan) — per 1k",
            y=comp_df["skill"], x=comp_df["Lowongan (per 1k)"],
            orientation="h", marker_color="#10b981",
        ))
        fig_comp.update_layout(
            barmode="group", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=500, margin=dict(l=0, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.05),
            xaxis_title="Frekuensi per 1.000 entri",
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("""<div class="insight-box">
        <b>Metodologi:</b> Kedua dataset dinormalisasi per 1.000 entri untuk menghilangkan bias ukuran dataset.
        Skill dengan bar Resume jauh lebih panjang dari Lowongan menunjukkan <i>oversupply</i> —
        sedangkan skill dengan bar Lowongan lebih panjang mengindikasikan <i>undersupply</i> dari sisi kandidat.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# SKKNI GAP ANALYSIS
# ═══════════════════════════════════════════════
elif page == "ID SKKNI Gap Analysis":
    st.title("ID SKKNI Skill Gap Analysis")
    st.markdown("Seberapa besar gap antara **skill internasional** terhadap **standar kompetensi SKKNI Indonesia**?")
    st.markdown("---")

    MANUAL_MAPPING = {
        "python":             ("C.29TDI01.021.1", "Programmer",      "Terpetakan"),
        "java":               ("C.29TDI01.021.1", "Programmer",      "Terpetakan"),
        "javascript":         ("C.29TDI01.021.1", "Programmer",      "Terpetakan"),
        "sql":                ("C.29TDI01.046.1", "Data Analyst",    "Terpetakan"),
        "data analysis":      ("C.29TDI01.046.1", "Data Analyst",    "Terpetakan"),
        "machine learning":   ("C.29TDI01.050.1", "Data Scientist",  "Terpetakan"),
        "deep learning":      ("C.29TDI01.050.1", "Data Scientist",  "Terpetakan"),
        "network security":   ("C.29TDI01.030.1", "Cyber Security",  "Terpetakan"),
        "penetration testing":("C.29TDI01.030.1", "Cyber Security",  "Terpetakan"),
        "encryption":         ("C.29TDI01.030.1", "Cyber Security",  "Terpetakan"),
        "graphic design":     ("C.29TDI01.060.1", "Desainer Grafis", "Terpetakan"),
        "figma":              ("C.29TDI01.060.1", "Desainer Grafis", "Terpetakan"),
        "leadership":         ("-", "Beberapa Jabatan", "Parsial"),
        "communication":      ("-", "Beberapa Jabatan", "Parsial"),
        "project management": ("-", "Beberapa Jabatan", "Parsial"),
        "agile":              ("-", "Beberapa Jabatan", "Parsial"),
        "scrum":              ("-", "Beberapa Jabatan", "Parsial"),
    }

    all_intl_skills = df_map["Skill_Internasional_Kaggle"].dropna().astype(str).str.lower().unique().tolist()

    skill_status = []
    for sk in all_intl_skills:
        # ── PERBAIKAN 3: CSV dicek dulu, MANUAL_MAPPING hanya fallback ──
        row_map   = df_map[df_map["Skill_Internasional_Kaggle"].astype(str).str.lower() == sk]
        csv_status = None

        if len(row_map) > 0:
            row       = row_map.iloc[0]
            kecocokan = str(row.get("Kecocokan", "")).lower()
            # ── PERBAIKAN 4: nama kolom disesuaikan dengan CSV asli ──
            kode      = str(row.get("Kode_Unit_SKKNI", "-"))
            jabatan   = str(row.get("Jabatan", "-"))

            if "cocok" in kecocokan and "parsial" not in kecocokan:
                csv_status = "Terpetakan"
            elif "parsial" in kecocokan or "sebagian" in kecocokan:
                csv_status = "Parsial"

            if csv_status is not None:
                skill_status.append({
                    "skill":   sk,
                    "status":  csv_status,
                    "jabatan": jabatan,
                    "kode":    kode,
                })
                continue

        # fallback ke MANUAL_MAPPING jika CSV tidak punya data valid
        if sk in MANUAL_MAPPING:
            kode, jabatan, status = MANUAL_MAPPING[sk]
            skill_status.append({
                "skill":   sk,
                "status":  status,
                "jabatan": jabatan,
                "kode":    kode,
            })
        else:
            skill_status.append({
                "skill":   sk,
                "status":  "Belum Dipetakan",
                "jabatan": "-",
                "kode":    "-",
            })

    status_df = pd.DataFrame(skill_status)

    total_intl   = len(status_df)
    mapped_n     = len(status_df[status_df["status"] == "Terpetakan"])
    partial_n    = len(status_df[status_df["status"] == "Parsial"])
    unmapped_n   = len(status_df[status_df["status"] == "Belum Dipetakan"])
    coverage_pct = round(((mapped_n + (partial_n * 0.5)) / total_intl) * 100, 1)

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f"""<div class="metric-card"><div class="metric-number" style="color:#10b981">{mapped_n}</div><div class="metric-label">Terpetakan ke SKKNI</div></div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""<div class="metric-card"><div class="metric-number" style="color:#f59e0b">{partial_n}</div><div class="metric-label">Terpetakan Parsial</div></div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown(f"""<div class="metric-card"><div class="metric-number" style="color:#ef4444">{unmapped_n}</div><div class="metric-label">Belum Dipetakan</div></div>""", unsafe_allow_html=True)
    with mc4:
        st.markdown(f"""<div class="metric-card"><div class="metric-number" style="color:#60a5fa">{coverage_pct}%</div><div class="metric-label">Coverage Rate</div></div>""", unsafe_allow_html=True)
    st.markdown("")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown('<div class="section-title">Status Pemetaan Skill</div>', unsafe_allow_html=True)
        pie_data = status_df["status"].value_counts().reset_index()
        pie_data.columns = ["Status", "Jumlah"]
        fig_pie = px.pie(
            pie_data, names="Status", values="Jumlah", hole=0.5,
            template="plotly_dark", color="Status",
            color_discrete_map={
                "Terpetakan":     "#10b981",
                "Parsial":        "#f59e0b",
                "Belum Dipetakan":"#ef4444",
            },
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=340,
            margin=dict(l=0, r=0, t=10, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_g2:
        st.markdown('<div class="section-title">Skill Terpetakan ke SKKNI</div>', unsafe_allow_html=True)
        mapped_df = status_df[status_df["status"] == "Terpetakan"]
        if len(mapped_df) > 0:
            jab_map = mapped_df["jabatan"].value_counts().reset_index()
            jab_map.columns = ["Jabatan SKKNI", "Jumlah Skill"]
            fig_jm = px.bar(
                jab_map, x="Jabatan SKKNI", y="Jumlah Skill",
                color="Jumlah Skill", color_continuous_scale="Greens",
                template="plotly_dark",
            )
            fig_jm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False, height=340,
                margin=dict(l=0, r=10, t=10, b=60),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_jm, use_container_width=True)

    st.markdown('<div class="section-title">Detail Tabel Pemetaan Skill</div>', unsafe_allow_html=True)
    filter_status = st.multiselect(
        "Filter Status",
        ["Terpetakan", "Parsial", "Belum Dipetakan"],
        default=["Terpetakan", "Parsial", "Belum Dipetakan"],
    )

    tbl = status_df[status_df["status"].isin(filter_status)].copy()

    def color_status(val):
        colors = {
            "Terpetakan":      "background-color: #064e3b; color: #6ee7b7",
            "Parsial":         "background-color: #78350f; color: #fde68a",
            "Belum Dipetakan": "background-color: #7f1d1d; color: #fca5a5",
        }
        return colors.get(val, "")

    styled = tbl.rename(columns={
        "skill":   "Skill Internasional",
        "status":  "Status Pemetaan",
        "jabatan": "Jabatan SKKNI",
        "kode":    "Kode Unit SKKNI",
    }).style.map(color_status, subset=["Status Pemetaan"])

    st.dataframe(styled, use_container_width=True, height=400)

    st.markdown('<div class="section-title">Skill Demand vs SKKNI</div>', unsafe_allow_html=True)
    resume_skill_counter = Counter()
    for lst in df_resume["skills_list"]:
        resume_skill_counter.update(lst)

    demand_rows = []
    for _, row in status_df.iterrows():
        sk = row["skill"]
        demand_rows.append({
            "Skill":          sk,
            "Demand (resume)": resume_skill_counter.get(sk, 0),
            "Status SKKNI":   row["status"],
        })

    demand_df = (
        pd.DataFrame(demand_rows)
        .sort_values("Demand (resume)", ascending=False)
        .head(30)
    )

    fig_demand = px.bar(
        demand_df, x="Skill", y="Demand (resume)", color="Status SKKNI",
        color_discrete_map={
            "Terpetakan":     "#10b981",
            "Parsial":        "#f59e0b",
            "Belum Dipetakan":"#ef4444",
        },
        template="plotly_dark",
    )
    fig_demand.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=400, margin=dict(l=0, r=10, t=10, b=100),
        xaxis_tickangle=-45,
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig_demand, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>Insight:</b> Sebagian besar skill internasional modern masih belum memiliki padanan eksplisit di SKKNI.
    Namun beberapa skill teknis seperti <b>Python, SQL, Machine Learning</b> sudah dapat dipetakan
    ke jabatan SKKNI seperti Programmer, Data Analyst, dan Data Scientist.
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#475569; font-size:0.8rem;'>"
    "© 2026 Pelet.by TIM CC26-PSU060 - Coding Camp 2026"
    "</div>",
    unsafe_allow_html=True,
)