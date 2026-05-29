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

# Page config
st.set_page_config(
    page_title="SkillBridge AI — Skill Gap Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
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

# Sidebar navigation
with st.sidebar:
    st.markdown("## SkillBridge AI Dashboard")
    st.markdown("**Skill Gap Analysis Indonesia**")
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

# Load data
@st.cache_data
def get_all_data():
    return load_jobs(), load_resumes(), load_skkni(), load_skill_map()

df_job, df_resume, df_skkni, df_map = get_all_data()


# PAGE 1 — OVERVIEW
if page == "Overview":
    st.title("SkillBridge A — Skill Gap Dashboard")
    st.markdown("Analisis kesenjangan skill tenaga kerja Indonesia terhadap standar kompetensi nasional (SKKNI)")
    st.markdown("---")

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        (col1, "2.516", "Lowongan Kerja"),
        (col2, "10.000", "Resume Kandidat"),
        (col3, "662", "Unit Kompetensi SKKNI"),
        (col4, "114", "Skill Internasional"),
        (col5, "0%", "Sudah Dipetakan ke SKKNI"),
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

    # Key insights
    st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        st.markdown("""<div class="insight-box">
        <b>🔴 Gap Kritis</b><br>
        114 skill internasional yang paling banyak muncul di resume dan lowongan
        <b>belum satupun</b> dipetakan ke unit kompetensi SKKNI.
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
        di SKKNI (Data Analyst, Programmer).
        </div>""", unsafe_allow_html=True)

# PAGE 2 — JOB MARKET
elif page == "Job Market":
    st.title("Analisis Job Market Indonesia")
    st.markdown("Insight dari **2.516 lowongan kerja** di platform Glints")
    st.markdown("---")

    # Filters
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

    # Row 1
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

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-title">Distribusi Gaji (IDR — juta/bulan)</div>', unsafe_allow_html=True)
        salary_df = dfjf[(dfjf["has_salary"]) & (dfjf["currency"] == "IDR") & (dfjf["salary_mid"] > 0) & (dfjf["salary_mid"] < 200)]
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

    # Row 3 — Top job titles
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

# PAGE 3 — SKILL DEMAND
elif page == "Skill Demand":
    st.title("Analisis Skill Demand")
    st.markdown("Skill apa yang paling banyak dibutuhkan dari **10.000 resume** dan **2.516 lowongan**?")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Dari Resume", "Dari Lowongan", "Perbandingan"])

    # Tab 1: Resume skills
    with tab1:
        col_f, col_g = st.columns([1, 3])
        with col_f:
            n_skills = st.slider("Jumlah skill ditampilkan", 10, 50, 25)
            cat_opts = ["Semua"] + sorted(df_resume["Category"].unique().tolist())
            cat_sel = st.selectbox("Filter Kategori", cat_opts)

        dfr_filt = df_resume if cat_sel == "Semua" else df_resume[df_resume["Category"] == cat_sel]
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

        # Heatmap skill per category
        st.markdown('<div class="section-title">Heatmap: Top 15 Skill per Kategori (Top 8 Kategori)</div>', unsafe_allow_html=True)
        top_cats = df_resume["Category"].value_counts().head(8).index.tolist()
        top_global_skills = [s for s, _ in Counter(all_skills).most_common(15)]
        heat_data = {}
        for cat in top_cats:
            sub = df_resume[df_resume["Category"] == cat]
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

    # Tab 2: Job skills 
    with tab2:
        st.markdown('<div class="section-title">Top Skills dari Requirement Lowongan</div>', unsafe_allow_html=True)

        # Extract meaningful skills from requirement
        TECH_KEYWORDS = set([
            "python", "java", "javascript", "sql", "react", "node", "html", "css",
            "php", "flutter", "kotlin", "typescript", "golang", "swift",
            "aws", "azure", "docker", "kubernetes", "git", "linux",
            "machine", "learning", "tensorflow", "pytorch", "analysis",
            "excel", "tableau", "powerbi", "figma", "sketch", "photoshop",
            "management", "negotiation", "communication", "leadership",
            "scrum", "agile", "jira", "confluence", "english", "bahasa",
            "sales", "marketing", "accounting", "finance", "networking",
        ])
        req_skill_counts = Counter()
        for req in df_job["requirement"].dropna():
            words = req.lower().split()
            for w in words:
                w_clean = re.sub(r"[^a-z0-9]", "", w)
                if len(w_clean) > 3:
                    req_skill_counts[w_clean] += 1

        top_req = pd.DataFrame(req_skill_counts.most_common(30), columns=["skill", "count"])

        col_j1, col_j2 = st.columns(2)
        with col_j1:
            fig_jsk = px.bar(
                top_req.sort_values("count"), x="count", y="skill",
                orientation="h", color="count",
                color_continuous_scale="Teal",
                template="plotly_dark",
                labels={"count": "Frekuensi", "skill": "Keyword"},
            )
            fig_jsk.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False, height=500,
                margin=dict(l=0, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_jsk, use_container_width=True)

        with col_j2:
            st.markdown('<div class="section-title">Requirement Skill per Kota (Top 8)</div>', unsafe_allow_html=True)
            top_kota = df_job["kota"].value_counts().head(8).index.tolist()
            kota_skill_data = []
            for kota in top_kota:
                sub = df_job[df_job["kota"] == kota]
                n = len(sub)
                avg_skill = sub["req_skill_count"].mean()
                kota_skill_data.append({"Kota": kota, "Avg skill count": round(avg_skill, 1), "Jumlah lowongan": n})
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

    # Tab 3: Comparison
    with tab3:
        st.markdown('<div class="section-title">Perbandingan Skill: Resume vs Lowongan Kerja</div>', unsafe_allow_html=True)
        st.markdown("Seberapa besar overlap antara skill yang dimiliki kandidat dan yang diminta perusahaan?")

        common_skills = [
            "python", "java", "javascript", "sql", "communication",
            "leadership", "problem solving", "data analysis", "sales",
            "marketing", "project management", "agile", "react", "css",
        ]

        resume_counts = Counter()
        for lst in df_resume["skills_list"]:
            resume_counts.update(lst)

        job_counts = req_skill_counts

        comp_rows = []
        for sk in common_skills:
            r_count = resume_counts.get(sk, 0)
            j_count = job_counts.get(sk.replace(" ", ""), 0) + job_counts.get(sk.split()[0] if " " in sk else sk, 0)
            comp_rows.append({"skill": sk, "Resume": r_count, "Lowongan": j_count * 5})
        comp_df = pd.DataFrame(comp_rows)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Resume (kandidat)", y=comp_df["skill"], x=comp_df["Resume"],
            orientation="h", marker_color="#3b82f6",
        ))
        fig_comp.add_trace(go.Bar(
            name="Lowongan (perusahaan)", y=comp_df["skill"], x=comp_df["Lowongan"],
            orientation="h", marker_color="#10b981",
        ))
        fig_comp.update_layout(
            barmode="group", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=480, margin=dict(l=0, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.caption("*Catatan: Frekuensi dari resume = hitungan langsung dari kolom skills. Frekuensi dari lowongan = keyword extraction dari kolom requirement × 5 (scaling untuk perbandingan visual).")

# PAGE 4 — SKKNI GAP ANALYSIS
elif page == "ID SKKNI Gap Analysis":
    st.title("ID SKKNI Skill Gap Analysis")
    st.markdown("Seberapa besar gap antara **114 skill internasional** terhadap **standar kompetensi SKKNI Indonesia**?")
    st.markdown("---")

    # Gap summary 
    SKILL_TO_SKKNI = {
        "python": ("C.29TDI01.021.1", "Programmer", "mapped"),
        "java": ("C.29TDI01.021.1", "Programmer", "mapped"),
        "javascript": ("C.29TDI01.021.1", "Programmer", "mapped"),
        "sql": ("C.29TDI01.046.1", "Data Analyst", "mapped"),
        "data analysis": ("C.29TDI01.046.1", "Data Analyst", "mapped"),
        "machine learning": ("C.29TDI01.050.1", "Data Scientist", "mapped"),
        "deep learning": ("C.29TDI01.050.1", "Data Scientist", "mapped"),
        "network security": ("C.29TDI01.030.1", "Cyber security", "mapped"),
        "penetration testing": ("C.29TDI01.030.1", "Cyber security", "mapped"),
        "encryption": ("C.29TDI01.030.1", "Cyber security", "mapped"),
        "graphic design": ("C.29TDI01.060.1", "Desainer Grafis", "mapped"),
        "figma": ("C.29TDI01.060.1", "Desainer Grafis", "mapped"),
        "leadership": (None, None, "partial"),
        "communication": (None, None, "partial"),
        "project management": (None, None, "partial"),
        "agile": (None, None, "partial"),
        "scrum": (None, None, "partial"),
    }

    all_intl_skills = df_map["Skill_Internasional_Kaggle"].tolist()

    skill_status = []
    for sk in all_intl_skills:
        if sk in SKILL_TO_SKKNI:
            status_info = SKILL_TO_SKKNI[sk]
            if status_info[2] == "mapped":
                skill_status.append({"skill": sk, "status": "Terpetakan", "jabatan": status_info[1]})
            else:
                skill_status.append({"skill": sk, "status": "Parsial", "jabatan": "Beberapa jabatan"})
        else:
            skill_status.append({"skill": sk, "status": "Belum Dipetakan", "jabatan": "-"})

    status_df = pd.DataFrame(skill_status)
    mapped_n = len(status_df[status_df["status"] == "Terpetakan"])
    partial_n = len(status_df[status_df["status"] == "Parsial"])
    unmapped_n = len(status_df[status_df["status"] == "Belum Dipetakan"])

    # Summary metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-number" style="color:#10b981">{mapped_n}</div>
            <div class="metric-label">Terpetakan ke SKKNI</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-number" style="color:#f59e0b">{partial_n}</div>
            <div class="metric-label">Terpetakan Parsial</div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-number" style="color:#ef4444">{unmapped_n}</div>
            <div class="metric-label">Belum Dipetakan</div>
        </div>""", unsafe_allow_html=True)
    with mc4:
        pct = round((mapped_n + partial_n * 0.5) / len(all_intl_skills) * 100, 1)
        st.markdown(f"""<div class="metric-card">
            <div class="metric-number" style="color:#60a5fa">{pct}%</div>
            <div class="metric-label">Coverage Rate (est.)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown('<div class="section-title">Status Pemetaan 114 Skill Internasional</div>', unsafe_allow_html=True)
        pie_data = status_df["status"].value_counts().reset_index()
        pie_data.columns = ["Status", "Jumlah"]
        fig_pie = px.pie(
            pie_data, names="Status", values="Jumlah",
            hole=0.5, template="plotly_dark",
            color="Status",
            color_discrete_map={
                "Terpetakan": "#10b981",
                "Parsial": "#f59e0b",
                "Belum Dipetakan": "#ef4444",
            },
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=340,
            margin=dict(l=0, r=0, t=10, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_g2:
        st.markdown('<div class="section-title">Skill Terpetakan ke Jabatan SKKNI</div>', unsafe_allow_html=True)
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

    # Detail table
    st.markdown('<div class="section-title">Detail Tabel Pemetaan Skill</div>', unsafe_allow_html=True)

    filter_status = st.multiselect(
        "Filter Status",
        ["Terpetakan", "Parsial", "Belum Dipetakan"],
        default=["Terpetakan", "Parsial", "Belum Dipetakan"],
    )
    tbl = status_df[status_df["status"].isin(filter_status)]

    def color_status(val):
        colors = {
            "Terpetakan": "background-color: #064e3b; color: #6ee7b7",
            "Parsial": "background-color: #78350f; color: #fde68a",
            "Belum Dipetakan": "background-color: #7f1d1d; color: #fca5a5",
        }
        return colors.get(val, "")

    styled = tbl.rename(columns={
        "skill": "Skill Internasional",
        "status": "Status Pemetaan",
        "jabatan": "Jabatan SKKNI",
    }).style.applymap(color_status, subset=["Status Pemetaan"])

    st.dataframe(styled, use_container_width=True, height=400)

    # Skill demand vs SKKNI coverage 
    st.markdown('<div class="section-title">Skill Demand di Pasar vs Ketersediaan di SKKNI</div>', unsafe_allow_html=True)
    st.markdown("Skill yang paling banyak muncul di resume — mana yang sudah ada di SKKNI?")

    resume_skill_counter = Counter()
    for lst in df_resume["skills_list"]:
        resume_skill_counter.update(lst)

    demand_rows = []
    for sk in all_intl_skills:
        demand = resume_skill_counter.get(sk, 0)
        row = status_df[status_df["skill"] == sk].iloc[0]
        demand_rows.append({
            "Skill": sk,
            "Demand (resume)": demand,
            "Status SKKNI": row["status"],
        })

    demand_df = pd.DataFrame(demand_rows).sort_values("Demand (resume)", ascending=False).head(30)

    fig_demand = px.bar(
        demand_df, x="Skill", y="Demand (resume)",
        color="Status SKKNI",
        color_discrete_map={
            "Terpetakan": "#10b981",
            "Parsial": "#f59e0b",
            "Belum Dipetakan": "#ef4444",
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

    st.markdown("""<div class="insight-box">
    <b>Insight:</b> Skill dengan demand tinggi seperti <b>communication</b> dan <b>problem solving</b>
    tidak memiliki unit SKKNI yang eksplisit — padahal ini adalah soft skill paling dibutuhkan di pasar kerja Indonesia.
    Skill teknis seperti <b>Python, SQL, Machine Learning</b> sudah memiliki padanan di jabatan SKKNI
    (Programmer, Data Analyst, Data Scientist) meskipun belum dipetakan secara resmi di dataset.
    </div>""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#475569; font-size:0.8rem;'>"
    "KOLINET Skill Gap Dashboard · Data: Glints Job Scrape + KOLINET Resume Dataset + SKKNI 2024"
    "</div>",
    unsafe_allow_html=True,
)
