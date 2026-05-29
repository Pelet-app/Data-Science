import pandas as pd
import numpy as np
import ast
import re
from functools import lru_cache


# City extraction helper
CITY_MAP = {
    "jakarta selatan": "Jakarta Selatan",
    "jakarta barat": "Jakarta Barat",
    "jakarta pusat": "Jakarta Pusat",
    "jakarta utara": "Jakarta Utara",
    "jakarta timur": "Jakarta Timur",
    "jakarta raya": "Jakarta",
    "dki jakarta": "Jakarta",
    "jakarta": "Jakarta",
    "bandung": "Bandung",
    "surabaya": "Surabaya",
    "tangerang selatan": "Tangerang Selatan",
    "tangerang": "Tangerang",
    "bekasi": "Bekasi",
    "yogyakarta": "Yogyakarta",
    "sleman": "Yogyakarta",
    "semarang": "Semarang",
    "medan": "Medan",
    "depok": "Depok",
    "bogor": "Bogor",
    "malang": "Malang",
    "bali": "Bali",
    "denpasar": "Bali",
    "makassar": "Makassar",
    "palembang": "Palembang",
    "balikpapan": "Balikpapan",
}


def extract_city(loc: str) -> str:
    if pd.isna(loc):
        return "Lainnya"
    loc_lower = loc.lower()
    for key, val in CITY_MAP.items():
        if key in loc_lower:
            return val
    return loc.split(",")[0].strip().title()


# Skills parser
def parse_skills(s):
    if pd.isna(s):
        return []
    if isinstance(s, list):
        return s
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed]
    except Exception:
        pass
    return [x.strip().lower() for x in str(s).split("|") if x.strip()]


# Job dataset 
@lru_cache(maxsize=1)
def load_jobs() -> pd.DataFrame:
    df = pd.read_csv("dataset/cleaned_job_5.csv")

    # City
    df["kota"] = df["location"].apply(extract_city)

    # Salary: keep only rows with real salary (non-zero)
    df["has_salary"] = (df["min_salary"] > 0) | (df["max_salary"] > 0)
    df["salary_mid"] = df.apply(
        lambda r: (r["min_salary"] + r["max_salary"]) / 2
        if r["has_salary"]
        else np.nan,
        axis=1,
    )

    # Type of work: fill nulls
    df["type_of_work"] = df["type_of_work"].fillna("Tidak Diketahui")

    # Currency fill
    df["currency"] = df["currency"].fillna("Tidak Diketahui")

    # Skills from requirement column
    df["req_skills"] = df["requirement"].apply(
        lambda x: [w.strip() for w in str(x).split() if len(w) > 2]
        if pd.notna(x)
        else []
    )
    df["req_skill_count"] = df["req_skills"].apply(len)

    # Experience range
    df["exp_mid"] = (df["min_work_experience"] + df["max_work_experience"]) / 2

    # Title cleaned
    df["title_clean"] = df["title"].str.lower().str.strip()

    return df


# Resume dataset
@lru_cache(maxsize=1)
def load_resumes() -> pd.DataFrame:
    df = pd.read_csv("dataset/cleaned_training_data.csv")
    df["skills_list"] = df["skills_clean"].apply(parse_skills)
    df["skill_count"] = df["skills_list"].apply(len)
    df["exp_bucket"] = pd.cut(
        df["Experience Years"],
        bins=[0, 1, 3, 5, 10, 100],
        labels=["Fresh (<1)", "Junior (1–3)", "Mid (3–5)", "Senior (5–10)", "Expert (10+)"],
        right=True,
    )
    return df


# SKKNI dataset
@lru_cache(maxsize=1)
def load_skkni() -> pd.DataFrame:
    df = pd.read_csv("dataset/skkni_reference_clean.csv")
    return df


# Skill mapping dictionary 
@lru_cache(maxsize=1)
def load_skill_map() -> pd.DataFrame:
    df = pd.read_csv("dataset/skill_mapping_dictionary.csv")
    return df


# Aggregated skill frequency from resumes
@lru_cache(maxsize=1)
def skill_freq_by_category():
    df = load_resumes()
    rows = []
    for _, row in df.iterrows():
        for sk in row["skills_list"]:
            rows.append({"category": row["Category"], "skill": sk})
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def top_skills_overall(n: int = 30):
    from collections import Counter
    df = load_resumes()
    all_skills = []
    for lst in df["skills_list"]:
        all_skills.extend(lst)
    counts = Counter(all_skills).most_common(n)
    return pd.DataFrame(counts, columns=["skill", "count"])


# Job skill frequency
@lru_cache(maxsize=1)
def job_skill_freq(n: int = 30):
    from collections import Counter
    df = load_jobs()
    all_req = []
    for lst in df["req_skills"]:
        all_req.extend(lst)
    counts = Counter(all_req).most_common(n)
    return pd.DataFrame(counts, columns=["skill", "count"])
