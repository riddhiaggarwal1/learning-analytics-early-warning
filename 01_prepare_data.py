"""Step 1: turn the raw OULAD tables into one clean row per enrolment.

The raw data (seven CSV files, about 460 MB) is not in this repository.
Download it from https://analyse.kmi.open.ac.uk/open_dataset, unzip it into a
folder, and pass that folder to this script:

    python 01_prepare_data.py path/to/oulad_folder

Output: student_features.csv (one row per student per course presentation)

Cleaning decisions are marked with "CLEANING" in the comments below.
"""
import sys
from pathlib import Path

import pandas as pd

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "oulad_raw")
OUT = Path(__file__).parent / "student_features.csv"
KEY = ["code_module", "code_presentation", "id_student"]
WINDOW_DAYS = 28            # "early" = the first four weeks of the course

# CLEANING 1: the files use "?" for missing values
info = pd.read_csv(RAW / "studentInfo.csv", na_values="?")
reg = pd.read_csv(RAW / "studentRegistration.csv", na_values="?")
print(f"enrolments in raw data: {len(info):,} ({info['id_student'].nunique():,} different students)")

# One student can take several courses, so the unit of analysis is the
# enrolment (student + module + presentation), which is unique:
assert not info.duplicated(KEY).any()

df = info.merge(reg, on=KEY, how="left", validate="one_to_one")

# CLEANING 2: one deprivation band is written without its % sign
df["imd_band"] = df["imd_band"].replace({"10-20": "10-20%"})
# CLEANING 3: deprivation band is missing for some students; keep them as "unknown"
print(f"missing deprivation band: {df['imd_band'].isna().sum():,} -> labelled 'unknown'")
df["imd_band"] = df["imd_band"].fillna("unknown")
# CLEANING 4: a few registration dates are missing; use the median
print(f"missing registration date: {df['date_registration'].isna().sum()} -> median")
df["date_registration"] = df["date_registration"].fillna(df["date_registration"].median())
# CLEANING 5: merge two tiny categories into their neighbours
df["age_band"] = df["age_band"].replace({"35-55": "35 and over", "55<=": "35 and over",
                                         "0-35": "under 35"})
df["highest_education"] = df["highest_education"].replace(
    {"No Formal quals": "Lower Than A Level",
     "Post Graduate Qualification": "HE Qualification"})
df["prev_attempts"] = df["num_of_prev_attempts"].clip(upper=2)      # 0, 1, 2 or more

# Outcome: did the student complete the course successfully?
df["not_completed"] = df["final_result"].isin(["Withdrawn", "Fail"]).astype(int)

# When did the student leave? (missing = never unregistered)
df["left_before_start"] = (df["date_unregistration"] <= 0).astype(int)
df["left_in_first_4_weeks"] = df["date_unregistration"].between(1, WINDOW_DAYS - 1).astype(int)
df["days_registered_before_start"] = -df["date_registration"]

# ---- early online activity, from the 10.6 million rows of the click log ----
# Read only clicks up to the end of week 4, in chunks to keep memory low.
parts = []
for chunk in pd.read_csv(RAW / "studentVle.csv", chunksize=2_000_000):
    parts.append(chunk[chunk["date"] < WINDOW_DAYS])
clicks = pd.concat(parts)
print(f"click-log rows up to day {WINDOW_DAYS - 1}: {len(clicks):,}")

before = clicks[clicks["date"] < 0].groupby(KEY)["sum_click"].sum().rename("clicks_before_start")
early = clicks[clicks["date"] >= 0]
agg = early.groupby(KEY).agg(clicks_first_4_weeks=("sum_click", "sum"),
                             active_days_first_4_weeks=("date", "nunique"),
                             last_active_day=("date", "max"))
agg["weeks_active"] = early.assign(week=early["date"] // 7).groupby(KEY)["week"].nunique()

df = df.merge(before, on=KEY, how="left").merge(agg, on=KEY, how="left")
# CLEANING 6: a student with no rows in the click log did not log in -> 0 clicks
for col in ["clicks_before_start", "clicks_first_4_weeks", "active_days_first_4_weeks", "weeks_active"]:
    df[col] = df[col].fillna(0).astype(int)
df["last_active_day"] = df["last_active_day"].fillna(-1).astype(int)   # -1 = never in weeks 1-4

# Courses differ in how much clicking they require, so also express activity
# relative to classmates in the same presentation (0 = least, 1 = most active).
df["clicks_rank_in_class"] = (df.groupby(["code_module", "code_presentation"])
                              ["clicks_first_4_weeks"].rank(pct=True).round(3))

keep = KEY + ["gender", "highest_education", "imd_band", "age_band", "disability",
              "prev_attempts", "studied_credits", "days_registered_before_start",
              "clicks_before_start", "clicks_first_4_weeks", "active_days_first_4_weeks",
              "weeks_active", "last_active_day", "clicks_rank_in_class",
              "left_before_start", "left_in_first_4_weeks", "final_result", "not_completed"]
df[keep].to_csv(OUT, index=False)
print(f"wrote {OUT.name}: {len(df):,} rows, {len(keep)} columns")
