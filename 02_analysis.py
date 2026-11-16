"""Step 2: which students are at risk of not completing, judged after four weeks?

Reads student_features.csv (made by 01_prepare_data.py) and writes
result tables (CSV) and charts (PNG) next to this script.

Usage:  python 02_analysis.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent
df = pd.read_csv(HERE / "student_features.csv")

# ------------------------------------------------------------ 1. overview
outcomes = (df["final_result"].value_counts()
            .reindex(["Distinction", "Pass", "Fail", "Withdrawn"])
            .rename_axis("final_result").reset_index(name="enrolments"))
outcomes["share_pct"] = (100 * outcomes["enrolments"] / len(df)).round(1)
outcomes.to_csv(HERE / "outcomes.csv", index=False)
print(outcomes.to_string(index=False))
print(f"\nleft before the course started:  {df['left_before_start'].sum():,}")
print(f"left during the first 4 weeks:   {df['left_in_first_4_weeks'].sum():,}")

# The early-warning question only makes sense for students who are still
# enrolled at the end of week 4, so everyone who left earlier is set aside.
at4 = df[(df["left_before_start"] == 0) & (df["left_in_first_4_weeks"] == 0)].copy()
base = at4["not_completed"].mean()
print(f"still enrolled after week 4:     {len(at4):,}  "
      f"({base:.1%} of them later fail or withdraw)\n")

# ------------------------------------------- 2. early activity and outcome
at4["activity_band"] = pd.cut(at4["active_days_first_4_weeks"], [-1, 0, 3, 7, 14, 28],
                              labels=["0 days", "1 to 3", "4 to 7", "8 to 14", "15 or more"])


def rate_table(col):
    t = at4.groupby(col, observed=True)["not_completed"].agg(students="size", rate="mean")
    t["not_completed_pct"] = (100 * t.pop("rate")).round(1)
    return t.reset_index()


by_activity = rate_table("activity_band")
by_activity.to_csv(HERE / "non_completion_by_activity.csv", index=False)
print(by_activity.to_string(index=False), "\n")

IMD_ORDER = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%",
             "60-70%", "70-80%", "80-90%", "90-100%"]
by_imd = rate_table("imd_band").set_index("imd_band").loc[IMD_ORDER].reset_index()
by_imd.to_csv(HERE / "non_completion_by_deprivation.csv", index=False)
print(by_imd.to_string(index=False), "\n")
by_edu = rate_table("highest_education")
print(by_edu.to_string(index=False), "\n")

# --------------------------------------------------- 3. statistical tests
# With 27,000 students almost any difference is "significant", so each test
# is reported together with an effect size.
done = at4.loc[at4["not_completed"] == 0, "active_days_first_4_weeks"]
not_done = at4.loc[at4["not_completed"] == 1, "active_days_first_4_weeks"]
u, p_u = stats.mannwhitneyu(done, not_done, alternative="two-sided")
prob_superiority = u / (len(done) * len(not_done))   # P(completer more active than non-completer)

known = at4[at4["imd_band"] != "unknown"]
table = pd.crosstab(known["imd_band"], known["not_completed"])
chi2, p_chi, dof, _ = stats.chi2_contingency(table)
cramers_v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))

tests = pd.DataFrame([
    {"question": "Are completers more active in weeks 1-4? (Mann-Whitney U)",
     "detail": f"median active days {done.median():.0f} vs {not_done.median():.0f}",
     "p_value": f"{p_u:.1e}" if p_u > 0 else "below 1e-300",
     "effect_size": f"probability of superiority {prob_superiority:.2f}"},
    {"question": "Is non-completion related to deprivation band? (chi-square)",
     "detail": f"chi2 = {chi2:.0f}, {dof} degrees of freedom",
     "p_value": f"{p_chi:.1e}", "effect_size": f"Cramer's V {cramers_v:.2f}"}])
tests.to_csv(HERE / "statistical_tests.csv", index=False)
print(tests.to_string(index=False), "\n")

# ------------------------------------------------------------ 4. models
# Train on the 2013 courses, test on the 2014 courses: the way a university
# would really use it (learn from last year, apply to this year).
train = at4[at4["code_presentation"].str.startswith("2013")]
test = at4[at4["code_presentation"].str.startswith("2014")]
print(f"train (2013): {len(train):,}   test (2014): {len(test):,}")

BACKGROUND_CAT = ["gender", "highest_education", "imd_band", "age_band", "disability"]
BACKGROUND_NUM = ["prev_attempts", "studied_credits"]
BEHAVIOUR = ["days_registered_before_start", "clicks_before_start", "clicks_first_4_weeks",
             "active_days_first_4_weeks", "weeks_active", "last_active_day",
             "clicks_rank_in_class"]
LOG = ["clicks_before_start", "clicks_first_4_weeks"]       # very skewed counts


def prepare(frame):
    out = frame.copy()
    out[LOG] = np.log1p(out[LOG])
    return out


def logistic(num, cat):
    steps = [("num", StandardScaler(), num)]
    if cat:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    return Pipeline([("prep", ColumnTransformer(steps)),
                     ("model", LogisticRegression(max_iter=2000))])


boosting = Pipeline([
    ("prep", ColumnTransformer([("num", "passthrough", BACKGROUND_NUM + BEHAVIOUR),
                                ("cat", OneHotEncoder(handle_unknown="ignore"), BACKGROUND_CAT)],
                               sparse_threshold=0)),
    ("model", HistGradientBoostingClassifier(random_state=42))])

candidates = {
    "Background only (logistic regression)": logistic(BACKGROUND_NUM, BACKGROUND_CAT),
    "Early behaviour only (logistic regression)": logistic(BEHAVIOUR, []),
    "Background + behaviour (logistic regression)": logistic(BACKGROUND_NUM + BEHAVIOUR, BACKGROUND_CAT),
    "Background + behaviour (gradient boosting)": boosting,
}
X_train, X_test = prepare(train), prepare(test)
y_train, y_test = train["not_completed"], test["not_completed"]

rows, probs = [], {}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    probs[name] = model.predict_proba(X_test)[:, 1]
    rows.append({"model": name, "test_auc": round(roc_auc_score(y_test, probs[name]), 3)})
models = pd.DataFrame(rows)
models.to_csv(HERE / "model_comparison.csv", index=False)
print(models.to_string(index=False), "\n")

# ------------------------------------------ 5. what would a tutor team get?
# Rank the 2014 students by predicted risk with the behaviour-only model
# and look at the top of the list.
chosen = "Early behaviour only (logistic regression)"
ranked = pd.DataFrame({"risk": probs[chosen], "not_completed": y_test.values}
                      ).sort_values("risk", ascending=False).reset_index(drop=True)
total_at_risk = ranked["not_completed"].sum()
rows = []
for share in [0.10, 0.20, 0.30, 0.50]:
    top = ranked.head(int(len(ranked) * share))
    rows.append({"students_contacted_pct": int(share * 100),
                 "students_contacted": len(top),
                 "of_whom_later_not_completed_pct": round(100 * top["not_completed"].mean(), 1),
                 "share_of_all_non_completers_reached_pct":
                     round(100 * top["not_completed"].sum() / total_at_risk, 1)})
outreach = pd.DataFrame(rows)
outreach.to_csv(HERE / "outreach_scenarios.csv", index=False)
# For comparison: a rule a tutor could apply without any model
rule = test["active_days_first_4_weeks"] <= 3
print(f"simple rule 'active on 3 days or fewer': flags {rule.mean():.1%} of students, "
      f"{y_test[rule].mean():.1%} of them do not complete, "
      f"reaching {y_test[rule].sum() / total_at_risk:.1%} of all non-completers")
print(f"base rate in the 2014 test group: {y_test.mean():.1%}")
print(outreach.to_string(index=False), "\n")

# ----------------------------------------------------------------- charts
BLUE, ORANGE, INK, SUB, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "font.size": 10, "text.color": INK, "axes.labelcolor": SUB,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#c3c2b7",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.titlepad": 14})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(HERE / name, dpi=160)
    plt.close(fig)


def barh(labels, values, title, name, fmt="{:.0f}%", refline=None, xmax=None):
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(labels) + 1.3))
    bars = ax.barh(labels, values, color=BLUE, height=0.6)
    ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=4, color=SUB)
    if refline is not None:
        ax.axvline(refline, color=MUTED, lw=1, ls="--")
        ax.text(refline, len(labels) - 0.35, f" overall {refline:.0f}%", color=MUTED, fontsize=9)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0, labelcolor=SUB)
    ax.set_xlim(0, xmax or max(values) * 1.18)
    ax.invert_yaxis()
    save(fig, name)


barh([f"{b}  (n={n:,})" for b, n in zip(by_activity["activity_band"], by_activity["students"])],
     by_activity["not_completed_pct"],
     "Share who later fail or withdraw, by days active online in weeks 1 to 4",
     "non_completion_by_activity.png", refline=100 * base)

barh(list(by_imd["imd_band"]), by_imd["not_completed_pct"],
     "Share who later fail or withdraw, by deprivation band of home area\n"
     "(0-10% = most deprived areas)", "non_completion_by_deprivation.png", refline=100 * base)

barh([m.replace(" (", "\n(") for m in models["model"]], models["test_auc"],
     "Model quality on the 2014 courses (AUC)\n0.5 = guessing, 1.0 = perfect ranking",
     "model_comparison.png", fmt="{:.2f}", xmax=1.0)

# cumulative gains
fig, ax = plt.subplots(figsize=(6, 5))
x = np.arange(1, len(ranked) + 1) / len(ranked) * 100
ax.plot(x, ranked["not_completed"].cumsum() / total_at_risk * 100, color=BLUE, lw=2,
        label="Ranked by the model")
ax.plot([0, 100], [0, 100], color=MUTED, lw=1, ls="--", label="Contacting students at random")
r20 = outreach.loc[outreach["students_contacted_pct"] == 20,
                   "share_of_all_non_completers_reached_pct"].iloc[0]
ax.scatter([20], [r20], color=BLUE, s=50, zorder=3, edgecolor="#fcfcfb", linewidth=2)
ax.annotate(f"contact 20 %, reach {r20:.0f} %", (20, r20), xytext=(14, -22),
            textcoords="offset points", color=SUB)
ax.set_title("Students reached by contacting the highest-risk first")
ax.set_xlabel("Share of students contacted (%)")
ax.set_ylabel("Share of eventual non-completers reached (%)")
ax.grid(color=GRID, lw=0.8)
ax.tick_params(length=0)
ax.legend(frameon=False, loc="lower right", labelcolor=SUB)
save(fig, "outreach_gains.png")

print("Done.")
