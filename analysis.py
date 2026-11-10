"""
Operating theatre staffing: role-only vs two competency hypotheses.

Illustrative method demo, not an assessment of any trust.
The competency weights below (SPECIALTY_TO_COMPETENCY and GRADE_COVERAGE) are
first guesses and have NOT been checked by clinicians.

HOW TO RUN
    1. Put the NHS files anywhere inside the `data/` folder next to this script.
       Unzipped CSVs or the original .zip files both work:
         - NHS Workforce Statistics (HCHS, trusts and core organisations) CSV pack:
             "Core 8. ..."  (ODPs by job role, per organisation)
             "Core 17. ..." (nurses by grade and area of work, per organisation)
         - NHS RTT (Referral to Treatment) waiting times full extract CSV
    2. pip install pandas numpy scipy
    3. python analysis.py            (add --sensitivity for the +/-30% weight test)
    Outputs go to `outputs/` (results.csv, heat.json).
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"

# --------------------------------------------------------------------------
# 1. ASSUMPTIONS (mine, not clinician-validated). Change these to explore.
# --------------------------------------------------------------------------
COMPETENCIES = ["Invasive monitoring", "Difficult airway", "Regional & positioning",
                "Paediatric", "Obstetric & gynae"]

# How much each surgical specialty's waiting list calls for each competency (0 to 1).
# Column order follows COMPETENCIES. Specialties not listed are ignored.
SPECIALTY_TO_COMPETENCY = {
    "Trauma and Orthopaedic Service":  [.1, .2, 1.0, .2, 0],
    "Ophthalmology Service":           [0, 0, .6, .3, 0],
    "Gynaecology Service":             [0, 0, .1, 0, 1.0],
    "Urology Service":                 [.1, 0, .2, .3, 0],
    "General Surgery Service":         [.4, .2, .2, .2, 0],
    "Ear Nose and Throat Service":     [0, 1.0, 0, .5, 0],
    "Oral Surgery Service":            [0, .8, 0, .4, 0],
    "Plastic Surgery Service":         [.3, .3, .5, 0, 0],
    "Neurosurgical Service":           [1.0, .4, 0, 0, 0],
    "Cardiothoracic Surgery Service":  [1.0, .2, 0, 0, 0],
    "Other - Paediatric Services":     [0, .2, 0, 1.0, 0],
    "Other - Surgical Services":       [.3, .2, .3, .1, 0],
    "Gastroenterology Service":        [0, 0, 0, 0, 0],
}

# How much experience each staff group is assumed to bring to each competency (0 to 1).
GRADE_COVERAGE = {
    "ODP_general":  [.3, .2, .7, .2, .4],   # ODP band 5/6 level
    "ODP_senior":   [1, 1, 1, 1, 1],        # specialist, advanced, manager ODPs
    "Nurse_b5":     [.3, .2, .7, .2, .4],   # anaesthetics-coded nurses, band 5 and below
    "Nurse_b6":     [.8, .8, 1, .7, .8],
    "Nurse_b7plus": [1, 1, 1, 1, 1],
}
SENIOR_NURSE_BANDS = ["Band 7", "Band 8a", "Band 8b", "Band 8c", "Band 8d", "Band 9"]

# Which trusts to include (enough data to be meaningful).
MIN_WAITING_PATIENTS = 1500
MIN_THEATRE_STAFF_FTE = 40


# --------------------------------------------------------------------------
# 2. LOADING THE DATA (looks inside data/, including zip files)
# --------------------------------------------------------------------------
def open_table(name_contains):
    """Find one CSV in data/ (or inside a .zip there) whose name has all the given
    words, and return it as a file-like object for pandas."""
    words = [w.lower() for w in name_contains]
    for path in sorted(DATA_DIR.rglob("*")):
        if path.suffix.lower() == ".csv" and all(w in path.name.lower() for w in words):
            return path
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as z:
                for member in z.namelist():
                    if member.lower().endswith(".csv") and all(w in Path(member).name.lower() for w in words):
                        return io.BytesIO(z.read(member))
    raise FileNotFoundError(f"No CSV matching {name_contains} found in {DATA_DIR}")


def latest_month(df):
    """Keep only the latest month if the file holds several."""
    if "DATA_MONTH" in df and df["DATA_MONTH"].nunique() > 1:
        df = df[df["DATA_MONTH"] == df["DATA_MONTH"].max()]
    return df


def load_waiting_list():
    """Patients waiting for an operation (a decision to admit has been made),
    per trust and specialty. Counts are in the 'Total All' column."""
    cols = ["Provider Org Code", "Provider Org Name", "RTT Part Type",
            "Treatment Function Name", "Total All"]
    rtt = pd.read_csv(open_table(["rtt", "full", "extract"]), usecols=cols)
    waiting = rtt[(rtt["RTT Part Type"] == "Part_2A")
                  & (rtt["Treatment Function Name"] != "Total")]       # 'Total' rows duplicate specialties
    waiting = waiting[waiting["Treatment Function Name"].isin(SPECIALTY_TO_COMPETENCY)]
    return (waiting.groupby(["Provider Org Code", "Provider Org Name", "Treatment Function Name"])
                   ["Total All"].sum().reset_index()
                   .rename(columns={"Provider Org Code": "code", "Provider Org Name": "name",
                                    "Treatment Function Name": "specialty", "Total All": "patients"}))


def load_theatre_staff():
    """Theatre staff FTE per trust, split into the groups used in GRADE_COVERAGE."""
    # ODPs by job role (Core 8). Students are excluded.
    odp = latest_month(pd.read_csv(open_table(["core 8"])))
    role = odp["NWD_JOB_ROLE"].fillna("")
    odp = odp[role.str.contains("Operating Department Practitioner") & ~role.str.contains("Student")].copy()
    odp["group"] = np.where(odp["NWD_JOB_ROLE"] == "Operating Department Practitioner",
                            "ODP_general", "ODP_senior")
    odp_fte = odp.pivot_table(index="ORG_CODE", columns="group", values="FTE", aggfunc="sum", fill_value=0)

    # Nurses whose area of work is Anaesthetics, by pay band (Core 17).
    nurses = latest_month(pd.read_csv(open_table(["core 17"])))
    nurses = nurses[(nurses["STAFF_GROUP_1"] == "Nurses & health visitors")
                    & (nurses["SECONDARY_AREA_OF_WORK"] == "Anaesthetics")].copy()
    nurses["group"] = nurses["GRADE"].map(
        lambda g: "Nurse_b6" if g == "Band 6" else "Nurse_b7plus" if g in SENIOR_NURSE_BANDS else "Nurse_b5")
    nurse_fte = nurses.pivot_table(index="ORG_CODE", columns="group", values="FTE", aggfunc="sum", fill_value=0)

    staff = odp_fte.join(nurse_fte, how="outer").fillna(0)
    staff = staff.reindex(columns=list(GRADE_COVERAGE), fill_value=0)
    staff["total"] = staff.sum(axis=1)
    return staff


# --------------------------------------------------------------------------
# 3. THE THREE VIEWS
# --------------------------------------------------------------------------
def competency_demand(waiting, weights):
    """Waiting patients turned into demand for each competency, per trust."""
    w = pd.DataFrame(weights, index=COMPETENCIES).T
    merged = waiting.merge(w, left_on="specialty", right_index=True)
    for c in COMPETENCIES:
        merged[c] = merged[c] * merged["patients"]
    by_trust = merged.groupby(["code", "name"])[COMPETENCIES].sum()
    by_trust["patients"] = merged.groupby(["code", "name"])["patients"].sum()
    return by_trust.reset_index()


def competency_cover(staff, codes, coverage):
    """Experience-weighted staff available for each competency, per trust."""
    cov = pd.DataFrame(coverage, index=COMPETENCIES)
    out = {c: sum(staff.loc[codes, g].values * cov.loc[c, g] for g in cov.columns) for c in COMPETENCIES}
    return pd.DataFrame(out)


def three_views(trusts, demand, cover):
    """Return (role_only, hypothesis1, hypothesis2). 1.0 = national average."""
    patients, staff = trusts["patients"].values, trusts["staff"].values
    role_only = (patients / staff) / (patients.sum() / staff.sum())
    # Hypothesis 1: staff competencies spread evenly -> look at what demand calls for.
    h1 = pd.DataFrame({c: (demand[c].values / patients) / (demand[c].sum() / patients.sum())
                       for c in COMPETENCIES})
    # Hypothesis 2: demand spread evenly -> look at how thinly the staff mix covers each competency.
    h2 = pd.DataFrame({c: (staff / cover[c].values) / (staff.sum() / cover[c].sum())
                       for c in COMPETENCIES})
    return role_only, h1, h2


def to_rank(df):
    """Rank each column 0 (lowest shortage risk) to 100 (highest) across trusts."""
    return df.apply(lambda col: (rankdata(col) - 1) / (len(col) - 1) * 100)


# --------------------------------------------------------------------------
# 4. RUN
# --------------------------------------------------------------------------
def build_trust_table(waiting, staff):
    demand = competency_demand(waiting, SPECIALTY_TO_COMPETENCY)
    table = demand.merge(staff[["total"]].rename(columns={"total": "staff"}),
                         left_on="code", right_index=True)
    table = table[(table["patients"] >= MIN_WAITING_PATIENTS) & (table["staff"] >= MIN_THEATRE_STAFF_FTE)]
    return table.reset_index(drop=True)


def sensitivity(waiting, staff, trusts, top1, top2, runs=300):
    """Nudge every guessed weight by up to +/-30% and see how stable the biggest competency risk is."""
    rng = np.random.default_rng(1)
    same1, same2 = [], []
    for _ in range(runs):
        w = {k: np.array(v) * rng.uniform(.7, 1.3, len(v)) for k, v in SPECIALTY_TO_COMPETENCY.items()}
        cov = {k: np.minimum(np.array(v) * rng.uniform(.7, 1.3, len(v)), 1.0) for k, v in GRADE_COVERAGE.items()}
        d = competency_demand(waiting, w).set_index("code").reindex(trusts["code"]).reset_index(drop=True)
        cv = competency_cover(staff, trusts["code"], cov)
        _, h1, h2 = three_views(trusts, d, cv)
        same1.append(np.mean(to_rank(h1).values.argmax(1) == top1))
        same2.append(np.mean(to_rank(h2).values.argmax(1) == top2))
    print(f"+/-30% weight noise: biggest competency risk unchanged for "
          f"{100*np.mean(same1):.0f}% of trusts (Hypothesis 1) vs {100*np.mean(same2):.0f}% (Hypothesis 2)")


def main(run_sensitivity=False):
    waiting, staff = load_waiting_list(), load_theatre_staff()
    trusts = build_trust_table(waiting, staff)
    demand = trusts[COMPETENCIES]
    cover = competency_cover(staff, trusts["code"], GRADE_COVERAGE)
    role_only, h1, h2 = three_views(trusts, demand, cover)
    r1, r2 = to_rank(h1), to_rank(h2)

    print(f"{len(trusts)} trusts | {int(trusts['patients'].sum()):,} waiting patients | "
          f"{trusts['staff'].sum():,.0f} theatre staff FTE")
    print("Agreement between Hypothesis 1 and 2, per competency (rank correlation):")
    for c in COMPETENCIES:
        print(f"   {c:24s} {spearmanr(h1[c], h2[c]).correlation:+.2f}")
    print(f"Role-only vs Hypothesis 1 average rank: {spearmanr(role_only, r1.mean(axis=1)).correlation:+.2f}")
    top1, top2 = r1.values.argmax(1), r2.values.argmax(1)
    print(f"Same biggest competency risk under both hypotheses: {100*np.mean(top1 == top2):.0f}% of trusts (chance 20%)")
    if run_sensitivity:
        sensitivity(waiting, staff, trusts, top1, top2)

    OUT_DIR.mkdir(exist_ok=True)
    results = trusts[["code", "name", "patients", "staff"]].assign(role_only=role_only)
    for c in COMPETENCIES:
        results["H1 " + c], results["H2 " + c] = h1[c].values, h2[c].values
    results.to_csv(OUT_DIR / "results.csv", index=False)
    rows = [{"n": trusts.loc[i, "name"].title(), "N": int(trusts.loc[i, "patients"]),
             "S": round(float(trusts.loc[i, "staff"])), "a": round(float(role_only[i]), 2),
             "p1": [round(float(x)) for x in r1.iloc[i]], "p2": [round(float(x)) for x in r2.iloc[i]]}
            for i in np.argsort(-role_only)]
    (OUT_DIR / "heat.json").write_text(json.dumps(rows, separators=(",", ":")))
    print(f"Wrote {OUT_DIR / 'results.csv'} and {OUT_DIR / 'heat.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensitivity", action="store_true", help="also run the +/-30% weight test (slower)")
    main(ap.parse_args().sensitivity)
