# Operating theatre staffing: three ways of looking at the same data

An illustrative method demo, not an assessment of any trust. It compares three views of 88 English acute NHS trusts (August 2026):

1. **Role-only:** waiting patients per theatre staff member, relative to the average.
2. **Hypothesis 1 (demand-driven):** assumes staff competencies are spread evenly and asks which competencies each trust's own waiting list calls for.
3. **Hypothesis 2 (staff-driven):** assumes demand is spread evenly and asks which competencies each trust's staff mix (seniority and grade) can cover.

The main finding is that headcount alone hides competency shortage risk. The three views disagree about which trusts are under pressure, which is the case for modelling competencies and seniority explicitly.

## Data (real)
- NHS England, NHS Workforce Statistics (HCHS, trusts and core organisations), August 2026: ODPs by job role and grade (Core 8) and registered nurses recorded to the Anaesthetics area of work, by band (Core 17).
- NHS England, Referral to Treatment (RTT) waiting times, full extract, August 2026: incomplete pathways with a decision to admit, by trust and treatment function (count in `Total All`).
- Licence: Open Government Licence v3.0.

## Stand-ins and assumptions (mine, not clinician-validated)
- Waiting list as a proxy for theatre workload. Obstetrics is not in RTT, so obstetric/gynae is only a gynaecology proxy. Emergency work is not captured.
- Nurses coded to Anaesthetics as the nursing part of theatre staff. Theatre nurses recorded under other areas are missed.
- Five competency groups, the specialty-to-competency weights (`SPECIALTY_TO_COMPETENCY`) and the grade-to-competency experience weights (`GRADE_COVERAGE`) (see `analysis.py`).
- Trusts included: at least 1,500 such pathways and 40 theatre staff FTE (88 trusts).
- Hypothesis 2 mostly reflects seniority only, because `GRADE_COVERAGE` differs only by grade.
- One month of data. Results were stable to ±30% perturbation of `W` and `q` for correlations, and for the top competency under Hypothesis 1 (93%) but not Hypothesis 2 (30%).

## Competency framework 

The five competency groups (invasive monitoring, difficult airway, regional/positioning, children, obstetric/gynae) were chosen by hand from the Operating Department Practitioner (ODP) competency framework. The framework file is not included in this repo. analysis.py does not read it; it only uses the five group names.

Framework: ODP competency framework (enriched and formalised version, March 2025).

The weights linking specialties and grades to these competencies are the author's guesses, not taken from the framework. A competency-level model of who holds which competency would replace them.

## Files
- `index.html`: the self-contained page (no external calls).
- `analysis.py`: the analysis. Put the NHS files anywhere inside `data/` (unzipped CSVs or the original .zip files both work), then run `pip install pandas numpy scipy` and `python analysis.py` (add `--sensitivity` for the +/-30% weight test). It prints the headline numbers and writes `outputs/results.csv` and `outputs/heat.json`. The assumptions are at the top of the script and are labelled as guesses.
- `data/`: your local copy of the NHS files. They are not included in this repo, to keep it small; download them from NHS England.

## Online visual
[https://honghan.github.io/workforce_competency/](https://honghan.github.io/workforce_competency/)
