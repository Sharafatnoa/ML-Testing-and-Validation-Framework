# ML Testing & Validation Framework

**Quality gates for machine learning pipelines, built with software QA principles.**

Most ML projects check one thing: accuracy. This framework treats an ML pipeline like any other software system. Every stage has automated tests with pass/fail rules, and **the pipeline stops when data or model quality drops below an agreed limit**, the same way a failing test blocks a software release.

> **Status:** early development. The data-quality gates work and are tested. Model, fairness and reporting gates are planned (see [Roadmap](#roadmap)).

---

## What is a quality gate?

A gate is a checkpoint with three parts:

1. **A measurement:** one number, for example the share of missing values in a column.
2. **A threshold:** a limit decided in advance and stored in `src/config.py`.
3. **An action:** pass, and the pipeline continues. Fail, and it stops with a clear message.

**Example:** the missing-value limit is 2%.

| Batch | Rows | Missing `savings` | Missing rate | Limit | Result |
|---|---|---|---|---|---|
| Week 1 | 2,000 | 30 | 1.5% | 2.0% | ✅ PASS, continue to training |
| Week 2 | 2,000 | 80 | 4.0% | 2.0% | ❌ FAIL, the model is never trained on this batch |

The gate catches the problem at the cheapest moment: before a model is built on top of bad data.

---

## Pipeline design

```
New data
   │
   ▼
1. DATA GATES        schema, allowed values, missing values, outliers    ✅ implemented
   │
   ▼
2. TRAIN MODEL       scikit-learn baseline                               📋 planned
   │
   ▼
3. MODEL GATES       accuracy ≥ 0.80, ROC AUC ≥ 0.75                     📋 planned
   │
   ▼
4. FAIRNESS GATES    true-positive-rate gap by gender ≤ 0.12,            📋 planned
   │                 by age group ≤ 0.15
   ▼
5. REGRESSION GATE   new model is not worse than the saved model         📋 planned
   │
   ▼
6. REPORT            one pass/fail table with every number                📋 planned
   │
   ▼
7. CI                GitHub Actions runs everything on every push         📋 planned
```

---

## What works today

### Synthetic dataset with fault injection

`src/data.py` generates a synthetic credit dataset (age, gender, age group, income, debt, savings, and a 0/1 label). It can **deliberately inject faults**, so each gate can be tested on both good and bad data:

```python
from src.data import makeDataSet

clean_data = makeDataSet(seed=1)
data_with_gaps = makeDataSet(seed=1, missing_rate=0.05)       # 5% of savings set to missing
data_with_spikes = makeDataSet(seed=1, outlier_rate=0.10)     # 10% of incomes set to 50× the median
```

### Data gates

| Gate | Rule | Test file |
|---|---|---|
| Schema | All expected columns are present | `tests/test_schema.py` |
| Allowed values | Age 18–59, gender in {F, M}, valid age groups, label in {0, 1} | `tests/test_schema.py` |
| Missing values | No column has more than 2% missing values | `tests/test_missing_values.py` |
| Outliers | No numeric column has more than 2% outliers | `tests/test_outliers.py` |
| Outlier gate triggers | The gate fails when outliers are injected | `tests/test_outliers.py` |

### Outlier detection: median and MAD

Outliers are detected with a **robust z-score** based on the median and the median absolute deviation (MAD), not the mean and standard deviation:

```
robust_z = 0.6745 × (value − median) / MAD
outlier if |robust_z| > 6.0
```

**Why not mean and standard deviation?** One extreme value distorts the ruler used to measure it. For the values `[100, 102, 98, 101, 99, 100, 103, 97, 100, 40]`:

| Method | Centre | Spread | Score for 40 | Caught at threshold 3.5? |
|---|---|---|---|---|
| Mean / std | 94 | 18.1 | −3.0 | ❌ No, the outlier hides itself |
| Median / MAD | 100 | 1.5 | −27.0 | ✅ Yes |

The median and MAD barely move when one bad value is added, so the gate is stable.

---

## Lessons learned

**A green test suite can hide a broken gate.** An earlier version of the outlier function subtracted the MAD instead of the median. On the example above it flagged **100%** of values as outliers instead of 10%, yet all tests still passed. The tests only checked that rates stayed under a limit on synthetic data; none checked the function against inputs with a known answer.

The fix was one word. The lasting change is a rule for this project: **every gate needs two kinds of tests.**

- **Gate tests** check real data against a threshold. They fail when the *data* is bad.
- **Known-answer unit tests** check the gate's code on small inputs where the correct result is known. They fail when the *code* is wrong.

**Known limitation:** when more than half of a column's values are identical, the MAD is 0 and the function returns 0 outliers. For `[5, 5, 5, 5, 500]` the value 500 is missed. A fallback for this case is on the roadmap.

---

## Getting started

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Sharafatnoa/ML-Testing-and-Validation-Framework.git
cd ML-Testing-and-Validation-Framework

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pytest -v
```

---

## Project structure

```
src/
  config.py          data contract (columns, allowed values) and gate thresholds
  data.py            synthetic dataset generator with fault injection
  validations.py     gate logic (robust outlier rate)
tests/
  test_schema.py           schema and allowed-value gates
  test_missing_values.py   missing-value gate
  test_outliers.py         outlier gate, including a test that it triggers
```

---

## Roadmap

### Phase 1: Solid foundation
- [x] Fix the outlier formula (median, not MAD)
- [ ] Known-answer unit tests for `robust_outlier_rate`, including the MAD = 0 case
- [ ] Test that the missing-value gate **fails** when 5% of values are missing
- [ ] `requirements.txt` and a GitHub Actions workflow that runs `pytest` on every push

### Phase 2: Full pipeline
- [ ] Every gate returns the same result format: name, value, limit, passed
- [ ] `train.py`: a baseline scikit-learn model
- [ ] Model gates: accuracy and ROC AUC thresholds
- [ ] Fairness gates: true-positive-rate gap by gender and by age group
- [ ] `pipeline.py` runs every stage in order and stops at the first failure
- [ ] `report.py` prints a pass/fail table, for example:

```
missing_rate      0.015  ≤ 0.020   PASS
outlier_rate      0.005  ≤ 0.020   PASS
accuracy          0.83   ≥ 0.80    PASS
tpr_gap_gender    0.15   ≤ 0.12    FAIL  → pipeline stopped
```

### Phase 3: Works on any tabular dataset
- [ ] Move dataset rules out of the code into a YAML **data contract** per dataset, so the same engine runs on any table:
  ```bash
  python -m src.pipeline --data data/credit.csv --contract contracts/credit.yaml
  ```
- [ ] Policy for unexpected columns: `strict` (fail), `warn` (report and drop before training) or `profile` (report statistics)
- [ ] Regression gate: a new model may not be worse than the saved one by more than 0.01 accuracy
- [ ] Run a second, real dataset with only a new contract file and no code changes

### Phase 4: LLM-generated tests
- [ ] Write data requirements in plain English, for example *"age must be between 18 and 59"*
- [ ] Ask an LLM to turn them into pytest tests
- [ ] Inject known faults and measure how many the generated tests catch, and how many false alarms they raise on clean data, compared with hand-written tests

---

## Author

**Shamsunnur Ibn Arefin**: MSc Data Science student at the University of Skövde, with 5+ years of experience as a software QA engineer. This project applies that QA background to machine learning systems.
