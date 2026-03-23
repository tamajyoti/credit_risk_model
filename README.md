# Credit Risk

A data pipeline and ML inference service for predicting 90-day consumer default risk from transaction history.

---

## Project Structure

```
.
├── data/
│   ├── transactions.csv          # Raw customer transactions
│   └── labels.csv                # 90-day default labels
├── artifacts/
│   ├── model.joblib              # Pre-trained logistic regression (provided)
│   ├── scaler.joblib             # StandardScaler fitted on training data (generated)
│   ├── scaler_params.json        # Human-readable scaler mean/std (generated)
│   └── training_set.csv          # Feature-engineered training set (generated)
├── prepare_data.py               # Part 1: data engineering + scaler fitting
├── app.py                        # Part 2: FastAPI inference service
├── logistic_regression_explorer.ipynb  # Model coefficient and scaling analysis
├── Dockerfile                    # Two-stage build: prepare → serve
├── Makefile                      # Single-command runner
├── requirements.txt
└── README.md
```

---

## Quickstart — run with Docker (recommended)

Requires Docker installed. One command does everything: builds the image, runs `prepare_data.py` inside the container, and starts the API.

```bash
make run
```

The API will be available at `http://localhost:8000`.

To stop:

```bash
make stop
```

---

## Quickstart run locally

```bash
git clone <repo-url>
cd credit_risk_model

pip install -r requirements.txt

# Step 1: build training set + fit scaler
python prepare_data.py

# Step 2: start the API
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.

---

## Testing the API

### Interactive docs

Open `http://localhost:8000/docs` in the browser FastAPI generates a full Swagger UI where we can test both endpoints directly.

### Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "scaler_loaded": true
}
```

### POST /predict — raw features (no scaling)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id":  "CUST_0002",
    "txn_count":    2,
    "total_debit":  -650.0,
    "total_credit": 1800.0,
    "avg_amount":   575.0,
    "kw_rent":      1,
    "kw_netflix":   0,
    "kw_tesco":     0,
    "kw_payroll":   1,
    "kw_bonus":     0
  }'
```

```json
{
  "customer_id": "CUST_0002",
  "probability": 0.0,
  "prediction": 0,
  "scaled": false
}
```

### POST /predict/scaled — recommended route

```bash
curl -X POST http://localhost:8000/predict/scaled \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id":  "CUST_0002",
    "txn_count":    2,
    "total_debit":  -650.0,
    "total_credit": 1800.0,
    "avg_amount":   575.0,
    "kw_rent":      1,
    "kw_netflix":   0,
    "kw_tesco":     0,
    "kw_payroll":   1,
    "kw_bonus":     0
  }'
```

```json
{
  "customer_id": "CUST_0002",
  "probability": 0.513,
  "prediction": 1,
  "scaled": true
}
```

### Run all tests at once (requires container to be running)

```bash
make test
```

---

## Data Analysis & Scaling

### Feature engineering

`prepare_data.py` aggregates raw transactions into one row per customer with the following features:

| Feature | Type | Description |
|---|---|---|
| `txn_count` | float | Number of transactions |
| `total_debit` | float | Sum of all outflows (≤ 0) |
| `total_credit` | float | Sum of all inflows (≥ 0) |
| `avg_amount` | float | Mean transaction amount |
| `kw_rent` | 0/1 | Rent payment seen in descriptions |
| `kw_netflix` | 0/1 | Streaming subscription seen |
| `kw_tesco` | 0/1 | Grocery spend seen |
| `kw_payroll` | 0/1 | Salary/payroll credit seen |
| `kw_bonus` | 0/1 | Bonus payment seen |

Three additional features are included in `training_set.csv` for future model iterations:
`net_cashflow`, `debit_to_credit_ratio`, `max_single_debit` and `has_large_single_debit`.

### The scaling problem — what the model coefficients revealed

Inspecting the model's learned coefficients exposed a significant issue:

```
Feature              Coefficient
---------------------------------
total_debit          -0.027162    dominates
total_credit         -0.016005    dominates
avg_amount           -0.007365
txn_count             0.000024
kw_rent               0.000032    near zero
kw_netflix           -0.000017    near zero
kw_tesco             -0.000000    near zero
kw_payroll           -0.000032    near zero
kw_bonus             -0.000000    near zero
```

The float features (`total_debit`, `total_credit`) are in units of hundreds to thousands.
The keyword flags are 0 or 1. Without scaling, a single unit change in `total_credit`
contributes `-0.016` to the log-odds, while `kw_rent = 1` contributes only `+0.000032`.
At `total_credit = 2500`, the credit term alone contributes `-40` to the log-odds — a
number so large it pushes the sigmoid to ~0% for almost everyone regardless of any other feature.

The model was effectively a `total_credit` threshold rather than a multi-feature classifier.
The full analysis with step-by-step equation breakdowns is in `logistic_regression_explorer.ipynb`.

### Fix: StandardScaler on float features

`prepare_data.py` fits a `StandardScaler` on the four float columns and saves it to
`artifacts/scaler.joblib`. At inference time `POST /predict/scaled` applies
`x_scaled = (x − μ) / σ` to each float feature before passing to the model.
The keyword flags are passed through unchanged.

Scaler parameters fitted on the training data:

| Feature | Mean (μ) | Std (σ) |
|---|---|---|
| `txn_count` | 2.000000 | 0.632456 |
| `total_debit` | -283.196000 | 234.884435 |
| `total_credit` | 2580.000000 | 2035.092136 |
| `avg_amount` | 1717.268000 | 2398.284832 |

After scaling every feature is on the same mean=0, std=1 scale, so the model can
weigh all features fairly.

### Confusion matrix comparison

Tested against all 5 prototype customers (2 actual defaults, 3 non-defaults):

**Without scaling — `POST /predict`**

```
                 Predicted: 0    Predicted: 1
Actual: 0              3               0
Actual: 1              2               0
```

- Predicted **no defaults at all** — every customer scored ~0% probability
- Correctly classified 3 non-defaults, missed both defaults entirely
- Accuracy: 60% — achieved by predicting the majority class every time
- True positive rate (recall on defaults): **0%**

**With scaling — `POST /predict/scaled`**

```
                 Predicted: 0    Predicted: 1
Actual: 0              2               1
Actual: 1              0               2
```

- Correctly identified **both defaults** — 0 missed defaults
- 1 false positive (CUST_0003 flagged as default, actually fine)
- Accuracy: 80%
- True positive rate (recall on defaults): **100%**

In credit risk, missing a default (false negative) is far more costly than a false positive.
The scaled model's perfect recall on defaults is the meaningful improvement here.

---

## Part 3 — Documentation

### 1. Most challenging part

The most challenging part was understanding the model's feature coefficients and how
their interaction with unscaled inputs caused the model to behave like a simple threshold
rather than a genuine multi-feature classifier.

The coefficient table made this concrete: `total_credit` at 2500 produced a log-odds
contribution of −40, which is so extreme that the sigmoid returned ~0% for virtually
every customer. The keyword flags — which should be meaningful behavioural signals —
had coefficients four orders of magnitude smaller and were completely drowned out.

Diagnosing this required going beyond just calling `model.predict_proba()` and instead
manually reconstructing the logistic regression equation term by term for each customer.
That analysis is documented step by step in `logistic_regression_explorer.ipynb`, which
shows both the broken unscaled behaviour and the corrected scaled predictions side by side.

A secondary challenge was the limited dataset. With only 5 customers and a 3-day
observation window, time-based features that would normally be among the strongest
signals in credit risk could not be meaningfully computed. Features worth implementing
once a larger dataset is available include:

- **`days_since_last_credit`** — long gaps suggest income disruption
- **`credit_trend`** — comparing income in the first vs second half of the observation window to detect declining earnings
- **`salary_regularity`** — standard deviation of the day-of-month salary arrives; high variance suggests gig work or payment issues
- **`spend_acceleration`** — ratio of spend in the last 7 days vs the prior 7 days; rising spend into month-end signals cash flow stress

### 2. Tradeoffs

The main focus was on producing a simple, interpretable, and trustworthy model rather
than chasing accuracy through complexity. In credit decisioning, a model that is
explainable matters as much as one that is accurate — lenders need to justify decisions
to customers and regulators.

When the provided `model.joblib` was producing near-zero probabilities for every customer,
the temptation would have been to retrain with a more complex algorithm. Instead, the
decision was to diagnose the root cause first. That led to the coefficient analysis,
which identified that the issue was not the algorithm but the feature scale a much
simpler fix that preserves the existing model and adds a principled preprocessing step.

The result is clear from the confusion matrix: the scaled model correctly catches
both defaults with one false positive, whereas the unscaled model produces all zeros
and catches nothing. Simplicity won — adding a scaler rather than retraining.

### 3. Running in production on Azure (£500/month, <100ms, 1000 req/hour)

#### Traffic and latency reality check

1000 predictions/hour is 0.28 requests/second. The model is a logistic regression —
inference takes under 1ms. The <100ms budget is almost entirely network round-trip.
The priority at this stage is **reliability, observability, and a clean retraining
loop** — not raw scale.

#### Recommended architecture

```
GitHub repo
    └── GitHub Actions (on merge to main)
          ├── run prepare_data.py
          │     └── pushes model.joblib + model_scaler_params.json  Azure Blob Storage (versioned)
          ├── docker build (pulls latest artifacts from Blob) + push  ACR
          └── az containerapp update --image <new-digest>
                    │
                    ▼
          Azure Container Apps
          (auto-scale 0 → N replicas based on HTTP queue depth)
          ├── GET  /health
          ├── POST /predict
          └── POST /predict/scaled
                    │
                    ▼
          Azure Application Insights
          (latency · error rate · prediction distribution · feature drift)
```

The scaler is not baked into the Docker image in production. It lives in
**Azure Blob Storage**, versioned and decoupled from the application code:

```
az-storage/
  └── credit-models/
        ├── model_scaler_params_v1.json      human-readable, auditable
        ├── model_v1.joblib           loaded by the API at startup
        ├── model_v2.joblib           after retraining on new data
```

**The reason why Blob Storage rather than baking into the image:**
When the training dataset changes (new customers, new date range), `prepare_data.py`
refits the scaler on the new data. That new `model_scaler.joblib` needs to be deployed
independently of the application code — a data change should not require a full
image rebuild and redeployment. Blob Storage allows the artifact to be updated,
versioned, and rolled back independently.

**Version control for training data — DVC**

DVC (Data Version Control) also needs to be added to the MLOPs pipeline. The training data (`transactions.csv`,
`labels.csv`) and the generated artifacts (`model_scaler.joblib`, `training_set.csv`) change
together if we retrain on a new dataset, we need to be able to reproduce exactly
which data produced which model_scaler. DVC tracks this:

```bash
dvc add data/transactions.csv data/labels.csv
dvc push   # pushes data to Azure Blob Storage remote
git commit -m "retrain on Feb 2025 dataset"
# the .dvc files committed to git point to the exact data version
```

This means every model and scaler artifact in Blob Storage can be traced back to the
exact dataset that produced it — important for audit trails in lending.

**Startup behaviour:**
On container startup, `app.py` downloads `model_scaler.joblib` and `model.joblib` from Blob
Storage. A readiness probe on `/health` ensures traffic is not routed until both
artifacts are loaded. Cold start adds ~1–2 seconds (blob download) but does not
affect the <100ms latency SLA for warm containers.

#### Azure service choices and cost breakdown

| Service | Purpose | Est. cost/month |
|---|---|---|
| Azure Container Apps (consumption) | Runs the API; scales 0→N on HTTP queue depth | ~15–40 |
| Azure Container Registry (Basic) | Stores versioned Docker images | ~4 |
| Azure Blob Storage (LRS) | Stores model + scaler artifacts, training data | ~2 |
| Azure Application Insights | Latency, errors, prediction drift monitoring | ~10–20 |
| GitHub Actions | CI/CD (2000 free minutes/month) | 0 |
| **Total** | | **~31–66 pounds/month** |

Well within 500 pounds, leaving headroom for a staging slot, Azure API Management for
auth/rate-limiting, or traffic growth.

**Usage of Container Apps over AKS:**
Container Apps consumption plan bills per vCPU-second — at 0.28 req/sec the
idle cost is near zero where as AKS is quite costly unless multiple services are being run with complex requirements.
For a single inference API it is overhead with no benefit.

**Scaling considerations**

At 0.28 req/sec a single replica handles all traffic with headroom. The container
scales to zero during idle periods (nights, weekends) and back up on the first
request. Cold start from zero is ~3–5 seconds (image pull + blob download).

**Trade-off:** 
In case <100ms SLA is quite important in scaling we can keep atleast 1 replica up and warm which might cost around 15 more to the cost.
At 1000 req/hour traffic is unlikely to be bursty enough

#### CI/CD pipeline

Every merge to `main` triggers a full pipeline:

```
1. Run prepare_data.py
      → fits scaler on latest training data
      → uploads model_scaler_vN.joblib + model_scaler_params_vN.json to Blob Storage
      → tags artifact with git SHA for traceability

2. Docker build
      → Dockerfile pulls scaler model and base model from Blob Storage at build time
      → image tagged with git SHA

3. Push to Azure Container Registry

4. Deploy to Container Apps
      → az containerapp update --image acr.azurecr.io/credit-risk-api:<sha>
      → rolling update: new replica healthy before old one terminates
      → /health probe gates traffic until model + scaler confirmed loaded

5. On failure → auto-rollback to previous image digest
```

Every deployment is fully reproducible: the git SHA links to the code, the DVC
hash links to the training data, and the Blob Storage version links to the artifacts.

#### Model versioning and retraining trigger

The model is not retrained on a schedule — it is retrained when performance degrades.
The trigger is the **prediction distribution drift metric** in Application Insights:

```
If 7-day rolling mean(probability) shifts > 0.1 from baseline
    → alert fires
    → data team reviews
    → if confirmed drift: run prepare_data.py on new data, retrain, deploy
```

When a new model is trained:
1. New `model_vN.joblib` and `model_scaler_vN.joblib` pushed to Blob Storage
2. Previous versions retained (never deleted) — full rollback available
3. CI/CD pipeline deploys new image
4. Prediction distribution monitored for 48 hours post-deploy to confirm improvement

#### Monitoring and alerting

| Metric | Alert threshold | Why |
|---|---|---|
| p99 request latency | > 80ms | Early warning before SLA breach |
| HTTP 5xx error rate | > 1% over 5 min | Model or scaler failed to load |
| HTTP 4xx rate | > 10% over 5 min | Upstream schema change breaking validation |
| Mean prediction probability (7-day rolling) | Drift > 0.1 from baseline | Population shift — model may no longer be valid |
| Feature null rate (`kw_*` flags all zero) | > 50% of requests | Text processing pipeline broken upstream |
| Container replica count | > 3 sustained | Unexpected traffic spike |

All alerts route to the on-call channel. The prediction distribution alert is the
most important — it fires before model accuracy visibly degrades and gives the data
team time to investigate before the lender notices.

#### What to improve first — priority order

1. **API authentication** — the endpoints are currently open. Azure API Management
   in front of Container Apps adds key-based auth and rate limiting.
2. **Refit scaler on real training data** — the current scaler is fitted on 5 prototype
   rows; its mean/std values are not representative of a real customer population.
3. **Application Insights integration** — without the prediction distribution metric
   we are blind to model drift in production.
4. **DVC for training data** — ensures every deployed scaler can be traced to the
   exact dataset that produced it, which is a regulatory requirement in lending.
5. **Set minReplicas: 1** — if the <100ms cold-start breach is unacceptable to the
   lender's platform team.

### 4. How would we deploy the FastAPI service and make the model artifact available?
The deployment model is a two-layer separation: the **application image** is managed
by CI/CD, and the **model artifacts** are managed independently in Azure Blob Storage.
They are versioned separately because they change for different reasons — code changes
trigger an image rebuild, but a new training dataset should be able to update the
scaler without touching the application code.

The final image contains only `app.py`, the installed packages, and the compiled
artifacts — no raw data, no training code, no matplotlib. Keeps the image small
and the attack surface minimal as its in the `DOCKERFILE`

#### Making model artifacts available

Artifacts flow through three stages:
1. Generate (CI pipeline) where it creates the model files or retrained model params and uploads to Blob
2. Package (Docker build) Dockerfile copies artifacts from Blob Storage into the image
3. Load (runtime) app.py loads model and scaler from /app/artifacts/ on startup

If artifacts need to be updated independently of a code deployment (e.g. emergency
scaler refit on new data), the pipeline can push new artifacts to Blob Storage and
trigger a Container Apps revision without a code change. This is the main reason
artifacts live outside the git repository.

#### Rollback

Every deployed image is tagged with its git SHA and retained in ACR indefinitely.


### 5. If transaction volume jumped from thousands to millions per day, how would we rethink Part 1?

The current `prepare_data.py` is a single pandas script that loads everything into
memory. At millions of transactions per day that breaks in two ways: the machine
runs out of RAM, and a nightly full reprocess becomes too slow to be useful.
The pipeline needs to be refactored at every stage.

#### Storage layer — move from CSV to Parquet

```
Current:   data/transactions.csv  (local, full reload each run)
           data/labels.csv

At scale:  Azure Data Lake Storage Gen2
           transactions/
             year=2025/month=02/day=01/transactions.parquet
             year=2025/month=02/day=02/transactions.parquet
           labels/
             labels.parquet
```

Parquet over CSV because: columnar storage means we only read the columns we
need; Snappy compression reduces storage cost by ~5–10x; partition pruning means
a daily job reads only that day's partition, not the full history.

#### Processing layer — incremental over full reprocess

The most important change. Instead of reaggregating every customer from scratch
every night:

```
Current:  read all transactions → aggregate all customers → save training_set.csv
          (cost scales with total history size)

At scale: read only NEW transactions since last run
          → update feature store for affected customers only
          → cost scales with daily volume, not total history
```

This requires a feature store — a database that holds the latest aggregate state
per customer and can be updated incrementally:

```
Azure Synapse or Databricks Delta Lake
  customer_features table
  ├── customer_id   
  ├── txn_count     
  ├── total_debit   
  ├── total_credit  
  ├── avg_amount    
  ├── kw_*          
  └── updated_at    
```
One important point regarding txn_count, total_debit, total_credit etc
they might be running total and we need to handle the time frame, running total of last 30 days, 60 days 90 days etc
and also for flags we need to check if its accumulated one or some different feature

New transactions arriving daily update only the affected customer rows.
The API reads features from the feature store at prediction time — no
batch aggregation step needed at inference.

#### Compute layer — Spark over pandas

```python
# Current (pandas) — breaks at scale
df = pd.read_csv('data/transactions.csv')
agg = df.groupby('customer_id').agg(...)

# At scale (PySpark on Azure Databricks or Synapse Spark)
df = spark.read.parquet('data_path')
agg = df.groupBy('customer_id').agg(
    count('transaction_id').alias('txn_count'),
    sum(when(col('amount') < 0, col('amount'))).alias('total_debit'),
    sum(when(col('amount') > 0, col('amount'))).alias('total_credit'),
    mean('amount').alias('avg_amount'),
)
```
Obviously at scale the data preperation, the feature extraction everything will be refactored
so that its fast and also doesn't put a lot of load into the system or run out of memory

#### Text processing layer — broadcast keyword matching

The current regex keyword matching runs row-by-row in Python. At millions of
rows this becomes the bottleneck. At scale, a merchant category
lookup table and join rather than scanning descriptions at runtime will be better

```
merchant_categories table (maintained separately)
├── merchant_name_pattern   TESCO.*
├── category                groceries
└── kw_flag                 kw_tesco

transactions JOIN merchant_categories ON regexp_match
→ category flags added as a column, no Python loop
```

This also makes the keyword list maintainable without
touching the pipeline code.

#### Data quality layer — gates not warnings

At prototype scale, data quality issues are logged and noted. At millions of
rows per day, a bad batch can silently corrupt features for thousands of customers
before anyone notices. Quality checks become hard gates:

```python
# Current: print and continue
print(f"  Amount/type mismatches: {len(mismatched)}")

# At scale: fail the pipeline, alert, and stop
from great_expectations import DataContext
result = context.run_checkpoint('transactions_checkpoint')
if not result['success']:
    raise ValueError(f"Data quality gate failed: {result}")
    # → pipeline halts, alert fires, no corrupt features reach the model
```

#### Scaler refit — on sample, not full dataset

At millions of customers, fitting `StandardScaler` on the full training set
becomes expensive. The scaler only needs a representative sample to estimate
mean and std accurately. At scale, refit on a stratified sample of 50–100k
customers rather than the full population.

#### Summary of changes at scale

| Layer | Now | At millions/day |
|---|---|---|
| Storage | Local CSV | Azure Data Lake (Parquet, partitioned) |
| Processing | pandas full reload | PySpark incremental on Databricks |
| Feature store | CSV file | Delta Lake table, updated daily |
| Text processing | Python regex loop | Pre-computed merchant lookup + join |
| Data quality | Print warnings | Great Expectations hard gates |
| Orchestration | Manual script run | Azure Data Factory or Databricks Workflows |
| Scaler refit | Full dataset | Stratified sample of 50–100k customers |

### 6. What metrics would we should track in production and why? What could go wrong?

#### Metrics to track

**Infrastructure metrics** — is the service running correctly?

| Metric | Threshold | Why |
|---|---|---|
| p50 / p95 / p99 request latency | p99 > 80ms → alert | Early warning before SLA breach; p99 catches tail latency spikes that p50 hides |
| HTTP 5xx error rate | > 1% over 5 min | Model or scaler failed to load; upstream dependency broken |
| HTTP 4xx error rate | > 10% over 5 min | Upstream schema change — field renamed, type changed, sign convention flipped |
| Container CPU / memory | > 80% sustained | Approaching resource limit; scale before it becomes an incident |

**Model health metrics** — is the model still making sensible predictions?

| Metric | Threshold | Why |
|---|---|---|
| Mean prediction probability (7-day rolling) | Drift > 0.1 from baseline | The single most important signal — if average score shifts, the model has encountered a different population than it was trained on |
| Prediction distribution (histogram) | Shape change from baseline | Catches bimodal collapse — model outputting all 0s or all 1s — which the mean alone can miss |
| Feature null / missing rate per field | > 5% on any feature | Upstream data pipeline failure; missing features silently default to 0 and corrupt predictions |
| `kw_*` flags all-zero rate | > 60% of requests | Text processing broken; all merchant category information lost |

**Business metrics** — is the model actually working? (lagged ~90 days)

| Metric | Why |
|---|---|
| Default rate of approved customers | Ground truth — the only metric that tells us if the model is predictive in the real world |
| False negative rate on defaults | Missed defaults are the most costly outcome for the lender |
| False positive rate on non-defaults | Customers incorrectly declined; regulatory and reputational risk |
| Score distribution by acquisition channel | If a new marketing channel brings different customers, their score distribution will differ — we need to catch this before it dilutes model performance |

#### What could go wrong

**1. Population drift**
The model is trained on one cohort. If the lender changes their acquisition
channel, targets a new demographic, or launches a new product, the incoming
customers may look nothing like the training data. The model will still produce
a number — it just won't mean anything. The prediction distribution metric
catches this before accuracy visibly degrades.

**2. The feedback loop (survivorship bias)**
The most structurally dangerous failure mode. If the model rejects high-risk
customers, we never observe whether they would have defaulted. Future training
data only contains approved (lower-risk) customers. The model learns from a
progressively biased sample and systematically underestimates risk over time.
We need to mitigate with a small random approval holdout (~2–5%) to maintain an unbiased
sample of the full risk spectrum.

**3. Silent feature corruption**
If the upstream transaction schema changes — `txn_type` values change from
`debit/credit` to `DR/CR`, a merchant renames, amounts start arriving unsigned
— the keyword flags silently drop to zero for all customers. No exception is
raised, predictions just become wrong. The `kw_*` all-zero rate alert catches
this. Hard data quality gates in the pipeline (Q5) prevent corrupt features
from reaching the model in the first place.

**4. Label delay blindspot**
Default labels only resolve 90 days after origination. For the first 90 days
after a model deployment we cannot measure whether the new model is better
or worse than the old one — we are flying blind. We need to mitigate by tracking leading
indicators (30-day missed payment, overdraft usage, sudden drop in credit
inflows) as proxy labels that resolve faster.

**5. The unscaled route in production**
`POST /predict` (no scaling) is retained in the API for comparison and debugging.
If a caller mistakenly uses the raw route in production — expecting the scaled
output — they will receive near-zero probabilities for all customers and approve
everyone. This is a silent failure. In production the unscaled route should be
either removed, renamed to `/predict/debug`, or gated behind an internal-only
header so it cannot be called by the lender's platform accidentally.

**6. Regulatory risk on unexplainability**
Logistic regression is inherently explainable — each coefficient has a direct
interpretation. However, if a customer is declined credit, the lender must be
able to state which features drove that decision. The coefficient table and the
`predict_manual()` function in the notebook provide this, but they need to be
surfaced in the API response for production use. Adding a `feature_contributions`
field to the response (each feature's `coefficient × scaled_value`) would make
the model auditable per-prediction without changing the underlying logic.


### 7. AI tools used

Claude was used in specific, bounded ways throughout this exercise.

**Documentation and README**
The bulk of Claude's contribution was in this README. Given the volume of ground
to cover — setup instructions and six technical questions based on my own
architectural decisions, tradeoff reasoning, and technical positions Claude was used to help
structure sections clearly, write tables and code snippets using Readme syntax, and make them more readable. 
The Azure cost estimates were cross-referenced with Claude as a sanity check against general public pricing
knowledge, not taken from live Azure pricing pages

**Syntax**
Based on all the logic decisions, the `MODEL_FEATURES` order, the `_build_feature_vector`
split between float and flag columns, the 503 vs 500 distinction claude was used to update the draft syntaxes.
Some verbose lines in app.py and prepare.py is still kept to give idea about the decisions.

**Debugging**
A lot of the print statement was added using claude to speed up code and check the code flow

**What Claude did not do**
Claude did not design the features, identify the scaling problem, choose the
approach for scaling floats while passing flags through, write
the `logistic_regression_explorer.ipynb` analysis, or make any of the production
architecture decisions. The core insight of the exercise — that the model was
behaving as a threshold because unscaled float magnitudes were drowning out the
keyword flags — came from manually reading the coefficient table and tracing the
equation term by term. That analysis is documented in the notebook.

**On using AI tools in general**
The main risk with AI-assisted code is that it produces plausible-looking solutions
to the wrong problem. The scaling issue is a good example: Claude could have
suggested adding a scaler without explaining why the model was broken in the first
place. Understanding the root cause — coefficient magnitudes, log-odds arithmetic,
sigmoid saturation — required working through the problem independently first.