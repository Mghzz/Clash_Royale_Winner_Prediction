# Clash Royale Winner Prediction

This project predicts the winner of a Clash Royale match. We treat winner
prediction as a comparison between the two players, so each row is one match with
both players' features and the target `is_A_winner`. The project has two parts:
data engineering (a PostgreSQL database, a scripted import, and a preparation
pipeline: cleaning, feature engineering, model-ready output) and modeling (model
selection, training with hyperparameter tuning and MLflow tracking, evaluation,
and a prediction script).

## Pipeline overview

The database is the source of truth. Each stage reads its input from the database
and writes its output back, so the stages are independent.

```
CSV (raw matches)
  -> scripts/load_data.py            stage 1: load into relational tables
  -> Match, MatchPlayer, MatchPlayerCard, Card, Clan
  -> scripts/preprocess.py           stage 2: clean, impute, encode
  -> table preprocessed_match        (+ artifacts/preprocessed_data.csv)
  -> scripts/feature_engineering.py  stage 3: A/B sides, comparison features, scale
  -> table model_ready_match         (+ artifacts/model_ready_data.csv, scaler.pkl)
```

The database stays normalized (two `MatchPlayer` rows per match). The
one-row-per-match shape is built in the pipeline with a query, not stored that way.

Stage 2 reads the match rows with `database_connection.build_match_base_query()`,
which self-joins the winner and loser into `w_*` and `l_*` feature blocks. It fills
missing values, drops fully-empty columns (such as `tournamentTag`), frequency-
encodes clan tags, adds time features, and writes `preprocessed_match`.

Stage 3 reads `preprocessed_match`, randomly relabels the winner/loser sides as
`A`/`B` (seeded, so the winner is not always the same column), builds comparison
features (`diff_*`, `ratio_*`) and the target `is_A_winner`, drops near-constant
and highly correlated columns, scales the numeric columns, and writes
`model_ready_match`. The fitted scaler is saved to `artifacts/scaler.pkl`.

Note on leakage: `crowns`, `kingTowerHitPoints`, `princessTowersHitPoints` and
`trophyChange` are recorded after the match. They are kept in `model_ready_match`
for completeness, but `scripts/train_model.py` drops them (and their `A_*`/`B_*`/
`diff_*`/`ratio_*` variants) before training, since a real pre-match model must not
see them. The EDA already leaves them out too.

```
table model_ready_match
  -> scripts/train_model.py          stage 4: train, tune, evaluate, track in MLflow
  -> artifacts/best_model.joblib (+ MLflow run + registered model)
  -> scripts/make_predictions.py     stage 5: score new/held-out matches
  -> table match_predictions
```

Stage 4 loads `model_ready_match`, drops the leakage columns above, and splits the
data into train/validation/test (70/15/15, stratified on `is_A_winner`). It trains
four candidate classifiers — Logistic Regression, Random Forest, XGBoost, and a
neural network (`MLPClassifier`) — each with `GridSearchCV` (5-fold cross-validation
plus a small regularization/hyperparameter grid). Every candidate run (params,
cross-validation score, validation metrics, the fitted model) is logged to MLflow.
The candidate with the best validation ROC-AUC is picked automatically — the four
models, including the neural network, are compared on equal footing, so whichever
genuinely performs best wins. That model is evaluated once on the held-out test set
(accuracy, precision, recall, F1, ROC-AUC, confusion matrix), all logged to the same
MLflow run, then registered in the MLflow Model Registry and saved to
`artifacts/best_model.joblib` (model + feature column list) for reuse without
retraining.

Stage 5 loads `preprocessed_match`, rebuilds the same `A_*`/`B_*`/`diff_*`/`ratio_*`
features with `scripts/feature_engineering.add_features()`, applies the **saved**
scaler (`artifacts/scaler.pkl`, `.transform()` only, never refit) and the **saved**
model (`artifacts/best_model.joblib`, no retraining), and writes `match_id`,
`predicted_label`, `predicted_probability`, and (when available) `actual_is_A_winner`
to the `match_predictions` table.

## Project structure

```
Clash_Royale_Winner_Prediction/
├── schema.sql                      DB schema (single source of truth)
├── schema.png                      ER diagram
├── pipeline.py                     runs the data-engineering stages (1-3)
├── run_pipeline.py                 runs the full pipeline (1-5), end to end
├── requirements.txt
├── Dockerfile                      container for the pipeline
├── docker-compose.yml              pipeline + Postgres for local testing
├── .dockerignore
├── eda.ipynb                       exploratory data analysis
├── data_understanding.ipynb        first look at the raw CSV
├── .github/workflows/ci.yml        GitHub Actions CI
├── scripts/
│   ├── database_connection.py      engine, schema, read/write helpers, base query
│   ├── load_data.py                stage 1
│   ├── preprocess.py               stage 2
│   ├── feature_engineering.py      stage 3
│   ├── train_model.py              stage 4: model selection, training, MLflow
│   └── make_predictions.py         stage 5: score matches, write match_predictions
├── sample/
│   └── ci_sample.csv               small sample used by CI and Docker
├── mlruns/ , mlflow.db             MLflow tracking store (gitignored)
└── artifacts/                      generated CSV / scaler / model (gitignored)
```

## Requirements

- Python 3.10+
- A running PostgreSQL server you can write to
- `pip install -r requirements.txt` also installs `mlflow` and `xgboost`, used by
  the modeling stages

## Setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1        # Windows PowerShell
   # source .venv/bin/activate       # Linux / macOS
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill in your database credentials:

   ```
   DB_HOST=127.0.0.1
   DB_PORT=5432
   DB_NAME=clash_royale
   DB_USER=postgres
   DB_PASSWORD=your_password_here
   ```

The schema is created automatically from `schema.sql` the first time the loader
runs, so you do not need to apply it by hand. To apply it manually:
`psql -d clash_royale -f schema.sql`.

## Running the pipeline

Run from the project root with the virtual environment active.

Run everything:

```powershell
python pipeline.py --csv-path "sample/sample.csv"
```

Useful flags: `--limit-rows N` (load only the first N rows), `--chunk-size N`
(rows per insert batch, default 5000), `--no-reset` (append instead of truncating).

Or run the stages one at a time:

```powershell
python -m scripts.load_data --csv-path "sample/sample.csv"
python -m scripts.preprocess
python -m scripts.feature_engineering
python -m scripts.train_model
python -m scripts.make_predictions
```

Stages 2-5 read from the database, so you can re-run any of them on their own once
the stages before them have populated their input table. Stages 2, 3, and 5 accept
`--input-path file.csv` to read a CSV instead of the database. Stage 3 also has
`--corr-threshold` (default 0.95), `--scaler-path`, and `--seed` (default 42,
controls the A/B side assignment). Stage 4 has `--seed` (controls the train/val/test
split). Stage 5 has `--model-path` and `--scaler-path` (defaults point at the
`artifacts/` files stage 4 and stage 3 produce).

To run everything, including training and predictions, in one command:

```powershell
python run_pipeline.py --csv-path "sample/sample.csv"
```

It accepts the same flags as `pipeline.py` (`--limit-rows`, `--chunk-size`,
`--no-reset`, `--seed`, etc.) and simply chains all five stages.

After training, inspect the experiment runs with the MLflow UI:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Outputs

| Output | Location | Description |
| --- | --- | --- |
| `preprocessed_match` | PostgreSQL table | cleaned, one row per match (`w_*` / `l_*`) |
| `model_ready_match`  | PostgreSQL table | feature matrix (`A_*`/`B_*`/`diff_*`/`ratio_*`, target `is_A_winner`) |
| `match_predictions`  | PostgreSQL table | `match_id`, `predicted_label`, `predicted_probability`, `actual_is_A_winner` |
| `preprocessed_data.csv` | `artifacts/` | CSV copy of the preprocessed table |
| `model_ready_data.csv`  | `artifacts/` | CSV copy of the model-ready table |
| `scaler.pkl` | `artifacts/` | fitted scaler + column list, reused (not refit) at prediction time |
| `best_model.joblib` | `artifacts/` | best model + its feature column list, reused (not retrained) at prediction time |
| `confusion_matrix.png` | `artifacts/` | confusion matrix of the best model on the test set |
| MLflow runs | `mlruns/`, `mlflow.db` | one run per candidate model: params, CV score, validation/test metrics, model artifact |
| `clash_royale_winner_predictor` | MLflow Model Registry | registered best model |

## Exploratory data analysis

`eda.ipynb` covers the dataset overview, missing values, the modeling target,
winner-vs-loser pre-match distributions and gaps, a "does the higher value win?"
analysis, outliers, the correlation of the gaps, time patterns, and card usage. It
reads `preprocessed_match` (or `artifacts/preprocessed_data.csv` if the database is
not reachable), so run the pipeline first. Post-match columns are left out as
leakage.

`data_understanding.ipynb` is the earlier raw-CSV check (column list, row count,
chunk shapes).

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request to `main`. It starts
a Postgres service, installs the dependencies, compile-checks the source, and runs
the full pipeline on `sample/ci_sample.csv` (a small 3000-row sample committed for
this purpose). If the pipeline finishes, the check passes.

To turn it on, push the repository to GitHub. The workflow runs by itself; nothing
else is required because it uses its own throwaway Postgres service and the small
sample, not your real database. You can watch the runs under the repository's
Actions tab.

## Docker

`Dockerfile` builds an image with the pipeline and its dependencies.
`docker-compose.yml` runs that image together with a Postgres container so you can
test the whole thing locally.

Test with Compose (recommended):

```bash
docker compose up --build
```

This starts Postgres, waits for it to be ready, then runs the pipeline on
`sample/ci_sample.csv`. The pipeline container exits when it finishes. Stop and
clean up with:

```bash
docker compose down -v
```

To run the full sample instead, mount it and override the command:

```bash
docker compose run --rm \
  -v "$(pwd)/sample/sample.csv:/app/sample/sample.csv" \
  pipeline python pipeline.py --csv-path sample/sample.csv
```

Build the image on its own (without Compose):

```bash
docker build -t clash-pipeline .
```

Running that image directly needs a reachable Postgres and the `DB_*` environment
variables, which is why Compose is the simpler way to test.
