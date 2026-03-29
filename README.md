# Multimodal Stock Prediction: NSE Equities

Next-day stock direction prediction on NSE (National Stock Exchange of India) large-cap equities using three data modalities: numeric OHLCV indicators, Chart2Tokens event features, and FinBERT-derived filing sentiment. Evaluated under walk-forward validation with purge/embargo to prevent look-ahead bias.

## Research Questions

1. Does adding Chart2Tokens (discrete chart events) to numeric features improve prediction?
2. Do first-seen-aligned corporate filing features add value beyond time-series features?
3. Does per-fold Platt calibration improve probability quality?
4. How sensitive are Chart2Token features to the summary window (W) and decay half-life (h)?

## Project Structure

Important Note: Datasets and cache are not included in this repo due to size contraints in GitHub.

```
stock-prediction-cmf/
|
|-- configs/
|   |-- default.yaml              # All configurable parameters
|
|-- pipeline/                      # Main pipeline (run in order)
|   |-- _common.py                 # Label creation (next-day direction)
|   |-- step1_data_assembly.py     # Load OHLCV, filings, run FinBERT
|   |-- step2_feature_engineering.py  # Numeric + Chart2Tokens features
|   |-- step3_filings_modality.py  # Text sentiment pooling + memory features
|   |-- step4_modeling_evaluation.py  # Walk-forward train/calibrate/evaluate
|
|-- src/                           # Core library modules
|   |-- data/
|   |   |-- ohlcv_loader.py        # Load minute CSVs or cached daily parquet
|   |   |-- filings_loader.py      # Filing loading + classification
|   |   |-- nse_calendar.py        # NSE trading calendar, first-seen rules
|   |-- features/
|   |   |-- numeric.py             # 6 OHLCV indicators (leak-safe)
|   |   |-- chart2tokens.py        # Event detectors + milestone attraction
|   |   |-- text_sentiment.py      # Frozen FinBERT inference + daily pooling
|   |   |-- intraday_reaction.py   # First-hour reaction features
|   |-- splits/
|   |   |-- walk_forward.py        # Walk-forward splitter with purge/embargo
|   |-- models/
|   |   |-- training.py            # LightGBM and Logistic Regression training
|   |   |-- calibration.py         # Platt scaling, temperature scaling
|   |   |-- fusion.py              # Weighted average and meta-LR late fusion
|   |-- evaluation/
|   |   |-- metrics.py             # ROC-AUC, PR-AUC, F1, Brier, ECE
|   |   |-- plots.py               # Reliability curves, metric comparisons
|   |-- utils/
|       |-- config.py              # YAML config loader
|       |-- reproducibility.py     # Seeding, environment logging
|
|-- scripts/                       # Experiment runners
|   |-- run_sensitivity_grid.py    # W x h sensitivity analysis (RQ4)
|   |-- run_concat_crossmodal.py   # Cross-modal fusion experiments
|   |-- run_feature_pruning.py     # 28-strategy feature pruning analysis
|   |-- run_milestone_attraction.py  # Milestone attraction feature evaluation
|   |-- run_v2_validation.py       # Chart2Tokens v2 vs v1 comparison
|   |-- exploratory/
|       |-- run_per_stock.py       # Per-stock walk-forward analysis
|       |-- run_ablation.py        # Text feature ablation (T1-T5 tiers)
|       |-- run_volatility_prediction.py  # Alternative target: next-day volatility
|       |-- run_category_cross_modal.py   # M4 category-aware fusion
|       |-- analyze_filing_profile.py     # Filing profile analysis
|
|-- tests/                         # Unit tests
|   |-- test_chart2tokens.py       # Event detector correctness tests
|   |-- test_first_seen.py         # Filing session assignment tests
|   |-- test_purge_embargo.py      # Walk-forward leak-safety tests
|
|-- cache/                         # Cached intermediate data (parquet files)
|-- outputs/                       # Experiment results (CSVs + plots)
|-- requirements.txt               # Python dependencies
```

## Setup

### Prerequisites

- Python 3.10+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

## Data

### Raw data (not included in repo)

- `price_data/`: Minute-level OHLCV CSVs for 98 NSE stocks (4.8 GB). Source: Kaggle "Stock Market Data - Nifty 100 Stocks (1 min)"
- `filings_data/`: Corporate governance filings from NSE India, organized by date

### Cached data (included)

Pre-computed intermediate artifacts that allow running experiments without raw data:

| File | Description |
|------|-------------|
| `cache/daily_ohlcv.parquet` | Resampled daily OHLCV (227K rows, 98 symbols) |
| `cache/features_all.parquet` | Complete feature matrix with labels |
| `cache/daily_text_features.parquet` | FinBERT sentiment per symbol/day (25 symbols) |
| `cache/filings_processed.parquet` | Preprocessed filings with sentiment scores |
| `cache/sentiment_cache.parquet` | Raw FinBERT scores per filing chunk |
| `cache/completed_symbols.json` | Tracks which symbols have been processed |

## Running Experiments

### Run tests (verify everything works)

```bash
python -m pytest tests/ -v
```

### Sensitivity analysis

Grid search over Chart2Token parameters W and h:

```bash
python scripts/run_sensitivity_grid.py
```

Output: `outputs/sensitivity_grid.csv`

### Per-stock analysis

Evaluate each stock individually:

```bash
python scripts/exploratory/run_per_stock.py
```

Output: `outputs/per_stock_results.csv`

### Text feature ablation

Test text feature tiers T1 through T5:

```bash
python scripts/exploratory/run_ablation.py
```

Output: `outputs/ablation_pooled.csv`

### Feature pruning analysis

Test 28 pruning strategies across 5 model architectures:

```bash
python scripts/run_feature_pruning.py
```

### Chart2Tokens v2 validation

Compare v2 (pruned events + milestone attraction) against v1:

```bash
python scripts/run_v2_validation.py
```

### Category-aware fusion / M4

```bash
python scripts/exploratory/run_category_cross_modal.py
```

### Volatility prediction

Alternative target: next-day realized volatility:

```bash
python scripts/exploratory/run_volatility_prediction.py
```

### Full pipeline from raw data (optional, ~8 hours for FinBERT)

Only needed if you want to reprocess everything from scratch:

```bash
python pipeline/step1_data_assembly.py
python pipeline/step2_feature_engineering.py
python pipeline/step3_filings_modality.py
python pipeline/step4_modeling_evaluation.py
```

## Three Model Variants

| Variant | Features | Count | Architecture |
|---------|----------|-------|-------------|
| M1: Numeric-only | log_return, atr, rolling_vol, ma_ratio, ma_long_slope, volume_zscore | 6 | Single LightGBM |
| M2: Numeric+Tokens | M1 + Chart2Tokens (breakout, gap, volume burst, round touch, engulfing) | 27 | Single LightGBM |
| M3: TS+Text Fusion | M2 branch + FinBERT text branch, late fusion | up to 62 | 2x LightGBM + weighted avg |

## Walk-Forward Evaluation

- 6-month training window, 2-month validation, 2-month test
- 1-month step (sliding forward)
- 25-day purge + 5-day embargo at train/test boundary
- Platt calibration fit per fold on validation data only
- Scaler fit per fold on training data only

## Key Results

| Variant | ROC-AUC | PR-AUC | Brier |
|---------|---------|--------|-------|
| M1: Numeric-only | 0.520 +/- 0.052 | 0.522 +/- 0.070 | 0.255 |
| M2: Numeric+Tokens | 0.517 +/- 0.040 | 0.507 +/- 0.062 | 0.257 |
| M3: TS+Text Fusion | 0.510 +/- 0.043 | 0.502 +/- 0.059 | 0.257 |

No modality provides significant incremental value over the numeric baseline for next-day direction prediction on liquid NSE large-cap equities under walk-forward evaluation.

## Leak-Safety Rules

1. All rolling features use `shift(1)` -- no current-day data in features
2. Labels defined as next-day direction: `y[t] = sign(close[t+1] - close[t])`
3. StandardScaler fit only on training split per fold
4. Purge (25 days) + embargo (5 days) at train/test boundaries
5. First-seen alignment for filings (post-15:30 IST shifted to next session)
6. Calibration fit per fold on validation data only
7. FinBERT is frozen (no fine-tuning on target data)

## Configuration

All parameters are in `configs/default.yaml`. Key settings:

```yaml
# Walk-forward
train_months: 6
val_months: 2
test_months: 2
purge_days: 25
embargo_days: 5

# LightGBM
n_estimators: 300
max_depth: 4
learning_rate: 0.05

# Chart2Tokens
lookback_W: 20
half_life_h: 5.0

# FinBERT
model: ProsusAI/finbert
max_chunk_tokens: 510
```

## Dependencies

- pandas, numpy -- data manipulation
- scikit-learn -- Logistic Regression, StandardScaler, metrics
- lightgbm -- primary classifier
- transformers, torch -- FinBERT inference
- scipy -- statistical tests
- matplotlib -- plotting
- pyyaml -- config loading
- pyarrow -- parquet I/O
