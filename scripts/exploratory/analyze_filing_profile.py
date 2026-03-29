"""Analyze what makes the 6 hypothesis-winners different from the rest.

Looks at: filing frequency, category mix, body-text availability,
sentiment distribution, and sector patterns.
"""

import argparse, json, os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_parser = argparse.ArgumentParser(description="Filing profile analysis")
_parser.add_argument("--dry-run", action="store_true",
                     help="Run full pipeline but skip writing any files")
_args = _parser.parse_args()
dry_run = _args.dry_run

from src.utils.config import load_config

cfg = load_config("configs/default.yaml")
cache_dir = Path(cfg["data"]["cache_dir"])
output_dir = Path(cfg["data"]["output_dir"])

# Load per-stock results
results = pd.read_csv(output_dir / "per_stock_results.csv")

# Load processed filings (has category + sentiment)
filings = pd.read_parquet(cache_dir / "filings_processed.parquet")
filings["assigned_date"] = pd.to_datetime(filings["assigned_date"])

# Load daily text features for filing density
daily_text = pd.read_parquet(cache_dir / "daily_text_features.parquet")
daily_text["date"] = pd.to_datetime(daily_text["date"])

# Winners vs losers
winners = ["ADANIPORTS", "BOSCHLTD", "CIPLA", "DABUR", "APOLLOHOSP", "ASIANPAINT"]
losers = ["AMBUJACEM", "ADANIGREEN", "ATGL", "BAJAJ-AUTO", "BANKBARODA"]  # both steps hurt
rest = [s for s in results["symbol"].tolist() if s not in winners and s not in losers]

results["group"] = results["symbol"].apply(
    lambda s: "winner" if s in winners else ("loser" if s in losers else "partial")
)

# ============================================================
# 1. Filing frequency & density
# ============================================================
print("=" * 90)
print("1. FILING FREQUENCY & DENSITY")
print("=" * 90)

freq_rows = []
for sym in results["symbol"]:
    sf = filings[filings["symbol"] == sym]
    sd = daily_text[daily_text["symbol"] == sym]
    n_trading_days = len(sd)
    n_filings = len(sf)
    n_days_with_filings = sd[sd["doc_count"] > 0].shape[0] if "doc_count" in sd.columns else 0
    filing_density = n_days_with_filings / n_trading_days if n_trading_days > 0 else 0
    avg_docs_per_filing_day = sf.groupby("assigned_date").size().mean() if len(sf) > 0 else 0

    freq_rows.append({
        "symbol": sym,
        "n_filings": n_filings,
        "n_days_with_filings": n_days_with_filings,
        "filing_density": filing_density,
        "avg_docs_per_day": avg_docs_per_filing_day,
        "filings_per_month": n_filings / (n_trading_days / 21) if n_trading_days > 0 else 0,
    })

freq_df = pd.DataFrame(freq_rows)
freq_df = freq_df.merge(results[["symbol", "group"]], on="symbol")

print(f"\n{'Group':>10s} {'Filings':>8s} {'Days w/':>7s} {'Density':>8s} {'Docs/day':>9s} {'Per month':>10s}")
print("-" * 60)
for grp in ["winner", "partial", "loser"]:
    g = freq_df[freq_df["group"] == grp]
    print(f"{grp:>10s} {g['n_filings'].mean():8.0f} {g['n_days_with_filings'].mean():7.0f} "
          f"{g['filing_density'].mean():8.3f} {g['avg_docs_per_day'].mean():9.2f} "
          f"{g['filings_per_month'].mean():10.1f}")

print("\nPer-stock detail:")
for _, r in freq_df.sort_values("group").iterrows():
    print(f"  {r['symbol']:15s} [{r['group']:>7s}] filings={r['n_filings']:4.0f} "
          f"density={r['filing_density']:.3f} docs/day={r['avg_docs_per_day']:.2f} "
          f"per_month={r['filings_per_month']:.1f}")


# ============================================================
# 2. Filing CATEGORY composition
# ============================================================
print("\n" + "=" * 90)
print("2. FILING CATEGORY COMPOSITION (% of filings per group)")
print("=" * 90)

cat_rows = []
for sym in results["symbol"]:
    sf = filings[filings["symbol"] == sym]
    if sf.empty:
        continue
    cats = sf["filing_category"].value_counts(normalize=True)
    row = {"symbol": sym}
    for cat, pct in cats.items():
        row[cat] = pct
    cat_rows.append(row)

cat_df = pd.DataFrame(cat_rows).fillna(0)
cat_df = cat_df.merge(results[["symbol", "group"]], on="symbol")

# Aggregate by group
cat_cols = [c for c in cat_df.columns if c not in ["symbol", "group"]]
cat_summary = cat_df.groupby("group")[cat_cols].mean()

# Show top categories per group
print("\nTop filing categories by group (mean % of filings):")
for grp in ["winner", "partial", "loser"]:
    row = cat_summary.loc[grp].sort_values(ascending=False)
    top = row[row > 0.01]  # >1%
    print(f"\n  {grp.upper()}:")
    for cat, pct in top.items():
        print(f"    {cat:35s} {pct*100:5.1f}%")

# Key category differences
print("\n\nCategory differences (winner - loser):")
if "winner" in cat_summary.index and "loser" in cat_summary.index:
    diff = cat_summary.loc["winner"] - cat_summary.loc["loser"]
    diff = diff.reindex(diff.abs().sort_values(ascending=False).index)
    for cat, d in diff.items():
        if abs(d) > 0.01:
            print(f"  {cat:35s} {d*100:+5.1f}pp  (winner={cat_summary.loc['winner'][cat]*100:.1f}% vs loser={cat_summary.loc['loser'][cat]*100:.1f}%)")

# ============================================================
# 3. Body text availability (headline-only vs full body)
# ============================================================
print("\n" + "=" * 90)
print("3. BODY TEXT AVAILABILITY")
print("=" * 90)

body_rows = []
for sym in results["symbol"]:
    sf = filings[filings["symbol"] == sym]
    if sf.empty:
        continue
    n = len(sf)
    n_body = sf["has_body"].sum() if "has_body" in sf.columns else 0
    body_rows.append({
        "symbol": sym,
        "n_filings": n,
        "n_with_body": int(n_body),
        "body_pct": n_body / n if n > 0 else 0,
    })

body_df = pd.DataFrame(body_rows).merge(results[["symbol", "group"]], on="symbol")

print(f"\n{'Group':>10s} {'Filings':>8s} {'With body':>10s} {'Body %':>8s}")
print("-" * 40)
for grp in ["winner", "partial", "loser"]:
    g = body_df[body_df["group"] == grp]
    print(f"{grp:>10s} {g['n_filings'].mean():8.0f} {g['n_with_body'].mean():10.0f} {g['body_pct'].mean()*100:7.1f}%")

# ============================================================
# 4. Sentiment distribution
# ============================================================
print("\n" + "=" * 90)
print("4. SENTIMENT DISTRIBUTION")
print("=" * 90)

sent_rows = []
for sym in results["symbol"]:
    sf = filings[filings["symbol"] == sym]
    if sf.empty or "polarity" not in sf.columns:
        continue
    sent_rows.append({
        "symbol": sym,
        "mean_polarity": sf["polarity"].mean(),
        "std_polarity": sf["polarity"].std(),
        "pct_positive": (sf["polarity"] > 0.05).mean(),
        "pct_negative": (sf["polarity"] < -0.05).mean(),
        "pct_neutral": ((sf["polarity"] >= -0.05) & (sf["polarity"] <= 0.05)).mean(),
        "polarity_range": sf["polarity"].max() - sf["polarity"].min(),
    })

sent_df = pd.DataFrame(sent_rows).merge(results[["symbol", "group"]], on="symbol")

print(f"\n{'Group':>10s} {'Mean pol':>9s} {'Std pol':>8s} {'%pos':>6s} {'%neg':>6s} {'%neu':>6s} {'Range':>7s}")
print("-" * 60)
for grp in ["winner", "partial", "loser"]:
    g = sent_df[sent_df["group"] == grp]
    print(f"{grp:>10s} {g['mean_polarity'].mean():9.4f} {g['std_polarity'].mean():8.4f} "
          f"{g['pct_positive'].mean()*100:5.1f}% {g['pct_negative'].mean()*100:5.1f}% "
          f"{g['pct_neutral'].mean()*100:5.1f}% {g['polarity_range'].mean():7.4f}")

print("\nPer-stock sentiment:")
for _, r in sent_df.sort_values("group").iterrows():
    print(f"  {r['symbol']:15s} [{r['group']:>7s}] pol={r['mean_polarity']:+.4f} "
          f"std={r['std_polarity']:.4f} pos={r['pct_positive']*100:.0f}% "
          f"neg={r['pct_negative']*100:.0f}% range={r['polarity_range']:.4f}")

# ============================================================
# 5. High-impact filing types (earnings, board outcomes)
# ============================================================
print("\n" + "=" * 90)
print("5. HIGH-IMPACT FILINGS (earnings + board outcomes with direction)")
print("=" * 90)

impact_rows = []
for sym in results["symbol"]:
    sf = filings[filings["symbol"] == sym]
    if sf.empty:
        continue
    n = len(sf)
    cats = sf["filing_category"].value_counts()
    
    earn_pos = cats.get("earnings_positive", 0)
    earn_neg = cats.get("earnings_negative", 0)
    earn_neu = cats.get("earnings_neutral", 0)
    board_pos = cats.get("board_outcome_positive", 0)
    board_neg = cats.get("board_outcome_negative", 0)
    div = cats.get("dividend_announced", 0)
    
    high_impact = earn_pos + earn_neg + board_pos + board_neg + div
    directional = earn_pos + earn_neg + board_pos + board_neg
    
    impact_rows.append({
        "symbol": sym,
        "n_filings": n,
        "high_impact": high_impact,
        "high_impact_pct": high_impact / n,
        "directional": directional,
        "directional_pct": directional / n,
        "earn_pos": earn_pos, "earn_neg": earn_neg,
        "board_pos": board_pos, "board_neg": board_neg,
        "dividend": div,
    })

impact_df = pd.DataFrame(impact_rows).merge(results[["symbol", "group", "auc_fusion", "delta_tok_num", "delta_fus_tok"]], on="symbol")

print(f"\n{'Group':>10s} {'HI count':>9s} {'HI %':>6s} {'Dir count':>10s} {'Dir %':>6s} {'E+':>4s} {'E-':>4s} {'B+':>4s} {'B-':>4s} {'Div':>4s}")
print("-" * 80)
for grp in ["winner", "partial", "loser"]:
    g = impact_df[impact_df["group"] == grp]
    print(f"{grp:>10s} {g['high_impact'].mean():9.1f} {g['high_impact_pct'].mean()*100:5.1f}% "
          f"{g['directional'].mean():10.1f} {g['directional_pct'].mean()*100:5.1f}% "
          f"{g['earn_pos'].mean():4.1f} {g['earn_neg'].mean():4.1f} "
          f"{g['board_pos'].mean():4.1f} {g['board_neg'].mean():4.1f} "
          f"{g['dividend'].mean():4.1f}")

print("\nPer-stock high-impact breakdown:")
for _, r in impact_df.sort_values("group").iterrows():
    print(f"  {r['symbol']:15s} [{r['group']:>7s}] HI={r['high_impact']:3.0f} ({r['high_impact_pct']*100:4.1f}%) "
          f"E+={r['earn_pos']:.0f} E-={r['earn_neg']:.0f} B+={r['board_pos']:.0f} B-={r['board_neg']:.0f} "
          f"Div={r['dividend']:.0f}")

# ============================================================
# 6. Correlation: what predicts hypothesis success?
# ============================================================
print("\n" + "=" * 90)
print("6. WHAT PREDICTS HYPOTHESIS SUCCESS? (correlation with delta AUC)")
print("=" * 90)

# Merge everything
analysis = freq_df.merge(body_df[["symbol", "body_pct"]], on="symbol")
analysis = analysis.merge(sent_df[["symbol", "mean_polarity", "std_polarity", "pct_positive", "pct_negative", "polarity_range"]], on="symbol")
analysis = analysis.merge(impact_df[["symbol", "high_impact_pct", "directional_pct"]], on="symbol")
analysis = analysis.merge(results[["symbol", "delta_tok_num", "delta_fus_tok", "auc_fusion"]], on="symbol")

predictors = ["n_filings", "filing_density", "avg_docs_per_day", "filings_per_month",
              "body_pct", "mean_polarity", "std_polarity", "pct_positive", "pct_negative",
              "polarity_range", "high_impact_pct", "directional_pct"]

print("\nCorrelation with Δ(tokens - numeric) AUC:")
for p in predictors:
    corr = analysis[p].corr(analysis["delta_tok_num"])
    bar = "█" * int(abs(corr) * 20)
    sign = "+" if corr > 0 else "-"
    print(f"  {p:25s} r={corr:+.3f} {sign}{bar}")

print("\nCorrelation with Δ(fusion - tokens) AUC:")
for p in predictors:
    corr = analysis[p].corr(analysis["delta_fus_tok"])
    bar = "█" * int(abs(corr) * 20)
    sign = "+" if corr > 0 else "-"
    print(f"  {p:25s} r={corr:+.3f} {sign}{bar}")

print("\nCorrelation with fusion AUC (absolute):")
for p in predictors:
    corr = analysis[p].corr(analysis["auc_fusion"])
    bar = "█" * int(abs(corr) * 20)
    sign = "+" if corr > 0 else "-"
    print(f"  {p:25s} r={corr:+.3f} {sign}{bar}")

if not dry_run:
    analysis.to_csv(output_dir / "filing_profile_analysis.csv", index=False)
    print(f"\nSaved: {output_dir / 'filing_profile_analysis.csv'}")
else:
    print(f"\n  [DRY RUN] Would write: {output_dir / 'filing_profile_analysis.csv'}")
