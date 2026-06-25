# 06b_inspect_seeds.py
"""
Quick inspection of seed panels before GAN training.
Run this after 06_generate_splits.py completes.
Call: modal run 06b_inspect_seeds.py
"""

import modal
import numpy as np
import pandas as pd

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "pandas==2.2.0",
        "numpy==1.26.4",
        "pyarrow==15.0.0",
    ])
    .add_local_file("config.py", "/root/config.py")
    .add_local_file("utils.py",  "/root/utils.py")
)

from config import VOLUME_NAMES

app        = modal.App("taxxx-inspect-seeds")
final_vol  = modal.Volume.from_name(VOLUME_NAMES["final"],  create_if_missing=True)
panels_vol = modal.Volume.from_name(VOLUME_NAMES["panels"], create_if_missing=True)
logs_vol   = modal.Volume.from_name(VOLUME_NAMES["logs"],   create_if_missing=True)

VOLUMES = {
    "/final_dataset": final_vol,
    "/seed_panels":   panels_vol,
    "/logs":          logs_vol,
}


@app.function(
    image=image,
    volumes=VOLUMES,
    cpu=2,
    memory=8192,
    timeout=600,
)
def inspect_seeds():
    import os, sys
    sys.path.insert(0, "/root")
    os.environ["MODAL_TASK_ID"] = "1"

    from config import (
        SEED_W2, SEED_SE, SEED_ITEMIZERS,
        SEED_COMPLIANT, SEED_EVADERS,
        TRAIN_ALL, VAL_2024, TEST_2025,
        FINAL_BASE,
    )
    from utils import get_logger
    log = get_logger("06b_inspect", "06b_inspect_seeds.log")

    # ── IRS benchmarks for quick comparison ───────────────────────────────────
    IRS = {
        "agi":                {"median": 46_000, "mean": 78_000},
        "w2_wages":           {"median": 42_000, "mean": 65_000},
        "total_tax_liability":{"median":  4_200, "mean": 12_000},
        "net_se_income":      {"median": 18_000, "mean": 35_000},
    }

    # ── All files to check ────────────────────────────────────────────────────
    # (label, path, is_required)
    FILES = [
        ("TRAIN_ALL",        TRAIN_ALL,    True),
        ("VAL_2024",         VAL_2024,     True),
        ("TEST_2025",        TEST_2025,    True),
        ("SEED_W2",          SEED_W2,      True),
        ("SEED_SE",          SEED_SE,      True),
        ("SEED_ITEMIZERS",   SEED_ITEMIZERS, False),
        ("SEED_COMPLIANT",   SEED_COMPLIANT, True),
        ("SEED_EVADERS",     SEED_EVADERS,   True),
        ("TRAIN_PERSON_LEVEL",
         FINAL_BASE / "individuals" / "person_level" / "train_person_level.parquet",
         False),
        ("VAL_PERSON_LEVEL",
         FINAL_BASE / "individuals" / "person_level" / "val_person_level.parquet",
         False),
        ("TEST_PERSON_LEVEL",
         FINAL_BASE / "individuals" / "person_level" / "test_person_level.parquet",
         False),
    ]

    results = {}

    for label, path, required in FILES:

        log.info("=" * 60)
        log.info("Checking: %s", label)
        log.info("  Path: %s", path)

        # ── Existence check ───────────────────────────────────────────────────
        if not path.exists():
            status = "MISSING_REQUIRED" if required else "MISSING_OPTIONAL"
            log.warning("  %s — %s", status, path)
            results[label] = {"status": status, "rows": 0}
            continue

        # ── Load ──────────────────────────────────────────────────────────────
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            log.error("  LOAD ERROR: %s", e)
            results[label] = {"status": "LOAD_ERROR", "rows": 0, "error": str(e)}
            continue

        r = {
            "status":     "ok",
            "rows":       len(df),
            "cols":       len(df.columns),
            "checks":     [],
            "warnings":   [],
            "failures":   [],
        }

        log.info("  Rows: %d  Cols: %d", r["rows"], r["cols"])

        # ── 1. Fraud rate ─────────────────────────────────────────────────────
        if "fraud_label" in df.columns:
            fraud_rate = float(df["fraud_label"].mean())
            r["fraud_rate"] = round(fraud_rate, 4)

            fraud_ok = 0.18 <= fraud_rate <= 0.24
            symbol   = "✓" if fraud_ok else "✗"
            msg      = f"  {symbol} fraud_rate={fraud_rate:.4f}  (expect 0.18-0.24)"
            log.info(msg)
            r["checks"].append(msg)
            if not fraud_ok:
                r["failures"].append(f"fraud_rate={fraud_rate:.4f} out of range")
        else:
            log.warning("  ✗ fraud_label column missing")
            r["failures"].append("fraud_label column missing")

        # ── 2. fraud_label=1 with fraud_type=none ─────────────────────────────
        if "fraud_label" in df.columns and "fraud_type" in df.columns:
            contradiction = (
                (df["fraud_label"] == 1) & (df["fraud_type"] == "none")
            ).sum()
            symbol = "✓" if contradiction == 0 else "✗"
            msg    = f"  {symbol} fraud_label=1 & fraud_type=none: {contradiction} rows"
            log.info(msg)
            r["checks"].append(msg)
            if contradiction > 0:
                r["failures"].append(
                    f"{contradiction} rows have fraud_label=1 but fraud_type=none"
                )

        # ── 3. Evasion amount populated for evaders ───────────────────────────
        if "fraud_label" in df.columns and "evasion_amount" in df.columns:
            evaders      = df[df["fraud_label"] == 1]
            zero_evasion = (evaders["evasion_amount"].fillna(0) == 0).sum()
            pct_zero     = zero_evasion / max(len(evaders), 1)
            symbol       = "✓" if pct_zero < 0.05 else "✗"
            msg          = (
                f"  {symbol} evaders with $0 evasion_amount: "
                f"{zero_evasion} ({pct_zero:.1%})"
            )
            log.info(msg)
            r["checks"].append(msg)
            if pct_zero >= 0.05:
                r["failures"].append(
                    f"{pct_zero:.1%} of evaders have zero evasion_amount"
                )

            # Evasion amount distribution
            ea = evaders["evasion_amount"].dropna()
            ea = ea[ea > 0]
            if len(ea):
                log.info(
                    "  evasion_amount (non-zero evaders): "
                    "median=$%,.0f  mean=$%,.0f  p99=$%,.0f",
                    float(ea.median()), float(ea.mean()),
                    float(ea.quantile(0.99)),
                )

        # ── 4. Evasion rate sanity ────────────────────────────────────────────
        if "evasion_rate" in df.columns:
            er = df["evasion_rate"].dropna()
            if len(er):
                er_mean = float(er.mean())
                er_max  = float(er.max())
                symbol  = "✓" if er_mean < 1.0 and er_max <= 1.0 else "✗"
                msg     = (
                    f"  {symbol} evasion_rate: "
                    f"mean={er_mean:.4f}  max={er_max:.4f}  "
                    f"(expect mean<1.0, max<=1.0)"
                )
                log.info(msg)
                r["checks"].append(msg)
                if er_mean >= 1.0 or er_max > 1.0:
                    r["failures"].append(
                        f"evasion_rate out of range: mean={er_mean:.2f} max={er_max:.2f}"
                    )

        # ── 5. Income distributions vs IRS ───────────────────────────────────
        log.info("  Income distributions vs IRS benchmarks:")
        for col, bench in IRS.items():
            if col not in df.columns:
                continue
            s = df[col].dropna()
            s = s[s > 0]   # exclude structural zeros
            if len(s) == 0:
                log.warning("  ⚠ %s: all null/zero", col)
                r["warnings"].append(f"{col} all null/zero")
                continue

            med      = float(s.median())
            mn       = float(s.mean())
            med_dev  = abs(med - bench["median"]) / bench["median"]
            mean_dev = abs(mn  - bench["mean"])   / bench["mean"]
            ok       = med_dev < 0.40 and mean_dev < 0.40
            symbol   = "✓" if ok else "✗"

            log.info(
                "  %s %-25s  median=%10,.0f (IRS %10,.0f  dev=%+.0f%%)  "
                "mean=%10,.0f (IRS %10,.0f  dev=%+.0f%%)",
                symbol, col,
                med,  bench["median"], (med - bench["median"]) / bench["median"] * 100,
                mn,   bench["mean"],   (mn  - bench["mean"])   / bench["mean"]   * 100,
            )
            if not ok:
                r["warnings"].append(
                    f"{col} median dev={med_dev:.0%} mean dev={mean_dev:.0%}"
                )

        # ── 6. W2 wages zero rate ─────────────────────────────────────────────
        if "w2_wages" in df.columns and "taxpayer_type" in df.columns:
            w2_filers = df[df["taxpayer_type"].isin(["pure_w2", "w2_with_side_biz"])]
            if len(w2_filers):
                zero_w2   = (w2_filers["w2_wages"].fillna(0) == 0).sum()
                pct_zero  = zero_w2 / len(w2_filers)
                symbol    = "✓" if pct_zero < 0.05 else "✗"
                msg       = (
                    f"  {symbol} W2-type persons with zero wages: "
                    f"{zero_w2} ({pct_zero:.1%})  (expect <5%)"
                )
                log.info(msg)
                r["checks"].append(msg)
                if pct_zero >= 0.05:
                    r["failures"].append(
                        f"{pct_zero:.1%} of W2 persons have zero wages"
                    )

        # ── 7. Tax logic: taxable_income <= AGI ──────────────────────────────
        if "taxable_income" in df.columns and "agi" in df.columns:
            bad = (df["taxable_income"] > df["agi"]).sum()
            symbol = "✓" if bad == 0 else "✗"
            msg    = f"  {symbol} taxable_income > agi: {bad} rows"
            log.info(msg)
            r["checks"].append(msg)
            if bad > 0:
                r["failures"].append(f"{bad} rows where taxable_income > agi")

        # ── 8. Tax logic: COGS <= gross_receipts ─────────────────────────────
        if "cogs" in df.columns and "gross_receipts" in df.columns:
            both = df[df["cogs"].notna() & df["gross_receipts"].notna()]
            bad  = (both["cogs"] > both["gross_receipts"]).sum()
            symbol = "✓" if bad == 0 else "✗"
            msg    = f"  {symbol} cogs > gross_receipts: {bad} rows"
            log.info(msg)
            r["checks"].append(msg)
            if bad > 0:
                r["warnings"].append(f"{bad} rows where cogs > gross_receipts")

        # ── 9. Null rate on critical columns ──────────────────────────────────
        MUST_NOT_BE_NULL = [
            "person_id", "tax_year", "zone", "age",
            "fraud_label", "fraud_type", "agi",
            "total_tax_liability", "taxpayer_type",
            "filing_status", "deduction_taken",
        ]
        null_issues = []
        for col in MUST_NOT_BE_NULL:
            if col not in df.columns:
                null_issues.append(f"{col} MISSING")
                continue
            n_null = int(df[col].isna().sum())
            if n_null > 0:
                null_issues.append(f"{col}={n_null} nulls")

        if null_issues:
            log.warning("  ✗ Null violations: %s", ", ".join(null_issues))
            r["failures"].extend(null_issues)
        else:
            log.info("  ✓ All critical columns non-null")

        # ── 10. Fraud type distribution ───────────────────────────────────────
        if "fraud_type" in df.columns and "fraud_label" in df.columns:
            evaders = df[df["fraud_label"] == 1]
            if len(evaders):
                ft_dist = (
                    evaders["fraud_type"]
                    .value_counts(normalize=True)
                    .round(3)
                )
                log.info("  Fraud type distribution:\n%s", ft_dist.to_string())

        # ── 11. Year-over-year AGI trend (train only) ─────────────────────────
        if (label in ("TRAIN_ALL", "TRAIN_PERSON_LEVEL")
                and "tax_year" in df.columns
                and "agi" in df.columns):
            yoy = df.groupby("tax_year")["agi"].median()
            pct_changes = yoy.pct_change() * 100
            log.info("  YoY AGI median change:")
            for yr, chg in pct_changes.items():
                if not pd.isna(chg):
                    flag = "✓" if abs(chg) < 20 else "✗"
                    