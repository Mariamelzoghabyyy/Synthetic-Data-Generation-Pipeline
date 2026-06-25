# 00_check_distributions.py
"""
Verify master_distributions.pkl has all required keys and
sensible values before starting the generation pipeline.
Run once before 02_generate_persons.py.
"""

import modal

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

app      = modal.App("taxxx-check-distributions")
dist_vol = modal.Volume.from_name(VOLUME_NAMES["dists"], create_if_missing=True)
logs_vol = modal.Volume.from_name(VOLUME_NAMES["logs"],  create_if_missing=True)

VOLUMES = {
    "/distributions": dist_vol,
    "/logs":          logs_vol,
}


@app.function(
    image=image,
    volumes=VOLUMES,
    cpu=2,
    memory=4096,
    timeout=300,
)
def check_distributions():
    import os, sys, pickle
    sys.path.insert(0, "/root")
    os.environ["MODAL_TASK_ID"] = "1"

    from config import DIST_PKL
    from utils import get_logger
    log = get_logger("00_check_dist", "00_check_distributions.log")

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info("Loading: %s", DIST_PKL)
    if not DIST_PKL.exists():
        log.error("DIST_PKL not found at %s", DIST_PKL)
        return {"status": "MISSING", "path": str(DIST_PKL)}

    with open(DIST_PKL, "rb") as f:
        dist = pickle.load(f)

    log.info("Top-level keys: %s", list(dist.keys()))

    results   = {}
    failures  = []
    warnings  = []

    # ── Required top-level structure ──────────────────────────────────────────
    # These are accessed by panels.py — all must exist
    REQUIRED_KEYS = {
        "bls":          "BLS occupation wage data",
        "schedule_c":   "Schedule C industry distributions",
        "housing":      "Housing data by zone",
    }

    for key, desc in REQUIRED_KEYS.items():
        if key not in dist:
            log.error("  ✗ MISSING key: '%s' (%s)", key, desc)
            failures.append(f"Missing top-level key: {key}")
        else:
            log.info("  ✓ Found key: '%s' (%s)", key, desc)
            results[key] = "present"

    if failures:
        log.error("Critical keys missing — cannot proceed")
        return {"status": "FAIL", "failures": failures}

    # ── BLS occupations ───────────────────────────────────────────────────────
    log.info("Checking BLS occupations...")
    bls = dist["bls"]

    if "occupations" not in bls:
        failures.append("dist['bls']['occupations'] missing")
        log.error("  ✗ dist['bls']['occupations'] missing")
    else:
        occs = bls["occupations"]
        log.info("  Occupations found: %d", len(occs))

        # Check a sample of occupations for required sub-keys
        REQUIRED_OCC_KEYS = ["log_mean", "log_std", "emp_share"]
        sample_occs = list(occs.keys())[:5]

        for occ in sample_occs:
            occ_data = occs[occ]
            missing  = [k for k in REQUIRED_OCC_KEYS if k not in occ_data]
            if missing:
                failures.append(f"occ '{occ}' missing keys: {missing}")
                log.error("  ✗ '%s' missing: %s", occ, missing)
            else:
                log.info(
                    "  ✓ %-30s  log_mean=%.2f  log_std=%.2f  emp_share=%.4f",
                    occ,
                    occ_data["log_mean"],
                    occ_data["log_std"],
                    occ_data["emp_share"],
                )

            # Sanity: log_mean should be in [9, 14] → annual wages $8k-$1.2M
            if "log_mean" in occ_data:
                lm = occ_data["log_mean"]
                if not (9.0 <= lm <= 14.0):
                    warnings.append(
                        f"occ '{occ}' log_mean={lm:.2f} outside [9, 14]"
                    )

        # Check emp_shares sum to ~1.0
        total_share = sum(
            o.get("emp_share", 0) for o in occs.values()
        )
        if abs(total_share - 1.0) > 0.01:
            warnings.append(
                f"emp_shares sum={total_share:.4f} (expected ~1.0)"
            )
            log.warning(
                "  ⚠ emp_shares sum=%.4f (expected 1.0)", total_share
            )
        else:
            log.info("  ✓ emp_shares sum=%.4f", total_share)

    # ── Schedule C industries ─────────────────────────────────────────────────
    log.info("Checking Schedule C industries...")
    sch_c = dist["schedule_c"]
    log.info("  Industries found: %d", len(sch_c))

    REQUIRED_INDUSTRY_KEYS = [
        "median_gross_receipts",
        "gross_receipts_log_std",
        "cogs_ratio",
        "deduction_ratios",
    ]

    # Check each industry used in ZONE_PROFILES
    EXPECTED_INDUSTRIES = [
        "retail", "food_service", "construction",
        "professional_services", "healthcare",
        "real_estate", "tech",
    ]

    for ind in EXPECTED_INDUSTRIES:
        if ind not in sch_c:
            warnings.append(
                f"Industry '{ind}' not in schedule_c "
                f"(panels.py will use professional_services fallback)"
            )
            log.warning("  ⚠ Industry '%s' not found (fallback will apply)", ind)
            continue

        ind_data = sch_c[ind]
        missing  = [k for k in REQUIRED_INDUSTRY_KEYS if k not in ind_data]
        if missing:
            failures.append(f"Industry '{ind}' missing keys: {missing}")
            log.error("  ✗ '%s' missing: %s", ind, missing)
            continue

        # Sanity checks
        mgr = ind_data["median_gross_receipts"]
        std = ind_data["gross_receipts_log_std"]
        cr  = ind_data["cogs_ratio"]

        ok = (
            1_000 <= mgr <= 50_000_000   # $1k to $50M is plausible
            and 0.1 <= std <= 2.5
            and "mean" in cr
            and "std"  in cr
            and 0.0 <= cr["mean"] <= 0.95
        )

        symbol = "✓" if ok else "✗"
        log.info(
            "  %s %-22s  median_gr=$%10,.0f  log_std=%.2f  "
            "cogs_mean=%.2f",
            symbol, ind, mgr, std, cr.get("mean", 0),
        )

        if not ok:
            warnings.append(f"Industry '{ind}' has suspicious values")

        # Check deduction_ratios
        ded_r = ind_data.get("deduction_ratios", {})
        if not ded_r:
            warnings.append(f"Industry '{ind}' has empty deduction_ratios")
            log.warning("  ⚠ '%s' deduction_ratios is empty", ind)
        else:
            log.info(
                "  ✓ '%s' deduction categories: %s",
                ind, list(ded_r.keys()),
            )

    # ── Housing by zone ───────────────────────────────────────────────────────
    log.info("Checking housing by zone...")
    housing = dist["housing"]

    REQUIRED_HOUSING_KEYS = [
        "median_home_value",
        "median_rent_2br",
        "rental_property_rate",
        "mortgage_rates",
    ]

    for zone in range(1, 6):
        # Try both int and str keys — pkl may use either
        zone_data = housing.get(zone, housing.get(str(zone), None))

        if zone_data is None:
            failures.append(f"housing zone {zone} missing")
            log.error("  ✗ Zone %d missing from housing", zone)
            continue

        missing = [k for k in REQUIRED_HOUSING_KEYS if k not in zone_data]
        if missing:
            failures.append(f"housing zone {zone} missing keys: {missing}")
            log.error("  ✗ Zone %d missing keys: %s", zone, missing)
            continue

        mhv  = zone_data["median_home_value"]
        rent = zone_data["median_rent_2br"]
        rpr  = zone_data["rental_property_rate"]
        mr   = zone_data["mortgage_rates"]

        ok = (
            50_000 <= mhv  <= 5_000_000
            and 400  <= rent <= 10_000
            and 0.01 <= rpr  <= 0.50
            and isinstance(mr, dict)
            and len(mr) > 0
        )

        symbol = "✓" if ok else "✗"
        log.info(
            "  %s Zone %d  home=$%9,.0f  rent=$%5,.0f  "
            "rental_rate=%.2f  mortgage_years=%d",
            symbol, zone, mhv, rent, rpr, len(mr),
        )

        if not ok:
            warnings.append(f"Zone {zone} housing has suspicious values")

        # Check mortgage_rates has entries for 2019-2025
        for yr in range(2019, 2026):
            if yr not in mr and str(yr) not in mr:
                warnings.append(
                    f"Zone {zone} mortgage_rates missing year {yr}"
                )
                log.warning(
                    "  ⚠ Zone %d mortgage_rates missing year %d", zone, yr
                )

    # ── Optional keys panels.py uses ──────────────────────────────────────────
    log.info("Checking optional keys...")
    OPTIONAL_KEYS = {
        "acs":      "ACS demographic distributions",
        "zillow":   "Zillow home price data",
        "hmda":     "HMDA mortgage data",
    }
    for key, desc in OPTIONAL_KEYS.items():
        if key in dist:
            log.info("  ✓ Optional '%s' present (%s)", key, desc)
        else:
            log.info("  - Optional '%s' not present (%s) — ok", key, desc)

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("DISTRIBUTION CHECK SUMMARY")
    log.info("=" * 60)
    log.info("Failures: %d", len(failures))
    log.info("Warnings: %d", len(warnings))

    for f in failures:
        log.error("  FAIL: %s", f)
    for w in warnings:
        log.warning("  WARN: %s", w)

    status = "PASS" if not failures else "FAIL"
    log.info("Status: %s", status)

    if status == "PASS":
        log.info("Distribution file looks good — safe to start generation")
    else:
        log.error("Fix failures before running generation pipeline")

    logs_vol.commit()
    return {
        "status":   status,
        "failures": failures,
        "warnings": warnings,
        "top_keys": list(dist.keys()),
    }


@app.local_entrypoint()
def main():
    result = check_distributions.remote()

    print("\n" + "=" * 60)
    print("DISTRIBUTION FILE CHECK")
    print("=" * 60)
    print(f"Status:   {result['status']}")
    print(f"Top keys: {result['top_keys']}")
    print(f"Failures: {len(result['failures'])}")
    print(f"Warnings: {len(result['warnings'])}")

    for f in result["failures"]:
        print(f"  FAIL: {f}")
    for w in result["warnings"]:
        print(f"  WARN: {w}")

    print()
    if result["status"] == "PASS":
        print("✓ GO — safe to run: modal run 02_generate_persons.py")
    else:
        print("✗ STOP — fix distribution file before generating")