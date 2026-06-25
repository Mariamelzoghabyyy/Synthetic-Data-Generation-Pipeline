"""
Patch master_distributions.pkl in-place on the Modal volume.
Adds missing mortgage_rates and optional industry distributions.
Run once before 02_generate_persons.py.
"""
import modal
import pickle

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["pandas==2.2.0", "numpy==1.26.4", "pyarrow==15.0.0"])
    .add_local_file("config.py", "/root/config.py")
    .add_local_file("utils.py",  "/root/utils.py")
)

from config import VOLUME_NAMES

app      = modal.App("taxxx-patch-distributions")
dist_vol = modal.Volume.from_name(VOLUME_NAMES["dists"], create_if_missing=True)
logs_vol = modal.Volume.from_name(VOLUME_NAMES["logs"],  create_if_missing=True)

VOLUMES = {
    "/distributions": dist_vol,
    "/logs":          logs_vol,
}

# Historical 30-yr fixed mortgage rates (Freddie Mac approx)
MORTGAGE_RATES_HIST = {
    2019: 0.039, 2020: 0.031, 2021: 0.030,
    2022: 0.053, 2023: 0.068, 2024: 0.069, 2025: 0.065,
}

# Fallback industry distributions (used if missing from schedule_c)
INDUSTRY_FALLBACKS = {
    "retail": {
        "median_gross_receipts": 180_000, "gross_receipts_log_std": 0.95,
        "cogs_ratio": {"mean": 0.65, "std": 0.12},
        "deduction_ratios": {
            "rent": {"mean": 0.08, "std": 0.03},
            "wages": {"mean": 0.15, "std": 0.05},
            "utilities": {"mean": 0.03, "std": 0.01},
            "advertising": {"mean": 0.04, "std": 0.02},
        },
    },
    "healthcare": {
        "median_gross_receipts": 320_000, "gross_receipts_log_std": 0.85,
        "cogs_ratio": {"mean": 0.25, "std": 0.08},
        "deduction_ratios": {
            "wages": {"mean": 0.28, "std": 0.07},
            "rent": {"mean": 0.06, "std": 0.02},
            "insurance": {"mean": 0.05, "std": 0.02},
            "supplies": {"mean": 0.08, "std": 0.03},
        },
    },
    "real_estate": {
        "median_gross_receipts": 450_000, "gross_receipts_log_std": 1.10,
        "cogs_ratio": {"mean": 0.15, "std": 0.06},
        "deduction_ratios": {
            "advertising": {"mean": 0.06, "std": 0.03},
            "car_truck": {"mean": 0.04, "std": 0.02},
            "office_expense": {"mean": 0.03, "std": 0.01},
            "commissions": {"mean": 0.12, "std": 0.05},
        },
    },
    "tech": {
        "median_gross_receipts": 520_000, "gross_receipts_log_std": 1.05,
        "cogs_ratio": {"mean": 0.20, "std": 0.10},
        "deduction_ratios": {
            "wages": {"mean": 0.35, "std": 0.08},
            "rent": {"mean": 0.05, "std": 0.02},
            "software": {"mean": 0.06, "std": 0.03},
            "advertising": {"mean": 0.07, "std": 0.04},
        },
    },
}


@app.function(
    image=image,
    volumes=VOLUMES,
    cpu=2,
    memory=4096,
    timeout=300,
)
def patch_distributions():
    import os, sys
    sys.path.insert(0, "/root")
    os.environ["MODAL_TASK_ID"] = "1"

    from config import DIST_PKL
    from utils import get_logger
    log = get_logger("00b_patch", "00b_patch_dist.log")

    log.info("Loading PKL: %s", DIST_PKL)
    with open(DIST_PKL, "rb") as f:
        dist = pickle.load(f)

    patched_keys = []

    # ── 1. Inject mortgage_rates into all zones ───────────────────────────────
    log.info("Patching mortgage_rates...")
    for z in range(1, 6):
        zone_data = dist["housing"].get(z, dist["housing"].get(str(z)))
        if zone_data is None:
            log.warning("Zone %s not found in housing dict", z)
            continue
        if "mortgage_rates" not in zone_data:
            zone_data["mortgage_rates"] = MORTGAGE_RATES_HIST.copy()
            patched_keys.append(f"zone_{z}.mortgage_rates")
            log.info("  ✓ Added mortgage_rates to Zone %s", z)
        else:
            log.info("  - Zone %s already has mortgage_rates", z)

    # ── 2. Inject missing industry distributions ──────────────────────────────
    log.info("Patching missing schedule_c industries...")
    sch_c = dist.get("schedule_c", {})
    for ind_name, ind_data in INDUSTRY_FALLBACKS.items():
        if ind_name not in sch_c:
            sch_c[ind_name] = ind_data
            patched_keys.append(f"schedule_c.{ind_name}")
            log.info("  ✓ Added industry: %s", ind_name)
        else:
            log.info("  - Industry %s already present", ind_name)

    # ── 3. Save if changed ────────────────────────────────────────────────────
    if patched_keys:
        log.info("Saving patched PKL...")
        with open(DIST_PKL, "wb") as f:
            pickle.dump(dist, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("PKL patched successfully. Keys added: %s", patched_keys)
    else:
        log.info("No changes needed. PKL is already complete.")

    logs_vol.commit()
    return {"status": "ok", "patched_keys": patched_keys}


@app.local_entrypoint()
def main():
    result = patch_distributions.remote()
    print("\n" + "="*50)
    print("PATCH RESULT")
    print("="*50)
    print(f"Status: {result['status']}")
    if result['patched_keys']:
        print("Patched keys:")
        for k in result['patched_keys']:
            print(f"  ✓ {k}")
        print("\n✓ Run 00_check_distributions.py again to verify, then start generation.")
    else:
        print("No patches applied. PKL was already complete.")