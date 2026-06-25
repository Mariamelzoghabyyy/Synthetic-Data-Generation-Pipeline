# run_all.py

import time
import sys
import importlib.util
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


def run_phase(phase_index: int, name: str, module_path: Path) -> float:
    print()
    print("=" * 65)
    print(f"RUNNING: {name}")
    print("=" * 65)
    t0 = time.time()

    module_name = f"pipeline_phase_{phase_index:02d}_{module_path.stem}"
    spec        = importlib.util.spec_from_file_location(module_name, module_path)
    module      = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.run()

    elapsed = time.time() - t0
    print(f"\n[OK] {name} completed in {elapsed:.1f}s")
    return elapsed


if __name__ == "__main__":
    base = Path(__file__).parent

    phases = [
        ("Phase 1 -- Preprocessing",          base / "00_preprocessing.py"),
        ("Phase 2 -- Core Features",           base / "01_core_features.py"),
        ("Phase 3 -- Peer Statistics",         base / "02_peer_statistics.py"),
        ("Phase 4 -- Z-Score Features",        base / "03_zscore_features.py"),
        ("Phase 5 -- Forensic Features",       base / "04_forensic_features.py"),
        ("Phase 6 -- Network Features",        base / "05_network_features.py"),
        ("Phase 7 -- Impossibility Flags",     base / "06_impossibility_flags.py"),
        ("Phase 8 -- Composite Scores",        base / "07_composite_scores.py"),
        ("Phase 9 -- Split to ML Ready",       base / "08_feature_selection.py"),
    ]

    t_total = time.time()
    timings: dict[str, float] = {}

    for idx, (name, path) in enumerate(phases):
        if not path.exists():
            print(f"\n[SKIP] {name}: file not found at {path}")
            timings[name] = 0.0
            continue
        elapsed       = run_phase(idx, name, path)
        timings[name] = elapsed

    print()
    print("=" * 65)
    print("FULL PIPELINE COMPLETE")
    print("=" * 65)
    for name, t in timings.items():
        status = "[SKIP]" if t == 0.0 else "      "
        print(f"  {status} {name:<45} {t:>6.1f}s")
    print(f"  {'TOTAL':<52} {time.time() - t_total:>6.1f}s")

    print()
    print("Output:")
    from config import ML_READY_DIR
    for f in sorted(ML_READY_DIR.rglob("*.parquet")):
        mb = f.stat().st_size / 1e6
        print(f"  {f.relative_to(ML_READY_DIR)}  ({mb:.1f} MB)")