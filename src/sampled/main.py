import pandas as pd
import logging
from pathlib import Path

from sampler   import sample_state
from validator import validate
from config    import (
    STATE_FILES, COLS, TARGET_FRAUD_RATES,
    COLS_TO_DROP, OUTPUT_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":

    print(f"\n  Output → {output_dir}\n")
    results = {}

    for i, state in enumerate(STATE_FILES.keys()):

        print(f"\n[{i+1}/5] Loading {state}...")
        original = pd.read_parquet(STATE_FILES[state])
        original = original.drop(
            columns=[c for c in COLS_TO_DROP if c in original.columns]
        )

        sampled = sample_state(state, df=original, seed_offset=i * 100)
        validate(original, sampled, state)

        out_path = output_dir / f"{state}.parquet"
        sampled.to_parquet(out_path, index=False)
        print(f"  Saved → {out_path}")

        results[state] = sampled
        del original

    print("\n" + "█"*65)
    print("  FINAL SUMMARY")
    print("█"*65)
    print(f"\n{'State':<14} {'Rows':>8} {'Persons':>9} "
          f"{'FraudRate':>10} {'Target':>10} {'Drift':>8}")
    print("-"*65)

    for state, df in results.items():
        n_rows     = len(df)
        n_persons  = df[COLS["person_id"]].nunique()
        fraud_rate = df[COLS["fraud_label"]].mean()
        target     = TARGET_FRAUD_RATES[state]
        drift      = fraud_rate - target
        print(
            f"{state:<14} {n_rows:>8,} {n_persons:>9,} "
            f"{fraud_rate:>10.4f} {target:>10.4f} "
            f"{drift:>+8.4f} "
            f"{'✓' if abs(drift) < 0.005 else '⚠'}"
        )

    combined = pd.concat(results.values(), ignore_index=True)
    combined.to_parquet(output_dir / "all_states.parquet", index=False)
    print(f"\n  Combined → {output_dir / 'all_states.parquet'}")
    print(f"  Shape    : {combined.shape}")