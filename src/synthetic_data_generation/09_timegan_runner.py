# 09_timegan_runner.py
"""
TimeGAN training for individual temporal sequences.
Learns how income/fraud evolves year-over-year.

Two models:
  compliant : learns normal income trajectory patterns
  evaders   : learns fraud onset and escalation patterns

Run:
  modal run 09_timegan_runner.py::train_compliant
  modal run 09_timegan_runner.py::train_evaders
"""

import modal
import pandas as pd
import numpy as np
from pathlib import Path

# ── Image ─────────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.2.0",
        "pandas==2.2.0",
        "numpy==1.26.4",
        "pyarrow==15.0.0",
        "scikit-learn==1.4.0",
    ])
    .add_local_file("config.py", "/root/config.py")
    .add_local_file("utils.py",  "/root/utils.py")
)

# ── App & volumes ─────────────────────────────────────────────────────────────
from config import VOLUME_NAMES

app        = modal.App("taxxx-pipeline-09-timegan")
final_vol  = modal.Volume.from_name(VOLUME_NAMES["final"],   create_if_missing=True)
panels_vol = modal.Volume.from_name(VOLUME_NAMES["panels"],  create_if_missing=True)
logs_vol   = modal.Volume.from_name(VOLUME_NAMES["logs"],    create_if_missing=True)
gan_vol    = modal.Volume.from_name(VOLUME_NAMES["gan_out"], create_if_missing=True) 
VOLUMES = {
    "/final_dataset": final_vol,
    "/seed_panels":   panels_vol,
    "/logs":          logs_vol,
    "/gan_output":    gan_vol, 
}

# ── Feature declarations ──────────────────────────────────────────────────────
# Column names must exactly match 05_generate_panels.py output.
# "evasion_amount" excluded from compliant features (always 0 → uninformative).
# It IS included in evader features to capture escalation pattern.

TEMPORAL_FEATURES_COMPLIANT: list[str] = [
    "w2_wages",
    "net_se_income",
    "agi",
    "taxable_income",
    "total_tax_liability",        # script 05 name (not "tax_liability")
    "effective_tax_rate",
    "electricity_kwh",
    "utility_income_ratio",
    "lifestyle_income_ratio",     # script 05 name (not "_proxy")
]

TEMPORAL_FEATURES_EVADERS: list[str] = TEMPORAL_FEATURES_COMPLIANT
# Static features copied from the seed person onto synthetic rows
# "fraud_type" is the panel column name (script 05); not "primary_fraud_type"
STATIC_FEATURES: list[str] = [
    "zone",
    "sex",
    "education",
    "taxpayer_type",
    "fraud_type",                 # panel output column name
]

SEQUENCE_LENGTH = 7        # 2019-2025 inclusive
HIDDEN_DIM      = 24
NUM_LAYERS      = 3
ITERATIONS      = 50_000
BATCH_SIZE      = 128


def _build_sequences(
    df:            pd.DataFrame,
    feat_cols:     list[str],
    min_years:     int = 3,
) -> tuple[np.ndarray, list[dict], list, list[str]]:
    """
    Reshape flat panel (person × year rows) → 3D array
    (n_persons, seq_len, n_features).

    Only persons with >= min_years of data are included.
    Shorter sequences are zero-padded on the right.

    Returns:
        sequences   : float32 array (n, SEQUENCE_LENGTH, n_feats)
        statics     : list of dicts, one per person
        person_ids  : list of person_id strings
        feat_cols   : the feature columns actually used (subset of input)
    """
    # Use tax_year (script 05 column); fall back to "year" if somehow present
    year_col = "tax_year" if "tax_year" in df.columns else "year"

    # Only keep feature columns that actually exist in this dataframe
    feat_cols = [f for f in feat_cols if f in df.columns]
    stat_cols = [f for f in STATIC_FEATURES if f in df.columns]

    # Fill numeric NAs with 0.0 (structural nulls become 0 for the sequence model)
    df = df.copy()
    df[feat_cols] = df[feat_cols].fillna(0.0)

    df = df.sort_values(["person_id", year_col])
    grouped = df.groupby("person_id", sort=False)

    valid_pids = [
        pid for pid, grp in grouped if len(grp) >= min_years
    ]
    if not valid_pids:
        raise ValueError(
            f"No persons with >= {min_years} years of data found. "
            "Check that the seed parquet contains multi-year panel rows."
        )

    n         = len(valid_pids)
    n_feats   = len(feat_cols)
    sequences = np.zeros((n, SEQUENCE_LENGTH, n_feats), dtype=np.float32)
    statics: list[dict] = []
    person_ids: list    = []

    for i, pid in enumerate(valid_pids):
        grp  = grouped.get_group(pid).sort_values(year_col)
        T    = min(len(grp), SEQUENCE_LENGTH)
        sequences[i, :T, :] = grp[feat_cols].values[:T].astype(np.float32)
        statics.append(grp[stat_cols].iloc[0].to_dict())
        person_ids.append(pid)

    return sequences, statics, person_ids, feat_cols


def _normalize(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max normalize to [0, 1] per feature across all persons and time steps."""
    mins  = X.min(axis=(0, 1), keepdims=True)
    maxes = X.max(axis=(0, 1), keepdims=True)
    denom = np.where(maxes - mins == 0, 1.0, maxes - mins)
    return (X - mins) / denom, mins, maxes


def _denormalize(
    X_norm: np.ndarray,
    mins:   np.ndarray,
    maxes:  np.ndarray,
) -> np.ndarray:
    denom = np.where(maxes - mins == 0, 1.0, maxes - mins)
    return X_norm * denom + mins


def _timegan_components(
    input_dim:  int,
    hidden_dim: int,
    num_layers: int,
    device,
):
    """Build TimeGAN embedder, recovery, generator, supervisor, discriminator."""
    import torch
    import torch.nn as nn

    class GRUNet(nn.Module):
        def __init__(
            self,
            in_dim:     int,
            hid_dim:    int,
            out_dim:    int,
            n_layers:   int,
            activation  = None,
        ):
            super().__init__()
            self.gru = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True)
            self.fc  = nn.Linear(hid_dim, out_dim)
            self.act = activation() if activation else nn.Identity()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.gru(x)
            return self.act(self.fc(out))

    embedder      = GRUNet(input_dim,  hidden_dim, hidden_dim, num_layers,
                           nn.Sigmoid).to(device)
    recovery      = GRUNet(hidden_dim, hidden_dim, input_dim,  num_layers,
                           nn.Sigmoid).to(device)
    generator     = GRUNet(hidden_dim, hidden_dim, hidden_dim, num_layers,
                           nn.Sigmoid).to(device)
    supervisor    = GRUNet(hidden_dim, hidden_dim, hidden_dim, num_layers,
                           nn.Sigmoid).to(device)
    discriminator = GRUNet(hidden_dim, hidden_dim, 1,          num_layers,
                           nn.Sigmoid).to(device)

    return embedder, recovery, generator, supervisor, discriminator


def run_timegan(
    model_name:     str,
    input_path:     Path,
    output_path:    Path,
    target_persons: int,
    temporal_feats: list[str],
    is_evader:      bool,
    iterations:     int,
    seed:           int,
) -> None:
    """
    Full TimeGAN training + sampling pipeline.
    Writes a flat parquet of (person × year) rows to output_path.
    """
    import sys
    import os
    sys.path.insert(0, "/root")
    os.environ["MODAL_TASK_ID"] = "1"

    import torch
    from utils import get_logger, make_dirs, write_parquet

    log    = get_logger(f"timegan_{model_name}", f"09_timegan_{model_name}.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    make_dirs(output_path.parent)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Seed file not found: {input_path}\n"
            "Run 06_generate_splits.py first."
        )

    log.info("Loading seed: %s", input_path)
    df = pd.read_parquet(input_path)
    log.info("Rows: %d", len(df))

    sequences, statics, person_ids, feat_cols = _build_sequences(
        df, temporal_feats
    )
    n_persons, T, n_feats = sequences.shape
    log.info(
        "Persons: %d  seq_len=%d  features=%d",
        n_persons, T, n_feats,
    )

    X_norm, mins, maxes = _normalize(sequences)
    X_tensor = torch.FloatTensor(X_norm).to(device)

    # ── Build networks ────────────────────────────────────────────────────────
    embedder, recovery, generator, supervisor, discriminator = \
        _timegan_components(n_feats, HIDDEN_DIM, NUM_LAYERS, device)

    lr    = 1e-3
    opt_e = torch.optim.Adam(
        list(embedder.parameters()) + list(recovery.parameters()), lr=lr
    )
    opt_g = torch.optim.Adam(
        list(generator.parameters()) + list(supervisor.parameters()), lr=lr
    )
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

    bce   = torch.nn.BCELoss()
    mse   = torch.nn.MSELoss()
    gamma = 1.0

    def _sample_batch(size: int) -> "torch.Tensor":
        idx = rng.integers(0, n_persons, size)
        return X_tensor[idx]

    def _random_noise(size: int) -> "torch.Tensor":
        noise = rng.uniform(0, 1, (size, T, HIDDEN_DIM)).astype(np.float32)
        return torch.FloatTensor(noise).to(device)

    # ── Phase 1: Embedding network ────────────────────────────────────────────
    log.info("Phase 1: Embedding network (%d steps)...", iterations // 3)
    for step in range(iterations // 3):
        X_batch = _sample_batch(BATCH_SIZE)
        opt_e.zero_grad()
        H     = embedder(X_batch)
        X_hat = recovery(H)
        loss  = 10.0 * mse(X_hat, X_batch)
        loss.backward()
        opt_e.step()
        if step % 5_000 == 0:
            log.info("  embed step %6d  loss=%.6f", step, loss.item())

    # ── Phase 2: Supervised training ──────────────────────────────────────────
    log.info("Phase 2: Supervisor training (%d steps)...", iterations // 3)
    for step in range(iterations // 3):
        X_batch = _sample_batch(BATCH_SIZE)
        opt_g.zero_grad()
        H    = embedder(X_batch).detach()
        H_s  = supervisor(H)
        loss = mse(H[:, 1:, :], H_s[:, :-1, :])
        loss.backward()
        opt_g.step()
        if step % 5_000 == 0:
            log.info("  super step %6d  loss=%.6f", step, loss.item())

    # ── Phase 3: Joint GAN training ───────────────────────────────────────────
    log.info("Phase 3: Joint GAN training (%d steps)...", iterations)
    for step in range(iterations):
        # Train generator twice per discriminator update
        for _ in range(2):
            Z       = _random_noise(BATCH_SIZE)
            X_batch = _sample_batch(BATCH_SIZE)
            opt_g.zero_grad()

            H       = embedder(X_batch).detach()
            E_hat   = generator(Z)
            H_hat   = supervisor(E_hat)
            H_hat_s = supervisor(H)

            Y_fake   = discriminator(H_hat)
            G_loss_u = bce(Y_fake, torch.ones_like(Y_fake))
            G_loss_s = mse(H[:, 1:, :], H_hat_s[:, :-1, :])

            X_hat    = recovery(H_hat)
            G_loss_v = (
                mse(X_hat.mean(dim=0), X_batch.mean(dim=0))
                + mse(X_hat.std(dim=0),  X_batch.std(dim=0))
            )
            G_loss   = G_loss_u + gamma * G_loss_s + 100.0 * G_loss_v
            G_loss.backward()
            opt_g.step()

        # Discriminator step
        Z       = _random_noise(BATCH_SIZE)
        X_batch = _sample_batch(BATCH_SIZE)
        opt_d.zero_grad()

        H     = embedder(X_batch).detach()
        E_hat = generator(Z).detach()
        H_hat = supervisor(E_hat).detach()

        Y_real = discriminator(H)
        Y_fake = discriminator(H_hat)
        D_loss = (
            bce(Y_real, torch.ones_like(Y_real))
            + bce(Y_fake, torch.zeros_like(Y_fake))
        )
        # Only update discriminator when it hasn't collapsed
        if D_loss.item() > 0.15:
            D_loss.backward()
            opt_d.step()

        if step % 5_000 == 0:
            log.info(
                "  joint step %6d  G=%.4f  D=%.4f",
                step, G_loss.item(), D_loss.item(),
            )

    # ── Sample synthetic sequences ────────────────────────────────────────────
    log.info("Sampling %d persons...", target_persons)
    embedder.eval()
    recovery.eval()
    generator.eval()
    supervisor.eval()

    n_batches  = (target_persons + BATCH_SIZE - 1) // BATCH_SIZE
    synth_seqs: list[np.ndarray] = []

    with torch.no_grad():
        for _ in range(n_batches):
            Z     = _random_noise(BATCH_SIZE)
            E_hat = generator(Z)
            H_hat = supervisor(E_hat)
            X_hat = recovery(H_hat)
            synth_seqs.append(X_hat.cpu().numpy())

    synth_arr = np.concatenate(synth_seqs, axis=0)[:target_persons]
    synth_arr = np.clip(_denormalize(synth_arr, mins, maxes), 0.0, None)

    # ── Flatten to person × year rows ─────────────────────────────────────────
    years    = list(range(2019, 2019 + SEQUENCE_LENGTH))
    n_static = len(statics)
    rows: list[dict] = []

    for i in range(target_persons):
        # Cycle through real statics for demographic context;
        # assign a new unique synthetic person_id to avoid ID collisions
        static_src = statics[i % n_static].copy()
        static_src.pop("person_id", None)   # remove real person_id if present

        for t in range(SEQUENCE_LENGTH):
            row: dict = {
                "person_id": f"TG_{model_name[0].upper()}_{i:07d}",
                "tax_year":  years[t],
            }
            row.update(static_src)
            for j, feat in enumerate(feat_cols):
                row[feat] = float(synth_arr[i, t, j])

            # Derive fraud_label from evasion_amount if present,
            # otherwise use the is_evader model flag as a default
            if is_evader:
                    row["fraud_label"] = 1  # all rows from evader model are fraud
            else:
                row["fraud_label"] = 0  # all rows from compliant model are clean


            rows.append(row)

    result_df = pd.DataFrame(rows)
    log.info(
        "Flat rows: %d  fraud_rate=%.4f",
        len(result_df), result_df["fraud_label"].mean(),
    )

    write_parquet(result_df, output_path)
    log.info("Saved -> %s", output_path)


# ── Modal functions ───────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes=VOLUMES,
    gpu="T4",
    cpu=4,
    memory=16_384,
    timeout=86_400,
    retries=0,
)
def train_compliant():
    import sys, os
    sys.path.insert(0, "/root")
    os.environ["MODAL_TASK_ID"] = "1"
    from config import SEED_COMPLIANT, GAN_COMPLIANT_SEQ, RANDOM_SEED
    run_timegan(
        model_name     = "compliant",
        input_path     = SEED_COMPLIANT,
        output_path    = GAN_COMPLIANT_SEQ,
        target_persons = 280_000,
        temporal_feats = TEMPORAL_FEATURES_COMPLIANT,
        is_evader      = False,
        iterations     = ITERATIONS,
        seed           = RANDOM_SEED,
    )
    gan_vol.commit()
    logs_vol.commit()


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu="T4",
    cpu=4,
    memory=16_384,
    timeout=86_400,
    retries=0,
)
def train_evaders():
    import sys, os
    sys.path.insert(0, "/root")
    os.environ["MODAL_TASK_ID"] = "1"
    from config import SEED_EVADERS, GAN_EVADER_SEQ, RANDOM_SEED
    run_timegan(
        model_name     = "evaders",
        input_path     = SEED_EVADERS,
        output_path    = GAN_EVADER_SEQ,
        target_persons = 80_000,
        temporal_feats = TEMPORAL_FEATURES_EVADERS,
        is_evader      = True,
        iterations     = ITERATIONS,
        seed           = RANDOM_SEED,
    )
    gan_vol.commit()
    logs_vol.commit()


@app.local_entrypoint()
def main():
    print("Run one of:")
    print("  modal run 09_timegan_runner.py::train_compliant")
    print("  modal run 09_timegan_runner.py::train_evaders")