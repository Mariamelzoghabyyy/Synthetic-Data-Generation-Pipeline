# Synthetic Tax Evasion Panel Data Generation Framework
## Hybrid Generative Modeling + Forensic Feature Engineering (2019–2025)

---

## Overview

This project implements a high-fidelity synthetic longitudinal tax evasion dataset generation framework designed to overcome the structural unavailability of granular taxpayer data due to statutory confidentiality constraints (e.g., IRC § 6103).

The system produces a multi-state, seven-year panel dataset (2019–2025) incorporating:

- Empirical distribution anchoring from public statistical sources
- Hybrid generative modeling (CTAB‑GAN + TimeGAN)
- 15 IRS Tax Gap–aligned evasion mechanisms
- Lifecycle-preserving stratified down-sampling
- Six-phase forensic feature engineering pipeline
- Composite risk scoring architecture

The final benchmark dataset contains:

- ~50,000 records
- 15,000+ longitudinal taxpayer timelines
- 278 engineered features
- State-specific fraud rates (16%–22%)
- Composite risk ROC-AUC: 0.764

---

## Research Motivation

Supervised tax evasion detection research is constrained by the legal inaccessibility of individual-level tax return data. Public releases (e.g., SOI tables) are aggregated and unsuitable for machine learning classification.

This framework addresses that bottleneck by constructing a statistically anchored, demographically realistic, longitudinal synthetic panel dataset capable of supporting advanced fraud detection research.

The objective is not to replicate confidential IRS microdata, but to simulate structurally faithful financial behavior patterns calibrated to:

- Census ACS PUMS
- BLS Occupational Employment Statistics
- Zillow Home Value Index
- IRS Schedule C Statistics of Income
- IRS Tax Gap composition data

---

## System Architecture
[ Distribution Anchoring ]
↓
[ Person + Business Generation ]
↓
[ Panel Simulation Engine ]
↓
[ Evasion Injection (15 types) ]
↓
[ CTAB‑GAN (Tabular Fidelity) ]
[ TimeGAN (Temporal Coherence) ]
↓
[ Merge + Null Policy Enforcement ]
↓
[ Lifecycle Stratified Down-Sampling ]
↓
[ 6-Phase Feature Engineering ]
↓
[ Composite Risk Scoring ]

text


---

## Dataset Characteristics

| Metric | Value |
|--------|--------|
| Time Span | 2019–2025 |
| Raw Records | ~5.2 million |
| Downsampled Records | 50,242 |
| Unique Taxpayers | 15,264 |
| Engineered Features | 278 |
| Fraud Rate | 16%–22% (state-specific) |
| Evasion Types | 15 |
| Composite Risk ROC-AUC | 0.764 |

---

## Generative Modeling Components

### CTAB-GAN

Addresses tabular mixed-type generation challenges via:

- Mode-specific normalization (VGM encoding)
- Conditional categorical training
- Auxiliary LightGBM consistency signal

Improves marginal and joint distribution realism of income, deduction, and business fields.

---

### TimeGAN

Ensures longitudinal sequence coherence using:

- LSTM autoencoder reconstruction loss
- Supervised temporal loss
- Adversarial discriminator loss

Preserves realistic year-over-year income and compliance trajectories.

---

## Evasion Injection Framework

Fifteen behaviorally grounded evasion mechanisms aligned with IRS Tax Gap categories:

- Cash revenue suppression
- Fictitious deduction inflation
- 1099 income omission
- Expense recharacterization
- Offshore account concealment
- Worker misclassification
- Rental income omission
- Cryptocurrency gain suppression
- False dependent claims
- Pension underreporting
- Hobby loss abuse
- And others

Each mechanism modifies records through stochastic but structurally consistent transformations.

Evasion cannot be detected through single-field threshold rules; detection requires multivariate relational analysis.

---

## Six-Phase Feature Engineering Pipeline

The pipeline transforms 106 raw columns into 278 engineered features.

### Phase 1 — Demographic Context
- Age brackets
- Filing status normalization
- Income bracket encoding
- Zone-state interactions

### Phase 2 — Behavioral Ratios
- Shannon entropy of income streams
- Lifestyle mismatch signals
- Tax rate gap
- Deduction aggression score
- COGS-to-receipts ratio
- Crypto loss shield ratio

### Phase 3 — Peer Group Z-Scores
Reference group:
State × Industry × Taxpayer Type

text

Features standardized relative to peers.

### Phase 4 — Forensic Digit Analysis
- Benford’s Law χ² statistic
- Rounding tendency ratio
- Terminal digit uniformity deviation

### Phase 5 — Structural Contradictions
- W-2 > AGI flag
- Deductions exceed AGI
- Self-employment tax omission
- FBAR violation
- Persistent hobby loss

### Phase 6 — Composite Risk Aggregation

Weighted block scoring:

| Risk Block | Weight |
|------------|--------|
| IRS Risk | 0.50 |
| Bank Deposit | 0.20 |
| Peer Z-Score | 0.12 |
| Network Risk | 0.10 |
| Withholding | 0.05 |
| Forensic | 0.03 |

Final master_fraud_propensity includes interaction boosts.

Standalone ROC-AUC: **0.764**

---

## Validation Framework

Each state output validated across:

- Fraud rate compliance (±0.5%)
- Temporal continuity (no lifecycle gaps)
- Conditional null enforcement
- Duplicate detection
- Schema integrity
- Taxpayer type preservation
- Macro shock consistency
- Feature signal separability

---

## Machine Learning Implications

This dataset supports:

- Gradient boosting models
- Deep neural networks
- Temporal LSTM / GRU architectures
- Transformer-based sequence models
- Multi-task jurisdictional classifiers
- Domain adaptation research

The multivariate structure prevents trivial threshold classification and requires relational reasoning.

---

## Ethical Considerations

- No real taxpayer data is used.
- All distributions are anchored to public aggregate statistics.
- Synthetic data cannot be reverse-engineered to identify real individuals.
- The system is intended for academic research and fraud detection method development.

---

## Limitations

- Evasion intensity calibration cannot be validated against confidential IRS microdata.
- GAN architectures approximate distributions but do not guarantee full joint fidelity.
- Composite risk weights calibrated on synthetic ground truth.
- Corporate, partnership, and international tax forms not modeled.

---

## Reproducibility

- Random seeds fixed for deterministic generation
- CTAB-GAN training epochs: configurable
- TimeGAN embedding dimension: configurable
- Hardware: CPU-compatible; GPU recommended for GAN stages

Full pipeline scripts provided.

Large generated datasets are not included due to size constraints.

---

## Citation

If using this framework in research:
Synthetic Tax Evasion Panel Data Generation and Forensic Feature Engineering Framework (2025)

text


---

## License

MIT License

---
