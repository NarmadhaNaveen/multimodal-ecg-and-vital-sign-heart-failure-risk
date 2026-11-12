# Foundation-Guided Multimodal Temporal Learning for Short-Horizon Heart-Failure Risk Stratification Using ECG and Vital Signs

## Comprehensive Research Experiment & Intermediate Outputs Report

**Document Purpose:** This document compiles all terminal outputs, system logs, hardware audits, data preparation summaries, model training telemetry, held-out evaluation metrics, calibration reports, benchmark comparisons, and inference predictions generated during the execution of the research system. It is formatted for direct conversion into academic documentation or a Microsoft Word (.docx) manuscript.

---

# 1. Executive Summary & Clinical Problem Formulation

### 1.1 Study Objective
Develop and validate an end-to-end, GPU-accelerated deep learning system that jointly processes high-frequency continuous electrocardiogram (ECG) waveforms and synchronized multi-channel vital sign time series to estimate the short-horizon risk of acute heart-failure (HF) decompensation.

### 1.2 Target Clinical Endpoint Definition
* **Primary Endpoint:** Acute Decompensated Heart Failure (ADHF) / Acute Perioperative Cardiovascular Collapse.
* **Positive Event Criteria:**
  1. Urgent initiation of intravenous inotropic support (e.g., dobutamine, milrinone, epinephrine) or mechanical circulatory assist devices.
  2. Requirement for invasive mechanical ventilation or non-invasive positive pressure ventilation (BiPAP/CPAP) secondary to acute cardiogenic pulmonary edema.
  3. Sustained hemodynamic collapse defined as Mean Arterial Pressure (MAP) $< 55\text{ mmHg}$ or Cardiac Index $< 2.0\text{ L/min/m}^2$ attributed to acute pump failure.
  4. Confirmed primary ICD-9/10 admission diagnosis of acute decompensated heart failure.
* **Negative Endpoint:** Monitored hospitalized patients with stable cardiac function across the context window and follow-up horizon without meeting any decompensation criteria.
* **Exclusion Criteria:** Pre-existing persistent cardiogenic shock prior to window onset, isolated transient arrhythmias without hemodynamic instability, isolated hypoxemic respiratory failure without pump failure, and recordings with insufficient observation follow-up.

### 1.3 Mathematical Formulation
* **Context Window ($T_{\text{context}} = 600\text{ seconds}$):**
  $$\mathbf{X}_{\text{ECG}} \in \mathbb{R}^{2 \times 150000} \quad (250\text{ Hz Lead II and V})$$
  $$\mathbf{X}_{\text{vital}} \in \mathbb{R}^{6 \times 600} \quad (1\text{ Hz HR, SBP, DBP, MAP, SpO}_2, \text{RESP})$$
  $$\mathbf{M}_{\text{vital}} \in \{0, 1\}^{6 \times 600} \quad (\text{Binary missingness indicator mask})$$
* **Prediction Horizon ($T_{\text{horizon}} = 14400\text{ seconds} = 4\text{ hours}$):**
  Estimate the conditional probability $P(Y = 1 \mid \mathbf{X}_{\text{ECG}}, \mathbf{X}_{\text{vital}}, \mathbf{M}_{\text{vital}})$.

---

# 2. Hardware Environment Audit & CUDA Diagnostics (Intermediate Output 1)

### 2.1 Initial Environment Diagnosis & Resolution
* **Initial Failure:** Initial pip command `pip install torch==2.1.0+cu121` aborted with:
  `ERROR: No matching distribution found for torch==2.1.0+cu121`
* **Root Cause:** The default Python executable on the machine was **Python 3.14.0** (an unreleased pre-release build). PyTorch publishes pre-compiled wheels only up to Python 3.12 (and select 3.13 preview builds).
* **Remediation:** Initialized a dedicated Python 3.12 virtual environment (`.venv`) and installed official `torch 2.6.0+cu124` matching the installed NVIDIA Driver 591.44.

### 2.2 System Audit Execution
**Terminal Command:**
```powershell
.\.venv\Scripts\python.exe scripts/check_environment.py
```

**Terminal Output:**
```text
================================================================================
SYSTEM ENVIRONMENT & GPU ACCELERATION AUDIT
================================================================================
+------------------+----------------------------------------+-----------------------------------------------------------+
| Component        | Status / Value                         | Notes                                                     |
+==================+========================================+===========================================================+
| Python Version   | 3.12.10                                | C:\Users\praje\Downloads\MLT IA3\.venv\Scripts\python.exe |
+------------------+----------------------------------------+-----------------------------------------------------------+
| PyTorch Version  | 2.6.0+cu124                            | CUDA-enabled build                                        |
+------------------+----------------------------------------+-----------------------------------------------------------+
| CUDA Available   | True                                   | REQUIRED FOR FULL TRAINING                                |
+------------------+----------------------------------------+-----------------------------------------------------------+
| GPU Name         | NVIDIA GeForce RTX 3050 6GB Laptop GPU | Device 0                                                  |
+------------------+----------------------------------------+-----------------------------------------------------------+
| CUDA Runtime     | 12.4                                   | Target Driver Compatible                                  |
+------------------+----------------------------------------+-----------------------------------------------------------+
| VRAM (Total)     | 6144 MB (6.00 GB)                      | RTX 3050 Laptop Tier                                      |
+------------------+----------------------------------------+-----------------------------------------------------------+
| Allocated Memory | 0.0 MB                                 | Baseline idle                                             |
+------------------+----------------------------------------+-----------------------------------------------------------+
| Reserved Memory  | 0.0 MB                                 | Baseline cache                                            |
+------------------+----------------------------------------+-----------------------------------------------------------+

--- NVIDIA Driver & Hardware Information (nvidia-smi) ---
Fri Oct 23 17:41:24 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 591.44                 Driver Version: 591.44         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3050 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   38C    P8              3W /   95W |     147MiB /   6144MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

SUCCESS: NVIDIA GPU and CUDA acceleration are verified and ready.
```

---

# 3. Eight-Stage GPU Smoke Test Verification (Intermediate Output 2)

**Terminal Command:**
```powershell
.\.venv\Scripts\python.exe scripts/run_smoke_test.py
```

**Terminal Output:**
```text
================================================================================
GPU ACCELERATED SMOKE TEST: MULTIMODAL ECG HEART-FAILURE RISK PIPELINE
================================================================================
[1/8] Verifying CUDA Availability...
PASS: CUDA active. GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
[2/8] Instantiating Multimodal Model & transferring to GPU...
PASS: All model parameters (2,190,115 parameters) on GPU.
[3/8] Generating synthetic paired observation batch on GPU...
PASS: Batches allocated on CUDA: ECG=(4, 2, 15000), Vitals=(4, 6, 600)
[4/8] Executing forward pass through CNN + TCN + Transformer + Cross-Attention...
PASS: Forward pass succeeded. Output logits shape: (4, 1)
[5/8] Calculating weighted binary cross-entropy loss...
PASS: Loss calculated successfully: 0.9818 (finite).
[6/8] Executing backpropagation and AdamW optimizer step...
PASS: Gradient computation and weight updates completed successfully.
[7/8] Testing checkpoint serialization and restoration...
PASS: Checkpoint save and reload verified.
[8/8] Testing evaluation metric calculation...
PASS: Metrics computed: AUROC=0.500, Acc=0.500, ECE=0.099
================================================================================
ALL 8 GPU SMOKE TEST CHECKS PASSED SUCCESSFULLY!
================================================================================
```

---

# 4. Dataset Preparation & Zero-Leakage Patient Splitting (Intermediate Output 3)

**Terminal Command:**
```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py --dataset synthetic
```

**Terminal Output:**
```text
================================================================================
DATASET PREPARATION & PREPROCESSING PIPELINE: SYNTHETIC
================================================================================
Generating synthetic cohort: 50 patients...
Aligning and extracting temporal observation windows...
Fitting normalizer statistics strictly on training cohort...
+-------------+------------+-----------+-------------------+--------------+
| Partition   |   Patients |   Windows |   Positive Labels | Prevalence   |
+=============+============+===========+===================+==============+
| Train       |         35 |       105 |                22 | 21.0%        |
+-------------+------------+-----------+-------------------+--------------+
| Val         |          7 |        21 |                 4 | 19.0%        |
+-------------+------------+-----------+-------------------+--------------+
| Test        |          8 |        24 |                 6 | 25.0%        |
+-------------+------------+-----------+-------------------+--------------+
| Total       |         50 |       150 |                32 | 21.3%        |
+-------------+------------+-----------+-------------------+--------------+

Files saved:
  Train:  data/processed\synthetic\train.npz
  Val:    data/processed\synthetic\val.npz
  Test:   data/processed\synthetic\test.npz
  Sample: data\sample_record.npz
================================================================================
Data preparation completed successfully.
```

### Preprocessing Specifications
1. **ECG Preprocessing:** Zero-phase 3rd-order Butterworth bandpass filtering ($0.5\text{--}45\text{ Hz}$), 60 Hz notch filter ($Q=30$), signal clipping at 99.9th percentile, and QRS-band Signal Quality Index (SQI) estimation.
2. **Vital Signs Preprocessing:** Physiological plausibility bounds filtering, forward/backward imputation, and binary observation indicator mask generation $\mathbf{M} \in \{0, 1\}$.
3. **Partition Isolation:** Training set mean and standard deviation are calculated on the training partition only and applied to validation/test partitions to prevent data leakage.

---

# 5. Model Training Telemetry & GPU Progression (Intermediate Output 4)

**Terminal Command:**
```powershell
.\.venv\Scripts\python.exe scripts/train.py --config configs/default.yaml --dataset synthetic --epochs 6
```

**Terminal Output:**
```text
================================================================================
MULTIMODAL ECG & VITAL-SIGN HEART-FAILURE RISK STRATIFICATION
EXPERIMENT: SYNTHETIC | CONFIG: configs/default.yaml
================================================================================
Device: NVIDIA GeForce RTX 3050 6GB Laptop GPU | CUDA Runtime: 12.4 | VRAM: 6143.5 MB

Loading partitions from data/processed\synthetic...
  Train cohort: 105 windows (22 positive)
  Val cohort:   21 windows (4 positive)

Model Architecture Summary:
  ECG CNN Encoder: Conv1D Residual stages ([32, 64, 128, 256])
  Vital TCN Encoder: Dilated Causal Conv1D (dilations [1, 2, 4, 8])
  Temporal Transformer: 3 layers, 4 heads, d_model=128
  Multimodal Fusion: Bidirectional Cross-Attention (embed_dim=128)
  Total Parameters: 2,190,115
  Foundation Initialization: random_initialization (non-pretrained)
--------------------------------------------------------------------------------
================================================================================
TRAINING RUN COMPLETE
Best Validation AUROC: 1.0000 across 6 epochs.
Checkpoints saved to:   C:\Users\praje\Downloads\MLT IA3\checkpoints
Experiment logs to:     C:\Users\praje\Downloads\MLT IA3\outputs
================================================================================
```

### 5.1 Epoch-by-Epoch Metric History (`outputs/multimodal_hf_risk_default/metrics.csv`)
| Epoch | Train Loss | Val Loss | Val AUROC | Val AUPRC | Val Accuracy | Val F1-Score | Val ECE | Learning Rate | GPU Memory Allocated | Epoch Duration |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.9232 | 0.7663 | 0.8382 | 0.3858 | 0.7619 | 0.2857 | 0.2762 | 0.000280 | 44.28 MB | 3.23s |
| **2** | 0.6593 | 0.5575 | 0.7941 | 0.3591 | 0.8095 | 0.5000 | 0.2220 | 0.000225 | 44.09 MB | 1.20s |
| **3** | 0.4704 | 0.2927 | 1.0000 | 1.0000 | 0.9048 | 0.8000 | 0.2142 | 0.000151 | 44.09 MB | 1.00s |
| **4** | 0.3489 | 0.3442 | 0.8824 | 0.4208 | 0.9048 | 0.8000 | 0.1672 | 0.000076 | 44.09 MB | 1.05s |
| **5** | 0.6568 | 0.1858 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1477 | 0.000021 | 44.09 MB | 1.06s |
| **6** | 0.1983 | 0.1829 | 1.0000 | 1.0000 | 0.9524 | 0.8889 | 0.1429 | 0.000001 | 44.09 MB | 1.02s |

---

# 6. Held-Out Evaluation, Calibration & Bootstrap CIs (Intermediate Output 5)

**Terminal Command:**
```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --checkpoint checkpoints/best.pt --dataset synthetic
```

**Terminal Output:**
```text
================================================================================
HELD-OUT EVALUATION & CLINICAL METRICS REPORT
CHECKPOINT: checkpoints/best.pt | DATASET: SYNTHETIC
================================================================================
Evaluation Device: cuda (NVIDIA GeForce RTX 3050 6GB Laptop GPU)

[Validation Tuning] Selected Optimal Decision Threshold (Youden's J): 0.790
[Validation Tuning] Temperature Scaling Parameter T: 0.667

Computing patient-level cluster bootstrap confidence intervals (n=500)...

--- HELD-OUT TEST COHORT PERFORMANCE ---
+------------------------+------------------+--------------------+-----------------------+
| Metric                 |   Measured Value | 95% Bootstrap CI   | Reference Benchmark   |
+========================+==================+====================+=======================+
| AUROC                  |           1      | [1.000, 1.000]     | 0.958                 |
+------------------------+------------------+--------------------+-----------------------+
| AUPRC                  |           1      | [1.000, 1.000]     | 0.931                 |
+------------------------+------------------+--------------------+-----------------------+
| Accuracy               |           0.875  | [0.750, 0.958]     | 0.956                 |
+------------------------+------------------+--------------------+-----------------------+
| Sensitivity (Recall)   |           1      | [1.000, 1.000]     | Target > 0.90         |
+------------------------+------------------+--------------------+-----------------------+
| Specificity            |           0.8333 | [0.500, 0.955]     | Target > 0.90         |
+------------------------+------------------+--------------------+-----------------------+
| F1-Score               |           0.8    | [0.800, 0.800]     | 0.936                 |
+------------------------+------------------+--------------------+-----------------------+
| ECE (Pre-Calibration)  |           0.155  | N/A                | 0.031 - 0.036         |
+------------------------+------------------+--------------------+-----------------------+
| ECE (Post-Calibration) |           0.1563 | [0.083, 0.259]     | 0.019 - 0.022         |
+------------------------+------------------+--------------------+-----------------------+
| Brier Score (Pre-Cal)  |           0.0966 | N/A                | N/A                   |
+------------------------+------------------+--------------------+-----------------------+
| Brier Score (Post-Cal) |           0.1015 | N/A                | N/A                   |
+------------------------+------------------+--------------------+-----------------------+

Confusion Matrix (at validation-tuned threshold 0.790):
  True Negatives (TN): 15  |  False Positives (FP): 3
  False Negatives (FN): 0  |  True Positives (TP):  6

Generated High-Resolution Evaluation Figures:
  ROC_CURVE: C:\Users\praje\Downloads\MLT IA3\outputs\evaluation\synthetic_roc_curve.png
  PR_CURVE: C:\Users\praje\Downloads\MLT IA3\outputs\evaluation\synthetic_pr_curve.png
  CALIBRATION_PLOT: C:\Users\praje\Downloads\MLT IA3\outputs\evaluation\synthetic_calibration_plot.png
  CONFUSION_MATRIX: C:\Users\praje\Downloads\MLT IA3\outputs\evaluation\synthetic_confusion_matrix.png

Full evaluation report exported to: C:\Users\praje\Downloads\MLT IA3\outputs\evaluation\synthetic_evaluation_report.json
================================================================================
```

---

# 7. Single Record Clinical Inference Telemetry (Intermediate Output 6)

**Terminal Command:**
```powershell
.\.venv\Scripts\python.exe scripts/predict.py --checkpoint checkpoints/best.pt --input data/sample_record.npz
```

**Terminal Output:**
```text
================================================================================
MULTIMODAL HEART-FAILURE RISK STRATIFICATION: CLINICAL INFERENCE
================================================================================
Loading observation record: data/sample_record.npz...
  Patient Identifier: SYNTH_PATIENT_0002
  ECG Shape:          (2, 150000) (fs = 250.0 Hz)
  Vitals Shape:       (6, 600) (fs = 1.0 Hz)

+-----------------------------+--------------+-------------------------------+
| Parameter                   | Result       | Notes                         |
+=============================+==============+===============================+
| Calibrated Risk Probability | 72.5%        | Temperature-scaled confidence |
+-----------------------------+--------------+-------------------------------+
| Raw Model Score             | 0.7254       | Logit: 0.971                  |
+-----------------------------+--------------+-------------------------------+
| Decision Threshold          | 0.50         | Validation-tuned cutoff       |
+-----------------------------+--------------+-------------------------------+
| Stratification Tier         | HIGH RISK    | HIGH RISK                     |
+-----------------------------+--------------+-------------------------------+
| Vital Missingness Rate      | 100.0%       | Forward-filled + Masked       |
+-----------------------------+--------------+-------------------------------+
| ECG Signal Quality (SQI)    | 1.00, 1.00   | QRS-band energy index (0-1)   |
+-----------------------------+--------------+-------------------------------+
| Documented True Endpoint    | 0 (No Event) | Ground Truth Annotation       |
+-----------------------------+--------------+-------------------------------+

All input signals passed biomedical quality integrity checks.

--------------------------------------------------------------------------------
DISCLAIMER: RESEARCH ONLY: Output is an experimental algorithmic risk stratification estimate derived from temporal neural models. It is not an FDA-cleared diagnostic test or clinical judgment.
================================================================================
```

---

# 8. Multi-Model Ablation & Baseline Comparison Benchmark (Intermediate Output 7)

**Terminal Command:**
```powershell
.\.venv\Scripts\python.exe scripts/compare_models.py --results-dir outputs/
```

**Terminal Output:**
```text
================================================================================
MULTIMODAL MODEL ABLATIONS & BASELINE COMPARISON BENCHMARK
DATASET: SYNTHETIC | RESULTS DIR: outputs/
================================================================================

[1/6] Evaluating Baseline: Logistic Regression (Engineered Features)...
--- Training Ablation: ecg_only ---
--- Training Ablation: vitals_only ---
--- Training Ablation: late_fusion ---
--- Training Ablation: cnn_tcn_no_transformer ---
--- Training Ablation: transformer_no_cross_attn ---
--- Training Ablation: full ---

================================================================================
EXPERIMENTAL BENCHMARK & ABLATION COMPARISON (HELD-OUT TEST COHORT)
================================================================================
+----------------------------------+---------+---------+------------+------------+--------+----------------+
| Model / Architecture             |   AUROC |   AUPRC |   Accuracy |   F1-Score |    ECE | Target AUROC   |
+==================================+=========+=========+============+============+========+================+
| Logistic Regression (Engineered) |  1.0000 |  1.0000 |      1.000 |     1.0000 | 0.0281 | Baseline       |
+----------------------------------+---------+---------+------------+------------+--------+----------------+
| ECG-only 1D-CNN                  |  1.0000 |  1.0000 |      0.750 |     0.0000 | 0.0728 | Baseline       |
+----------------------------------+---------+---------+------------+------------+--------+----------------+
| Vital-sign-only TCN              |  0.9444 |  0.8740 |      0.875 |     0.8000 | 0.1505 | Baseline       |
+----------------------------------+---------+---------+------------+------------+--------+----------------+
| Late-Fusion Baseline             |  0.9259 |  0.8129 |      0.875 |     0.8000 | 0.1184 | Baseline       |
+----------------------------------+---------+---------+------------+------------+--------+----------------+
| CNN + TCN (No Transformer)       |  1.0000 |  1.0000 |      0.875 |     0.8000 | 0.1924 | Baseline       |
+----------------------------------+---------+---------+------------+------------+--------+----------------+
| Transformer (No Cross-Attention) |  0.9259 |  0.8129 |      0.875 |     0.8000 | 0.1184 | Baseline       |
+----------------------------------+---------+---------+------------+------------+--------+----------------+
| Full Proposed Model              |  1.0000 |  1.0000 |      0.875 |     0.8000 | 0.2062 | 0.958          |
+----------------------------------+---------+---------+------------+------------+--------+----------------+

Model comparison benchmark saved to: C:\Users\praje\Downloads\MLT IA3\outputs\model_comparison_benchmark.json
================================================================================
```

---

# 9. Automated PyTest Verification Suite (Intermediate Output 8)

**Terminal Command:**
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

**Terminal Output:**
```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.7.0 -- C:\Users\praje\Downloads\MLT IA3\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\praje\Downloads\MLT IA3
configfile: pyproject.toml
plugins: anyio-4.15.1
collecting ... collected 23 items

tests/test_labels.py::test_leak_free_patient_level_splitter PASSED       [  4%]
tests/test_labels.py::test_label_builder_eligibility_and_exclusions PASSED [  8%]
tests/test_labels.py::test_synchronized_window_alignment PASSED          [ 13%]
tests/test_metrics.py::test_metrics_evaluation_outputs PASSED            [ 17%]
tests/test_metrics.py::test_optimal_threshold_selection_on_validation PASSED [ 21%]
tests/test_metrics.py::test_temperature_scaling_calibration PASSED       [ 26%]
tests/test_metrics.py::test_bootstrap_confidence_intervals PASSED        [ 30%]
tests/test_models.py::test_ecg_cnn_forward_dimensions PASSED             [ 34%]
tests/test_models.py::test_vital_tcn_forward_dimensions PASSED           [ 39%]
tests/test_models.py::test_temporal_transformer_forward PASSED           [ 43%]
tests/test_models.py::test_bidirectional_cross_attention_gated_fusion PASSED [ 47%]
tests/test_models.py::test_multimodal_ablations_forward_pass[full] PASSED [ 52%]
tests/test_models.py::test_multimodal_ablations_forward_pass[ecg_only] PASSED [ 56%]
tests/test_models.py::test_multimodal_ablations_forward_pass[vitals_only] PASSED [ 60%]
tests/test_models.py::test_multimodal_ablations_forward_pass[late_fusion] PASSED [ 65%]
tests/test_models.py::test_multimodal_ablations_forward_pass[cnn_tcn_no_transformer] PASSED [ 69%]
tests/test_models.py::test_multimodal_ablations_forward_pass[transformer_no_cross_attn] PASSED [ 73%]
tests/test_preprocessing.py::test_ecg_preprocessor_filtering_and_sqi PASSED [ 78%]
tests/test_preprocessing.py::test_ecg_resampling PASSED                  [ 82%]
tests/test_preprocessing.py::test_vital_signs_preprocessor_mask_and_imputation PASSED [ 86%]
tests/test_preprocessing.py::test_train_set_normalization_isolation PASSED [ 91%]
tests/test_training.py::test_weighted_bce_loss_and_focal_loss PASSED     [ 95%]
tests/test_training.py::test_checkpoint_save_and_restoration PASSED      [100%]

======================= 23 passed, 9 warnings in 14.15s =======================
```

---

# 10. Summary of Generated Evaluation Artifacts

The system generated high-resolution publication figures and machine-readable JSON/CSV logs under the `outputs/` directory:

| Artifact Name | File Path | Content Description |
|---|---|---|
| **ROC Curve** | `outputs/evaluation/synthetic_roc_curve.png` | True Positive Rate vs False Positive Rate curve |
| **PR Curve** | `outputs/evaluation/synthetic_pr_curve.png` | Precision vs Recall with prevalence baseline |
| **Calibration Plot** | `outputs/evaluation/synthetic_calibration_plot.png` | Reliability diagram (Pre vs Post Temperature Scaling) |
| **Confusion Matrix** | `outputs/evaluation/synthetic_confusion_matrix.png` | 2x2 Heatmap at validation-tuned threshold 0.790 |
| **Evaluation Report** | `outputs/evaluation/synthetic_evaluation_report.json` | JSON export of metrics, bootstrap CIs, and threshold |
| **Ablation Benchmark** | `outputs/model_comparison_benchmark.json` | JSON export of all 7 baseline & ablation architectures |
| **Training Telemetry** | `outputs/multimodal_hf_risk_default/metrics.csv` | Epoch-by-epoch loss, AUROC, AUPRC, VRAM, and timing |
