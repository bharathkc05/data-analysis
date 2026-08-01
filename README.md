# ECG-Deli Feature Quality Assessment & Comparative Analysis Pipeline

> **Objective:** Systematically evaluate the quality, consistency, and reliability of ECG features extracted by the ECG-Deli library. The pipeline supports two operational modes:
> 1. **Extraction Validation** — Compare PTB-XL Original features against PTB-XL Re-extracted features using paired, record-level agreement metrics.
> 2. **Cross-Dataset Comparison** — Compare MIMIC-IV extracted features against PTB-XL Original as a trusted reference benchmark.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Operational Modes](#2-operational-modes)
3. [Architecture & Directory Structure](#3-architecture--directory-structure)
4. [Datasets](#4-datasets)
5. [Pipeline Stages](#5-pipeline-stages)
   - [Stage 0 — Data Loading, Intersection & Record Matching](#stage-0--data-loading-intersection--record-matching)
   - [Stage 1 — Dataset Overview](#stage-1--dataset-overview)
   - [Stage 2 — Missingness Analysis](#stage-2--missingness-analysis)
   - [Stage 3 — Distribution Analysis](#stage-3--distribution-analysis)
   - [Stage 4 — Plausibility Analysis](#stage-4--plausibility-analysis)
   - [Stage 5 — Extreme Value Extraction](#stage-5--extreme-value-extraction)
   - [Stage 6 — Normality Assessment](#stage-6--normality-assessment)
   - [Stage 7 — Outlier Detection](#stage-7--outlier-detection)
   - [Stage 8 — Correlation Analysis](#stage-8--correlation-analysis)
   - [Stage 9 — Feature Redundancy Analysis](#stage-9--feature-redundancy-analysis)
   - [Stage 10 — Dimensionality Reduction](#stage-10--dimensionality-reduction)
   - [Stage 11 — Feature Relationship Analysis](#stage-11--feature-relationship-analysis)
   - [Stage 12 — Batch Effect Detection](#stage-12--batch-effect-detection)
   - [Stage 13 — Comparative Statistical Analysis](#stage-13--comparative-statistical-analysis)
   - [Stage 14 — Similarity Scoring](#stage-14--similarity-scoring)
   - [Stage 15 — Paired Validation (Extraction Validation Mode Only)](#stage-15--paired-validation-extraction-validation-mode-only)
   - [Stage 16 — Quality Scoring](#stage-16--quality-scoring)
   - [Stage 17 — Report Generation](#stage-17--report-generation)
   - [Stage 18 — Publication Export](#stage-18--publication-export)
   - [Stage 19 — Automated Conclusions](#stage-19--automated-conclusions)
6. [Output Catalog](#6-output-catalog)
7. [Key Findings (Run: 2026-07-30 — Extraction Validation)](#7-key-findings-run-2026-07-30--extraction-validation)
8. [Configuration Reference](#8-configuration-reference)
9. [Setup & Execution](#9-setup--execution)

---

## 1. Project Overview

This pipeline performs a **multi-stage data quality assessment** comparing ECG features extracted from two datasets using the same feature extraction library (ECG-Deli). The analysis covers:

- **Univariate analysis** — Missingness, distribution statistics, plausibility, extremes, normality
- **Multivariate analysis** — Outlier detection, correlations, feature redundancy, dimensionality reduction, batch effects
- **Comparative analysis** — Statistical tests, effect sizes, distribution overlap, similarity scoring
- **Paired validation** *(extraction validation mode only)* — Record-level ICC, MAE, MAPE, Bland-Altman analysis, paired t-tests, Wilcoxon signed-rank tests
- **Reporting** — Quality scores, HTML dashboard, publication-ready tables (CSV + LaTeX + XLSX), automated conclusions

---

## 2. Operational Modes

The pipeline's behavior is controlled by the `mode` key in `config.yaml`:

| Mode | `mode` Value | Dataset 1 | Dataset 2 | Paired Validation | Output Subdirectory |
|---|---|---|---|---|---|
| **Extraction Validation** | `extraction_validation` | PTB-XL Original | PTB-XL Re-extracted | ✅ Yes (records matched by `ecg_id`) | `validation/` |
| **Cross-Dataset Comparison** | `cross_dataset_comparison` | MIMIC-IV | PTB-XL Original | ❌ No | `comparison/` |

**Key behavioral differences by mode:**

- **Record Matching:** In extraction validation mode, the `DataManager` performs an inner join on `ecg_id` to produce row-aligned `df1_matched` / `df2_matched` DataFrames. This enables paired statistical tests (Stage 15).
- **Plausibility Direction:** In extraction validation mode, reference bounds come from Dataset 1 (PTB-XL Original) and target values are Dataset 2 (PTB-XL Re-extracted). In cross-dataset mode, bounds come from PTB-XL and the target is MIMIC-IV.
- **Output Isolation:** All outputs (tables, figures, reports, publication tables) are written to mode-specific subdirectories (`validation/` or `comparison/`), so both modes can be run without overwriting each other.

---

## 3. Architecture & Directory Structure

```
new_analysis_pipeline/
├── main.py                          # Pipeline orchestrator (entry point, argparse)
├── config.yaml                      # Mode selection, all paths, thresholds, plot settings
├── requirements.txt                 # Python dependencies (13 packages)
├── pipeline.log                     # Execution log from last run (auto-overwritten)
│
├── core/                            # Infrastructure layer
│   ├── config.py                    # Config singleton — loads YAML, selects mode, sets up directories
│   └── logger.py                    # Centralized logging (stdout + pipeline.log, timestamped)
│
├── data/                            # Data management layer
│   └── data_manager.py              # DataManager — loading, record matching, intersection,
│                                    #   imputation, scaling, lazy-cached combined datasets
│
├── analysis/                        # Analysis modules
│   ├── univariate.py                # Missingness, distributions, plausibility, extremes, normality
│   ├── multivariate.py              # Outliers, correlations, redundancy, PCA/UMAP, batch effects
│   └── comparative.py               # Dataset comparisons, similarity scoring, paired validation
│                                    #   (ICC, Bland-Altman, paired t-test, Wilcoxon)
│
├── visualization/                   # Plotting utilities
│   └── plots.py                     # Plotter class — all chart types including
│                                    #   Bland-Altman and paired scatter plots
│
├── reporting/                       # Report generation
│   └── report_generator.py          # Overview, quality scores, mode-aware HTML report,
│                                    #   LaTeX/XLSX export, paired validation conclusions
│
└── outputs/                         # All generated artifacts (mode-partitioned)
    ├── tables/
    │   ├── *.csv                    #   25 tables (cross-dataset comparison run)
    │   └── validation/              #   19 tables (extraction validation run)
    ├── figures/
    │   ├── batch_effects/{validation,comparison}/
    │   ├── comparisons/{validation,comparison}/
    │   ├── correlations/{validation,comparison}/
    │   ├── dimensionality/{validation,comparison}/
    │   ├── distributions/{validation,comparison}/
    │   ├── missingness/{validation,comparison}/
    │   ├── normality/{validation,comparison}/
    │   ├── outliers/{validation,comparison}/
    │   ├── relationships/{validation,comparison}/
    │   └── paired_validation/       #   Bland-Altman & paired scatter (validation mode only)
    ├── report/
    │   ├── validation/              #   ECG_Report_validation.html + automated_conclusions.md
    │   └── comparison/              #   ECG_Report_comparison.html + automated_conclusions.md
    └── publication/
        ├── validation/              #   Tables 1–7 (CSV + LaTeX + XLSX)
        └── comparison/              #   Tables 1–5 (CSV + LaTeX + XLSX)
```

**Design patterns:**
- **Mode Singleton:** `core/config.py` instantiates a global `settings` object that determines dataset names, paths, colors, output directories, and whether paired validation runs — all driven by the `mode` key in `config.yaml`.
- **Lazy Caching:** `DataManager` provides lazy-computed, cached datasets (`get_combined_imputed_data()`, `get_combined_scaled_data()`), so heavy computations (median imputation, standard scaling) happen at most once.
- **Record Matching:** When `is_paired_validation=True`, `DataManager._match_records()` performs an inner join on `ecg_id`, producing row-aligned DataFrames with a detailed matching report saved to CSV.

---

## 4. Datasets

### Extraction Validation Mode (PTB-XL Original vs PTB-XL Re-extracted)

| Property | PTB-XL Original (Dataset 1) | PTB-XL Re-extracted (Dataset 2) |
|---|---|---|
| **Total ECG Records** | 21,799 | 21,796 |
| **Original Features** | 532 | 160 |
| **Overlapping Features** | 160 | 160 |
| **Successfully Matched** | 21,796 (99.99%) | 21,796 |
| **Unmatched Records** | 3 (only in Original) | 0 |
| **Duplicate IDs** | 0 | 0 |

### Cross-Dataset Comparison Mode (MIMIC-IV vs PTB-XL Original)

| Property | MIMIC-IV (Dataset 1) | PTB-XL Original (Dataset 2) |
|---|---|---|
| **Total ECG Records** | 407,504 | 21,799 |
| **Original Features** | 160 | 532 |
| **Overlapping Features** | 159 | 159 |
| **Memory Usage** | 497.44 MB | 26.61 MB |
| **Duplicate Subject IDs** | 307,084 (multiple ECGs per patient) | 0 |
| **Duplicate Rows** | 58 | 0 |

**Feature intersection:** Only the features common to both datasets are analyzed. The ID column (`ecg_id` / `SubjectID`) is preserved but excluded from numeric analyses.

---

## 5. Pipeline Stages

### Stage 0 — Data Loading, Intersection & Record Matching

| | |
|---|---|
| **Module** | `data/data_manager.py` → `DataManager.load_and_intersect()` |
| **Purpose** | Load both CSVs, normalize ID columns, identify overlapping columns, optionally match records |
| **Key Logic** | Set intersection of column names; auto-detect and rename ID column (`SubjectID` → `ecg_id`); drop exact duplicate rows |
| **Record Matching** | In `extraction_validation` mode, `_match_records()` performs an inner join on `ecg_id`, producing aligned `df1_matched` / `df2_matched` DataFrames and a `record_matching_report.csv` |
| **Output** | In-memory DataFrames filtered to overlapping features + ID column; optional `tables/<mode>/record_matching_report.csv` |

The `DataManager` also provides two derived datasets on demand:
- **Combined Imputed** — Median-imputed concatenation of both datasets (for multivariate analyses)
- **Combined Scaled** — StandardScaler-transformed version (for PCA/UMAP/classification)

---

### Stage 1 — Dataset Overview

| | |
|---|---|
| **Module** | `reporting/report_generator.py` → `ReportGenerator.generate_overview()` |
| **Purpose** | Generate high-level statistics about each dataset |
| **Checks Performed** | Record counts, feature counts, memory usage, duplicate IDs, duplicate rows, constant/near-constant columns, all-missing columns |
| **Output Files** | `tables/<mode>/overview.csv`, `tables/<mode>/feature_summary.csv` |

---

### Stage 2 — Missingness Analysis

| | |
|---|---|
| **Module** | `analysis/univariate.py` → `UnivariateAnalysis.analyze_missingness()` |
| **Purpose** | Quantify and visualize missing data patterns in both datasets |
| **Metrics** | Missing count and percentage per feature per dataset |
| **Visualizations** | Missingness matrix (via `missingno`), bar plot, heatmap (if correlated missingness exists) |
| **Output Files** | `tables/<mode>/missingness_report.csv`, `figures/missingness/<mode>/missing_matrix_*.png`, `figures/missingness/<mode>/missing_bar_*.png` |
| **Thresholds** | Warning at >5% missing; Critical at >10% missing (configurable) |

---

### Stage 3 — Distribution Analysis

| | |
|---|---|
| **Module** | `analysis/univariate.py` → `UnivariateAnalysis.analyze_distributions()` |
| **Purpose** | Compute descriptive statistics and generate comparative distribution visualizations |
| **Statistics Computed** | Mean, Median, SD, Min, Max, IQR, MAD, Variance, Skewness, Kurtosis, Percentiles (p01, p05, p25, p50, p75, p95, p99) |
| **Visualizations** | 4-panel plot per feature: Histogram+KDE overlay, Boxplot, Violin plot, ECDF |
| **Output Files** | `tables/<mode>/distribution_statistics.csv`, `figures/distributions/<mode>/<feature>_distribution.png` (one per feature) |

---

### Stage 4 — Plausibility Analysis

| | |
|---|---|
| **Module** | `analysis/univariate.py` → `UnivariateAnalysis.analyze_plausibility()` |
| **Purpose** | Check whether target dataset values fall within physiologically plausible ranges defined by the reference dataset |
| **Mode Behavior** | **Extraction validation:** Reference = PTB-XL Original, Target = PTB-XL Re-extracted (values outside bounds indicate extraction differences). **Cross-dataset:** Reference = PTB-XL, Target = MIMIC-IV (values outside bounds may reflect population differences). |
| **Method** | Reference bounds = Reference dataset's configurable percentile range (default 1st–99th) per feature. Count target values outside these bounds. |
| **Output Columns** | Feature, Lower/Upper Bound, Number & Percentage outside range, Example ECG IDs |
| **Output Files** | `tables/<mode>/plausibility_report.csv` |

---

### Stage 5 — Extreme Value Extraction

| | |
|---|---|
| **Module** | `analysis/univariate.py` → `UnivariateAnalysis.extract_extremes()` |
| **Purpose** | Identify the top-N and bottom-N values for each feature to support manual review |
| **Parameters** | `n=20` (configurable via `thresholds.extreme_n`; top 20 largest, bottom 20 smallest per feature per dataset) |
| **Output Columns** | Dataset, Feature, Extreme Type (Top/Bottom), Values, ECG IDs |
| **Output Files** | `tables/<mode>/feature_extremes.csv` |

---

### Stage 6 — Normality Assessment

| | |
|---|---|
| **Module** | `analysis/univariate.py` → `UnivariateAnalysis.assess_normality()` |
| **Purpose** | Determine whether each feature's distribution is approximately Gaussian |

**Tests Applied:**

| Test | Details |
|---|---|
| Shapiro-Wilk | Subsampled to 5,000 records |
| Anderson-Darling | 5% significance level |
| D'Agostino K² | Full sample |

**Decision Rule (updated):**
- **Large samples (N > 5,000):** Uses *practical normality* criterion — feature is approximately Gaussian if |skewness| < 2.0 AND |kurtosis| < 7.0 (thresholds configurable).
- **Small samples (N ≤ 5,000):** Uses *formal consensus* — all three tests must fail to reject normality (p > α, where α defaults to 0.05).

**Output columns include:** Skewness, Kurtosis, all three test statistics and p-values, `Practically_Normal`, `Formal_Consensus`, and final `Approx_Gaussian` flag.

**Visualizations:** Q-Q plot per feature per dataset

**Output Files:** `tables/<mode>/normality_report.csv`, `figures/normality/<mode>/<Dataset>_<Feature>_qq.png`

---

### Stage 7 — Outlier Detection

| | |
|---|---|
| **Module** | `analysis/multivariate.py` → `MultivariateAnalysis.detect_outliers()` |
| **Purpose** | Detect both univariate and multivariate outliers using multiple methods |

**Univariate Methods (per feature):**

| Method | Threshold | Description |
|---|---|---|
| IQR | 1.5× IQR below Q1 or above Q3 | Classical box-plot whisker method |
| Z-Score | \|z\| > 3 | Standard deviation-based |
| MAD | Modified Z-score > 3.5 | Median absolute deviation (robust) |

**Multivariate Methods (all features jointly):**

| Method | Description |
|---|---|
| Isolation Forest | Anomaly detection via random partitioning (contamination=1%) |
| Local Outlier Factor (LOF) | Density-based local anomaly detection (contamination=1%) |
| Mahalanobis Distance | Chi-squared threshold at α=0.001 with pseudoinverse for singular matrices |

**Visualizations:** PCA 2D scatter colored by outlier status (3-panel: IF, LOF, Mahalanobis)

**Output Files:** `tables/<mode>/outlier_summary.csv`, `figures/outliers/<mode>/pca_outliers_*.png`

---

### Stage 8 — Correlation Analysis

| | |
|---|---|
| **Module** | `analysis/multivariate.py` → `MultivariateAnalysis.analyze_correlations()` |
| **Purpose** | Compute pairwise correlations and identify highly correlated feature pairs |
| **Methods** | Pearson, Spearman, Kendall (all three computed for each dataset) |
| **Thresholds** | Feature pairs flagged at \|r\| > 0.90, > 0.95, and > 0.99 |
| **Visualizations** | Full correlation heatmap per method per dataset (6 heatmaps total) |
| **Output Files** | `tables/<mode>/high_correlation_pairs.csv`, `tables/<mode>/<Dataset>_<method>_correlation.csv`, `figures/correlations/<mode>/<Dataset>_<method>_heatmap.png` |

---

### Stage 9 — Feature Redundancy Analysis

| | |
|---|---|
| **Module** | `analysis/multivariate.py` → `MultivariateAnalysis.detect_redundancy()` |
| **Purpose** | Identify features that can be removed due to redundancy |
| **Checks** | 1. **Constant features** (nunique ≤ 1), 2. **Near-constant features** (variance < 1e-4 or one value ≥ 99%), 3. **Variance Inflation Factor (VIF)** — high multicollinearity |
| **VIF Handling** | VIF > 10.0 flagged for review; infinite VIF values (perfect multicollinearity) capped at 9999 and flagged via `VIF_Is_Infinite` column |
| **Recommendations** | Each feature receives: "Keep", "Remove (Constant)", "Review for Removal (Near Constant)", or "Review for Removal (High VIF)" |
| **Output Files** | `tables/<mode>/feature_redundancy.csv` |

---

### Stage 10 — Dimensionality Reduction

| | |
|---|---|
| **Module** | `analysis/multivariate.py` → `MultivariateAnalysis.run_dimensionality()` |
| **Purpose** | Visualize high-dimensional feature structure and assess intrinsic dimensionality |
| **Methods** | 1. **PCA** (up to 10 components) — linear projection, 2. **UMAP** (2D, n_neighbors=15, min_dist=0.1) — non-linear manifold embedding |
| **PCA Outputs** | Explained variance ratio (bar + cumulative step plot), 2D scatter, component loadings matrix |
| **UMAP Outputs** | 2D scatter plot |
| **Output Files** | `tables/<mode>/<Dataset>_pca_loadings.csv`, `figures/dimensionality/<mode>/<Dataset>_pca_variance.png`, `figures/dimensionality/<mode>/<Dataset>_pca_2d.png`, `figures/dimensionality/<mode>/<Dataset>_umap_2d.png` |

---

### Stage 11 — Feature Relationship Analysis

| | |
|---|---|
| **Module** | `analysis/multivariate.py` → `MultivariateAnalysis.analyze_relationships()` |
| **Purpose** | Visualize clinically meaningful feature pairs via scatter + regression |
| **Target Pairs** | RR_Mean_Global ↔ QRS_Dur_Global, S_Amp_V1 ↔ R_Amp_V5 (Sokolow-Lyon), ST_Elev_V2 ↔ ST_Elev_V3 (contiguous leads), QRS_Dur_Global ↔ R_Amp_I, Q_Amp_III ↔ T_Amp_III |
| **Visualizations** | Scatter plot with regression line per pair per dataset (10 plots total) |
| **Output Files** | `figures/relationships/<mode>/<Dataset>_<Y>_vs_<X>.png` |

---

### Stage 12 — Batch Effect Detection

| | |
|---|---|
| **Module** | `analysis/multivariate.py` → `MultivariateAnalysis.detect_batch_effects()` |
| **Purpose** | Assess whether a classifier can distinguish Dataset 1 from Dataset 2 records — indicating systematic batch/domain/extraction differences |

**Method:**
1. Combine both datasets (median-imputed, standard-scaled)
2. Subsample up to 10,000 per dataset for PCA/UMAP visualization
3. Train an **XGBoost classifier** (CUDA-accelerated, 100 estimators, max_depth=10) with 70/30 stratified split
4. Report classification accuracy and top-10 most distinguishing features by importance

**Interpretation:**
- Accuracy ≈ 0.50 → Datasets are indistinguishable (no batch effect / perfect extraction)
- Accuracy ≈ 1.00 → Severe differences (systematic domain differences or extraction discrepancies)

**Mode-aware context in report:**
- **Extraction validation:** High accuracy indicates differences in the extraction process
- **Cross-dataset:** High accuracy is expected due to different patient populations, recording equipment, and clinical contexts

**Output Files:** `tables/<mode>/batch_effect_report.txt`, `tables/<mode>/batch_effect_importance.csv`, `figures/batch_effects/<mode>/combined_pca.png`, `figures/batch_effects/<mode>/combined_umap.png`

---

### Stage 13 — Comparative Statistical Analysis

| | |
|---|---|
| **Module** | `analysis/comparative.py` → `ComparativeAnalysis.analyze_comparisons()` |
| **Purpose** | Formally compare each feature's distribution between Dataset 1 and Dataset 2 |

**Statistical Tests & Effect Sizes:**

| Metric | Description |
|---|---|
| Mean Difference | Arithmetic mean difference (Dataset 1 − Dataset 2) |
| Median Difference | Median difference |
| Cohen's d | Standardized mean difference (pooled SD) |
| Overlap Coefficient | Histogram intersection area (0=no overlap, 1=identical) |
| Kolmogorov-Smirnov Test | Maximum CDF difference + p-value |
| Mann-Whitney U Test | Non-parametric rank comparison + p-value |

**Similarity Classification:**
- Overlap > 0.8 → "Similar"
- Overlap 0.5–0.8 → "Moderately Different"
- Overlap < 0.5 → "Substantially Different"

**Visualizations:** 5-panel detailed comparison per feature (Histogram+KDE, Q-Q, ECDF, Violin, Boxplot)

**Output Files:** `tables/<mode>/comparison_report.csv`, `figures/comparisons/<mode>/<Feature>_comparison.png`

---

### Stage 14 — Similarity Scoring

| | |
|---|---|
| **Module** | `analysis/comparative.py` → `ComparativeAnalysis.analyze_similarity()` |
| **Purpose** | Compute a composite similarity score per feature combining multiple divergence metrics |

**Composite Score Formula:**

```
Similarity = [(1 − JSD) × 0.4 + Overlap × 0.4 + (1 − KS_stat) × 0.2] × 100
```

| Component | Weight | Description |
|---|---|---|
| Jensen-Shannon Divergence (JSD) | 40% | Symmetric information-theoretic divergence |
| Overlap Coefficient | 40% | Histogram intersection |
| KS Statistic (inverted) | 20% | Maximum CDF difference |

**Additional Metrics:** Cohen's d, Quantile-Quantile Pearson correlation (100 quantile points, 1st–99th percentile)

**Output Files:** `tables/<mode>/similarity_report.csv`

---

### Stage 15 — Paired Validation (Extraction Validation Mode Only)

> ⚠️ **This stage only runs when `mode: "extraction_validation"`**. It is automatically skipped in cross-dataset comparison mode.

| | |
|---|---|
| **Module** | `analysis/comparative.py` → `ComparativeAnalysis.analyze_paired_validation()` |
| **Purpose** | Perform record-level agreement analysis on matched ECG records |
| **Prerequisite** | `DataManager._match_records()` must have produced aligned `df1_matched` / `df2_matched` DataFrames |

**Per-Feature Metrics Computed:**

| Category | Metrics |
|---|---|
| **Error Metrics** | MAE, RMSE, MAPE (%), Mean Difference, Median Difference |
| **Paired Statistical Tests** | Paired t-test (statistic + p-value), Wilcoxon signed-rank test (statistic + p-value) |
| **Correlation** | Pearson r (+ p-value), Spearman ρ (+ p-value) |
| **Reliability** | ICC(3,1) — two-way mixed, single measures, consistency |
| **Bland-Altman** | Mean difference (bias), SD of differences, 95% Limits of Agreement (±1.96 SD) |

**Agreement Classification:**

| Category | Criteria |
|---|---|
| Excellent | ICC ≥ 0.95 AND MAPE < 5% |
| Good | ICC ≥ 0.90 AND MAPE < 10% |
| Moderate | ICC ≥ 0.75 AND MAPE < 20% |
| Fair | ICC ≥ 0.50 |
| Poor | ICC < 0.50 |

**Visualizations:**
- **Bland-Altman plot** per feature — scatter of differences vs. means, with mean bias line and 95% LoA bands
- **Paired scatter plot** per feature — Original vs. Re-extracted values with identity line and Pearson r annotation

**Output Files:** `tables/validation/paired_validation_report.csv`, `figures/paired_validation/<Feature>_bland_altman.png`, `figures/paired_validation/<Feature>_paired_scatter.png`

---

### Stage 16 — Quality Scoring

| | |
|---|---|
| **Module** | `reporting/report_generator.py` → `ReportGenerator.calculate_quality_score()` |
| **Purpose** | Produce a single 0–100 quality score per feature by penalizing issues found in earlier stages |

**Scoring System (starts at 100, penalties deducted):**

| Issue | Trigger | Max Penalty |
|---|---|---|
| Missingness | >5% missing | 30 pts (1.5 per % above 5) |
| Outliers (IQR) | >5% outliers | 20 pts (2 per % above 5) |
| Implausible Values | >1% outside reference bounds | 20 pts (2 per %) |
| Low Similarity | Score < 70/100 | 30 pts (0.5 per point below 70) |
| Constant Feature | nunique ≤ 1 | 50 pts |
| Near-Constant Feature | variance < 1e-4 | 30 pts |

**Quality Categories:**

| Score Range | Category |
|---|---|
| ≥ 90 | Excellent |
| 80–89 | Good |
| 70–79 | Fair |
| 60–69 | Poor |
| < 60 | Very Poor |

**Output Files:** `tables/<mode>/feature_quality_report.csv`

---

### Stage 17 — Report Generation

| | |
|---|---|
| **Module** | `reporting/report_generator.py` → `ReportGenerator.generate_html_report()` |
| **Purpose** | Generate an interactive HTML dashboard consolidating key findings |
| **Template Engine** | Jinja2 |
| **Sections (cross-dataset)** | Quality Score Summary, Batch Effects (PCA/UMAP), Missingness, Outliers, Dataset Comparison, Correlations |
| **Sections (extraction validation)** | All of the above, plus: Record Matching Report, Paired Validation agreement table |
| **Styling** | Bootstrap 5, fixed sidebar navigation, responsive cards, scrollable tables |
| **Output Files** | `report/<mode>/ECG_Report_<mode>.html` |

---

### Stage 18 — Publication Export

| | |
|---|---|
| **Module** | `reporting/report_generator.py` → `ReportGenerator.export_publication_tables()` |
| **Purpose** | Export key tables in publication-ready formats |

**Exported Tables (both modes):**

| Publication Name | Source Data | Formats |
|---|---|---|
| Table 1 — Missingness | `missingness_report.csv` | CSV, LaTeX, XLSX |
| Table 2 — Outliers | `outlier_summary.csv` | CSV, LaTeX, XLSX |
| Table 3 — Plausibility | `plausibility_report.csv` | CSV, LaTeX, XLSX |
| Table 4 — Similarity | `similarity_report.csv` | CSV, LaTeX, XLSX |
| Table 5 — Quality Scores | `feature_quality_report.csv` | CSV, LaTeX, XLSX |

**Additional tables (extraction validation mode only):**

| Publication Name | Source Data | Formats |
|---|---|---|
| Table 6 — Paired Validation | `paired_validation_report.csv` | CSV, LaTeX, XLSX |
| Table 7 — Record Matching | `record_matching_report.csv` | CSV, LaTeX, XLSX |

All numeric values are rounded to 3 decimal places. LaTeX tables include top-20 rows with captions and labels.

**Output Files:** `publication/<mode>/Table_*_*.{csv,tex,xlsx}`

---

### Stage 19 — Automated Conclusions

| | |
|---|---|
| **Module** | `reporting/report_generator.py` → `ReportGenerator.generate_conclusions()` |
| **Purpose** | Synthesize a human-readable markdown summary of the most important findings |
| **Sections (both modes)** | Missingness concerns (>10%), Physiological plausibility (>5% outside bounds), Distribution shifts (similarity < 60), Overall quality ratings breakdown |
| **Additional sections (extraction validation)** | Paired Validation Summary — Mean/Median ICC, Mean Pearson r, Mean MAE, Mean MAPE, agreement distribution, list of Fair/Poor agreement features |
| **Output Files** | `report/<mode>/automated_conclusions.md` |

---

## 6. Output Catalog

> All output paths shown below are relative to `outputs/`. Replace `<mode>` with `validation` or `comparison` depending on the active pipeline mode.

### Tables (`outputs/tables/<mode>/`)

| File | Stage | Description |
|---|---|---|
| `overview.csv` | 1 | High-level dataset metadata |
| `feature_summary.csv` | 1 | Per-feature data types, missing counts, constant flags |
| `missingness_report.csv` | 2 | Missing count & percentage per feature per dataset |
| `distribution_statistics.csv` | 3 | 18 descriptive stats per feature per dataset |
| `plausibility_report.csv` | 4 | Plausibility bounds and violation counts |
| `feature_extremes.csv` | 5 | Top/bottom 20 extreme values per feature |
| `normality_report.csv` | 6 | Shapiro-Wilk, Anderson-Darling, D'Agostino, practical normality |
| `outlier_summary.csv` | 7 | Outlier percentages (6 methods × 2 datasets) |
| `high_correlation_pairs.csv` | 8 | Feature pairs with \|r\| > 0.90/0.95/0.99 |
| `<Dataset>_pearson_correlation.csv` | 8 | Full Pearson correlation matrix |
| `<Dataset>_spearman_correlation.csv` | 8 | Full Spearman correlation matrix |
| `<Dataset>_kendall_correlation.csv` | 8 | Full Kendall correlation matrix |
| `feature_redundancy.csv` | 9 | VIF (with infinity cap), constant/near-constant flags, recommendations |
| `<Dataset>_pca_loadings.csv` | 10 | PCA component loadings |
| `batch_effect_report.txt` | 12 | Classification accuracy + mode-aware context + top distinguishing features |
| `batch_effect_importance.csv` | 12 | Feature importance rankings |
| `comparison_report.csv` | 13 | KS, Mann-Whitney, Cohen's d, overlap per feature |
| `similarity_report.csv` | 14 | JSD, overlap, composite similarity score per feature |
| `paired_validation_report.csv` | 15 | *(validation mode only)* ICC, MAE, RMSE, MAPE, paired tests, Bland-Altman, agreement |
| `record_matching_report.csv` | 0 | *(validation mode only)* Record matching statistics |
| `feature_quality_report.csv` | 16 | Final quality score (0–100) per feature |

### Figures (`outputs/figures/`)

| Directory | Stage | Contents |
|---|---|---|
| `missingness/<mode>/` | 2 | Matrix, bar, heatmap plots per dataset |
| `distributions/<mode>/` | 3 | 4-panel distribution plots (one per feature) |
| `normality/<mode>/` | 6 | Q-Q plots per feature per dataset |
| `outliers/<mode>/` | 7 | PCA scatter with outlier labels per dataset |
| `correlations/<mode>/` | 8 | Heatmaps (Pearson, Spearman, Kendall × 2 datasets) |
| `dimensionality/<mode>/` | 10 | PCA variance, PCA 2D, UMAP 2D per dataset |
| `relationships/<mode>/` | 11 | Scatter + regression for clinical feature pairs |
| `batch_effects/<mode>/` | 12 | Combined PCA and UMAP (Dataset 1 vs Dataset 2 colored) |
| `comparisons/<mode>/` | 13 | 5-panel comparison plots (one per feature) |
| `paired_validation/` | 15 | *(validation mode only)* Bland-Altman + paired scatter per feature |

### Reports (`outputs/report/<mode>/`)

| File | Description |
|---|---|
| `ECG_Report_<mode>.html` | Interactive HTML dashboard with sidebar navigation |
| `automated_conclusions.md` | Text summary of key findings (includes paired validation summary in validation mode) |

### Publication Tables (`outputs/publication/<mode>/`)

| File | Description |
|---|---|
| `Table_1_Missingness.{csv,tex,xlsx}` | Publication-ready missingness table |
| `Table_2_Outliers.{csv,tex,xlsx}` | Publication-ready outlier summary |
| `Table_3_Plausibility.{csv,tex,xlsx}` | Publication-ready plausibility report |
| `Table_4_Similarity.{csv,tex,xlsx}` | Publication-ready similarity scores |
| `Table_5_Quality_Scores.{csv,tex,xlsx}` | Publication-ready quality ratings |
| `Table_6_Paired_Validation.{csv,tex,xlsx}` | *(validation only)* Paired agreement metrics |
| `Table_7_Record_Matching.{csv,tex,xlsx}` | *(validation only)* Record matching report |

---

## 7. Key Findings (Run: 2026-07-30 — Extraction Validation)

> **Mode:** `extraction_validation`
> **Comparing:** PTB-XL Original vs PTB-XL Re-extracted
> **Runtime:** ~14 minutes
> **Date:** 2026-07-30 06:45 → 06:59

### Record Matching

| Metric | Value |
|---|---|
| Records in PTB-XL Original | 21,799 |
| Records in PTB-XL Re-extracted | 21,796 |
| Successfully matched | 21,796 |
| Unmatched (only in Original) | 3 |
| Match percentage | 99.99% |

### Batch Effect Detection
- **Classification accuracy: 0.9957** (XGBoost, CUDA) — indicating **high but not perfect distinguishability** between original and re-extracted features.
- This suggests the re-extraction process introduces measurable but minor systematic differences.

### Paired Validation Summary (21,796 matched records)

| Metric | Value |
|---|---|
| Features analyzed | 159 |
| Mean ICC | 0.6681 |
| Median ICC | 0.7918 |
| Mean Pearson r | 0.7054 |
| Mean MAE | 0.3743 |
| Mean MAPE | 44.27% |

### Agreement Distribution

| Category | Count | Criteria |
|---|---|---|
| Excellent | 42 features | ICC ≥ 0.95, MAPE < 5% |
| Good | 6 features | ICC ≥ 0.90, MAPE < 10% |
| Moderate | 7 features | ICC ≥ 0.75, MAPE < 20% |
| Fair | 59 features | ICC ≥ 0.50 |
| Poor | 45 features | ICC < 0.50 |

### Redundancy Warnings
- **PTB-XL Original:** 51 features with infinite VIF (perfect multicollinearity)
- **PTB-XL Re-extracted:** 52 features with infinite VIF

### Interpretation

- **42 features (26.4%)** show excellent paired agreement, indicating the re-extraction process faithfully reproduces these measurements.
- **45 features (28.3%)** show poor agreement (ICC < 0.50), suggesting these features are sensitive to extraction implementation details (e.g., signal processing differences, library version changes).
- The high batch effect accuracy (0.9957) combined with a median ICC of 0.79 suggests that while most features are reasonably consistent, a subset of features exhibits significant extraction variability.

---

## 8. Configuration Reference

All settings are defined in `config.yaml`:

```yaml
# Toggle this mode to change the pipeline's behavior:
# "extraction_validation": PTB-XL Original vs PTB-XL Re-extracted
# "cross_dataset_comparison": MIMIC-IV vs PTB-XL Original
mode: "extraction_validation"

paths:
  mimic_csv: "path/to/extracted_ecg_features.csv"
  ptbxl_original_csv: "path/to/ecgdeli_features.csv"
  ptbxl_reextracted_csv: "path/to/extracted_ecg_features_ptbxl.csv"

plotting:
  dpi: 300
  format: "png"
  colors:
    "MIMIC-IV": "#1f77b4"
    "PTB-XL Original": "#ff7f0e"
    "PTB-XL Re-extracted": "#2ca02c"

thresholds:
  missingness_pct_warn: 5.0          # Warn if >5% missing
  missingness_pct_critical: 10.0     # Critical if >10% missing
  vif_high: 10.0                     # VIF threshold for redundancy
  outlier_iqr: 1.5                   # IQR multiplier
  outlier_z: 3.0                     # Z-score threshold
  outlier_mad: 3.5                   # Modified Z-score threshold
  outlier_contamination: 0.01        # Isolation Forest / LOF contamination
  similarity_score_poor: 60.0
  similarity_score_fair: 70.0
  similarity_score_good: 80.0
  similarity_score_excellent: 90.0
  correlation_high: [0.90, 0.95, 0.99]
  normality_alpha: 0.05              # Significance level for formal normality tests
  normality_skew_threshold: 2.0      # Practical normality: max |skewness|
  normality_kurtosis_threshold: 7.0  # Practical normality: max |kurtosis|
  plausibility_lower_pct: 1.0        # Lower percentile for reference bounds
  plausibility_upper_pct: 99.0       # Upper percentile for reference bounds
  mahalanobis_alpha: 0.001           # Chi-squared threshold for Mahalanobis outliers
  histogram_bins: 100                # Bins for histogram-based calculations
```

**New configuration keys (vs. previous version):**
- `mode` — Controls pipeline behavior (extraction_validation or cross_dataset_comparison)
- `paths.ptbxl_original_csv` — Path to PTB-XL original features
- `paths.ptbxl_reextracted_csv` — Path to PTB-XL re-extracted features
- `plotting.colors["PTB-XL Re-extracted"]` — Color for third dataset
- `thresholds.normality_alpha` — Significance level for normality tests
- `thresholds.normality_skew_threshold` — Practical normality skewness bound
- `thresholds.normality_kurtosis_threshold` — Practical normality kurtosis bound
- `thresholds.plausibility_lower_pct` / `plausibility_upper_pct` — Configurable reference bounds
- `thresholds.mahalanobis_alpha` — Mahalanobis distance chi-squared threshold
- `thresholds.histogram_bins` — Bin count for histogram-based divergence calculations

---

## 9. Setup & Execution

### Prerequisites

```
pip install -r requirements.txt
```

**Core dependencies (13 packages):** pandas, numpy, scipy, matplotlib, seaborn, missingno, scikit-learn, umap-learn, statsmodels, jinja2, pyyaml, xgboost, openpyxl

**GPU-accelerated:** xgboost (with CUDA support — falls back to sklearn RandomForest if unavailable)

### Running the Pipeline

```bash
# Run with default config.yaml (uses mode set in config.yaml)
python main.py

# Run with a custom config file
python main.py --config my_custom_config.yaml
```

The pipeline runs all stages sequentially. Outputs are written to mode-specific subdirectories under `outputs/`.

**Approximate runtimes:**
- **Extraction validation mode:** ~14 minutes (21K paired records, CUDA GPU)
- **Cross-dataset comparison mode:** ~100 minutes (407K + 21K records, CUDA GPU)

### Switching Modes

To switch between extraction validation and cross-dataset comparison:

1. Edit `config.yaml` and change `mode`:
   ```yaml
   mode: "cross_dataset_comparison"   # or "extraction_validation"
   ```
2. Re-run `python main.py`

Outputs from each mode are stored in separate subdirectories, so switching modes does not overwrite previous results.
