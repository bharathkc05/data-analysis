import pandas as pd
import os
from datetime import datetime
from core.config import settings
from core.logger import logger
from data.data_manager import DataManager

try:
    import jinja2
except ImportError:
    jinja2 = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ECG-Deli {{ report_type }} Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; }
        .sidebar { height: 100vh; position: fixed; left: 0; padding: 20px; background-color: #343a40; color: white; overflow-y: auto; width: 250px;}
        .sidebar a { color: #adb5bd; text-decoration: none; display: block; margin: 10px 0; }
        .sidebar a:hover { color: white; }
        .main-content { margin-left: 260px; padding: 30px; }
        .card { margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .img-fluid { border: 1px solid #dee2e6; border-radius: 5px; }
        .table-responsive { max-height: 400px; overflow-y: auto; }
        h1, h2 { color: #212529; border-bottom: 2px solid #dee2e6; padding-bottom: 10px; margin-top: 30px;}
    </style>
</head>
<body>

<div class="sidebar">
    <h4>ECG-Deli Pipeline</h4>
    <hr>
    {% if is_paired %}
    <a href="#matching">1. Record Matching</a>
    {% endif %}
    <a href="#summary">Quality Summary</a>
    {% if is_paired %}
    <a href="#paired">Paired Validation</a>
    {% endif %}
    <a href="#batch">Batch Effects (PCA/UMAP)</a>
    <a href="#missing">Missingness</a>
    <a href="#outliers">Outliers</a>
    <a href="#compare">Distribution Comparison</a>
    <a href="#corr">Correlations</a>
</div>

<div class="main-content">
    <div class="card p-4 bg-primary text-white">
        <h1 class="text-white border-0">ECG-Deli {{ report_type }} Report</h1>
        <p>Generated on: {{ date }}</p>
        <p>Comparing: <strong>{{ dataset1_name }}</strong> vs <strong>{{ dataset2_name }}</strong>.</p>
    </div>

    {% if is_paired %}
    <section id="matching">
        <h2>Record Matching Report</h2>
        <div class="card p-3">
            <p>Inner join on <code>ecg_id</code> between {{ dataset1_name }} and {{ dataset2_name }}.</p>
            <div class="table-responsive">
                {{ matching_table | safe }}
            </div>
        </div>
    </section>
    {% endif %}

    <section id="summary">
        <h2>Feature Quality Score Summary</h2>
        <div class="card p-3">
            <p>Overall quality score combining missingness, outliers, physiological plausibility, and similarity to {{ dataset1_name }}.</p>
            <div class="table-responsive">
                {{ quality_table | safe }}
            </div>
        </div>
    </section>

    {% if is_paired %}
    <section id="paired">
        <h2>Paired Validation ({{ dataset1_name }} vs {{ dataset2_name }})</h2>
        <div class="card p-3">
            <p>Per-feature paired agreement metrics: ICC, MAE, MAPE, Pearson r, Bland-Altman limits of agreement.</p>
            <div class="table-responsive">
                {{ paired_table | safe }}
            </div>
        </div>
    </section>
    {% endif %}

    <section id="batch">
        <h2>Batch Effect / Dataset Difference Detection</h2>
        <div class="row">
            <div class="col-md-6">
                <div class="card p-3">
                    <h5>Combined PCA</h5>
                    <img src="../../figures/batch_effects/{{ output_subdir }}/combined_pca.png" class="img-fluid" alt="PCA">
                </div>
            </div>
            <div class="col-md-6">
                <div class="card p-3">
                    <h5>Combined UMAP</h5>
                    <img src="../../figures/batch_effects/{{ output_subdir }}/combined_umap.png" class="img-fluid" alt="UMAP">
                </div>
            </div>
        </div>
    </section>

    <section id="missing">
        <h2>Missingness</h2>
        <div class="row">
            <div class="col-md-6">
                <div class="card p-3">
                    <h5>{{ dataset1_name }} Missingness</h5>
                    <img src="../../figures/missingness/{{ output_subdir }}/missing_bar_{{ dataset1_name }}.png" class="img-fluid" alt="{{ dataset1_name }} Missingness">
                </div>
            </div>
            <div class="col-md-6">
                <div class="card p-3">
                    <h5>{{ dataset2_name }} Missingness</h5>
                    <img src="../../figures/missingness/{{ output_subdir }}/missing_bar_{{ dataset2_name }}.png" class="img-fluid" alt="{{ dataset2_name }} Missingness">
                </div>
            </div>
        </div>
    </section>

    <section id="outliers">
        <h2>Outliers</h2>
        <div class="row">
            <div class="col-md-6">
                <div class="card p-3">
                    <h5>{{ dataset1_name }} Outliers</h5>
                    <img src="../../figures/outliers/{{ output_subdir }}/pca_outliers_{{ dataset1_name }}.png" class="img-fluid" alt="{{ dataset1_name }} Outliers">
                </div>
            </div>
            <div class="col-md-6">
                <div class="card p-3">
                    <h5>{{ dataset2_name }} Outliers</h5>
                    <img src="../../figures/outliers/{{ output_subdir }}/pca_outliers_{{ dataset2_name }}.png" class="img-fluid" alt="{{ dataset2_name }} Outliers">
                </div>
            </div>
        </div>
    </section>
    
    <section id="compare">
        <h2>Distribution Comparison ({{ dataset1_name }} vs {{ dataset2_name }})</h2>
        <div class="card p-3">
            <p>Formal similarity rankings using Jensen-Shannon Divergence and Overlap Coefficient.</p>
            <div class="table-responsive">
                {{ sim_table | safe }}
            </div>
        </div>
    </section>

    <section id="corr">
        <h2>Correlations</h2>
        <div class="row">
            <div class="col-md-6">
                <div class="card p-3">
                    <h5>{{ dataset1_name }} Spearman Correlation</h5>
                    <img src="../../figures/correlations/{{ output_subdir }}/{{ dataset1_name }}_spearman_heatmap.png" class="img-fluid" alt="{{ dataset1_name }} Correlation">
                </div>
            </div>
            <div class="col-md-6">
                <div class="card p-3">
                    <h5>{{ dataset2_name }} Spearman Correlation</h5>
                    <img src="../../figures/correlations/{{ output_subdir }}/{{ dataset2_name }}_spearman_heatmap.png" class="img-fluid" alt="{{ dataset2_name }} Correlation">
                </div>
            </div>
        </div>
    </section>

</div>

</body>
</html>
"""

class ReportGenerator:
    """Handles quality scoring, conclusions, HTML generation, and publication tables."""
    
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager

    def generate_overview(self):
        """Generates high-level dataset overview (Part 1)."""
        logger.info("Running Reporting: Dataset Overview...")
        overviews = []
        feature_summaries = []
        
        for dataset_name, df in [(settings.dataset1_name, self.dm.df1), (settings.dataset2_name, self.dm.df2)]:
            num_ecgs = len(df)
            num_features = len(df.columns)
            mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
            
            id_col = self.dm.get_id_column(dataset_name)
            dup_ids = df[id_col].duplicated().sum() if id_col in df.columns else 0
            dup_rows = df.duplicated().sum()
            
            missing_counts = df.isnull().sum()
            
            constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
            near_const_cols = [c for c in df.columns if c not in constant_cols and df[c].value_counts(normalize=True).max() >= 0.99]
            all_missing = [c for c in df.columns if missing_counts[c] == num_ecgs]
            
            overviews.append({
                "Dataset": dataset_name, "Total_ECGs": num_ecgs, "Total_Features": num_features,
                "Memory_Usage_MB": round(mem_mb, 2), "Duplicate_IDs": dup_ids, "Duplicate_Rows": dup_rows,
                "Constant_Columns": len(constant_cols), "Near_Constant_Columns": len(near_const_cols), "All_Missing_Columns": len(all_missing)
            })
            
            fs = pd.DataFrame({
                "Dataset": dataset_name, "Feature": df.columns, "DataType": df.dtypes.values,
                "Missing_Count": missing_counts.values, "Missing_Percent": (missing_counts.values / num_ecgs) * 100,
                "Is_Constant": df.columns.isin(constant_cols), "Is_Near_Constant": df.columns.isin(near_const_cols), "Is_All_Missing": df.columns.isin(all_missing)
            })
            feature_summaries.append(fs)
            
        pd.DataFrame(overviews).to_csv(settings.tables_dir / "overview.csv", index=False)
        pd.concat(feature_summaries, ignore_index=True).to_csv(settings.tables_dir / "feature_summary.csv", index=False)

    def calculate_quality_score(self):
        """Calculates final data quality score (Part 15)."""
        logger.info("Running Reporting: Quality Scoring...")
        try:
            m_report = pd.read_csv(settings.tables_dir / "missingness_report.csv")
            m_target = m_report[m_report['Dataset'] == settings.dataset2_name]
            outlier_df = pd.read_csv(settings.tables_dir / "outlier_summary.csv")
            plaus_df = pd.read_csv(settings.tables_dir / "plausibility_report.csv")
            sim_df = pd.read_csv(settings.tables_dir / "similarity_report.csv")
            red_df = pd.read_csv(settings.tables_dir / "feature_redundancy.csv")
        except FileNotFoundError as e:
            logger.warning(f"Could not load reports for scoring: {e}")
            return
            
        results = []
        for feature in self.dm.numeric_features:
            score = 100
            penalties = []
            
            # Missingness
            m_row = m_target[m_target['Feature'] == feature]
            if not m_row.empty and m_row['Missing_Percent'].values[0] > settings.thresholds.get('missingness_pct_warn', 5.0):
                pct = m_row['Missing_Percent'].values[0]
                penalty = min(30, int((pct - 5) * 1.5))
                score -= penalty
                penalties.append(f"Missingness ({pct:.1f}%)")
                
            # Outliers (IQR)
            o_row = outlier_df[(outlier_df['Dataset'] == settings.dataset2_name) & (outlier_df['Method'] == 'IQR') & (outlier_df['Feature_or_Method'] == feature)]
            if not o_row.empty and o_row['Pct_Outliers'].values[0] > 5:
                pct = o_row['Pct_Outliers'].values[0]
                penalty = min(20, int((pct - 5) * 2))
                score -= penalty
                penalties.append(f"Outliers ({pct:.1f}%)")
                
            # Plausibility
            p_row = plaus_df[plaus_df['Feature'] == feature]
            if not p_row.empty and p_row['Pct_Outside_Range'].values[0] > 1:
                pct = p_row['Pct_Outside_Range'].values[0]
                penalty = min(20, int(pct * 2))
                score -= penalty
                penalties.append(f"Implausible Values ({pct:.1f}%)")
                
            # Similarity
            s_row = sim_df[sim_df['Feature'] == feature]
            if not s_row.empty and s_row['Similarity_Score'].values[0] < settings.thresholds.get('similarity_score_fair', 70):
                sim = s_row['Similarity_Score'].values[0]
                penalty = min(30, int((70 - sim) * 0.5))
                score -= penalty
                penalties.append(f"Low Similarity ({sim:.1f}/100)")
                
            # Redundancy
            r_row = red_df[(red_df['Dataset'] == settings.dataset2_name) & (red_df['Feature'] == feature)]
            if not r_row.empty:
                if r_row['Is_Constant'].values[0]:
                    score -= 50; penalties.append("Constant")
                elif r_row['Is_Near_Constant'].values[0]:
                    score -= 30; penalties.append("Near-Constant")
                    
            if score >= settings.thresholds.get('similarity_score_excellent', 90): cat = "Excellent"
            elif score >= settings.thresholds.get('similarity_score_good', 80): cat = "Good"
            elif score >= settings.thresholds.get('similarity_score_fair', 70): cat = "Fair"
            elif score >= settings.thresholds.get('similarity_score_poor', 60): cat = "Poor"
            else: cat = "Very Poor"
            
            results.append({'Feature': feature, 'Quality_Score': max(0, score), 'Category': cat, 'Primary_Issues': " | ".join(penalties) if penalties else "None"})
            
        qual_df = pd.DataFrame(results).sort_values('Quality_Score', ascending=False)
        qual_df.to_csv(settings.tables_dir / "feature_quality_report.csv", index=False)
        
    def generate_html_report(self):
        """Generates an HTML report summarizing findings (Part 16)."""
        logger.info("Running Reporting: HTML Report Generation...")
        if not jinja2:
            logger.warning("jinja2 not installed. Skipping HTML generation.")
            return
            
        try:
            qual_df = pd.read_csv(settings.tables_dir / "feature_quality_report.csv")
            sim_df = pd.read_csv(settings.tables_dir / "similarity_report.csv")
            qual_html = qual_df.to_html(classes="table table-striped table-hover", index=False)
            sim_html = sim_df.to_html(classes="table table-striped table-hover", index=False)
        except FileNotFoundError:
            qual_html, sim_html = "<p>Data not found.</p>", "<p>Data not found.</p>"
        
        match_html = ""
        paired_html = ""
        if settings.is_paired_validation:
            try:
                match_df = pd.read_csv(settings.tables_dir / "record_matching_report.csv")
                match_html = match_df.to_html(classes="table table-striped table-hover", index=False)
            except FileNotFoundError:
                match_html = "<p>Matching data not found.</p>"
            
            try:
                paired_df = pd.read_csv(settings.tables_dir / "paired_validation_report.csv")
                paired_html = paired_df.to_html(classes="table table-striped table-hover", index=False)
            except FileNotFoundError:
                paired_html = "<p>Paired validation data not found.</p>"
                
        output_subdir = "validation" if settings.is_paired_validation else "comparison"
        report_type = "Extraction Validation" if settings.is_paired_validation else "Cross-Dataset Comparison"
        
        template = jinja2.Template(HTML_TEMPLATE)
        html_out = template.render(
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            report_type=report_type,
            dataset1_name=settings.dataset1_name,
            dataset2_name=settings.dataset2_name,
            is_paired=settings.is_paired_validation,
            output_subdir=output_subdir,
            quality_table=qual_html,
            sim_table=sim_html,
            matching_table=match_html,
            paired_table=paired_html
        )
        
        report_filename = f"ECG_Report_{output_subdir}.html"
        with open(settings.report_dir / report_filename, "w", encoding="utf-8") as f:
            f.write(html_out)
            
    def export_publication_tables(self):
        """Exports tables to publication-ready formats (Part 17)."""
        logger.info("Running Reporting: Publication Export...")
        tables = [
            ("Table_1_Missingness", "missingness_report.csv"),
            ("Table_2_Outliers", "outlier_summary.csv"),
            ("Table_3_Plausibility", "plausibility_report.csv"),
            ("Table_4_Similarity", "similarity_report.csv"),
            ("Table_5_Quality_Scores", "feature_quality_report.csv"),
        ]
        
        if settings.is_paired_validation:
            tables.extend([
                ("Table_6_Paired_Validation", "paired_validation_report.csv"),
                ("Table_7_Record_Matching", "record_matching_report.csv"),
            ])
        
        for pub_name, filename in tables:
            source = settings.tables_dir / filename
            if source.exists():
                df = pd.read_csv(source)
                num_cols = df.select_dtypes(include=['float64']).columns
                df[num_cols] = df[num_cols].round(3)
                
                df.to_csv(settings.pub_dir / f"{pub_name}.csv", index=False)
                try:
                    df.to_excel(settings.pub_dir / f"{pub_name}.xlsx", index=False)
                except Exception:
                    pass
                
                with open(settings.pub_dir / f"{pub_name}.tex", "w") as f:
                    f.write(df.head(20).to_latex(index=False, float_format="%.3f", caption=f"{pub_name.replace('_', ' ')} (Top 20)", label=f"tab:{pub_name.lower()}"))
                    
    def generate_conclusions(self):
        """Synthesizes text-based conclusions (Part 18)."""
        logger.info("Running Reporting: Synthesizing Conclusions...")
        try:
            m_report = pd.read_csv(settings.tables_dir / "missingness_report.csv")
            m_target = m_report[m_report['Dataset'] == settings.dataset2_name]
            plaus_df = pd.read_csv(settings.tables_dir / "plausibility_report.csv")
            sim_df = pd.read_csv(settings.tables_dir / "similarity_report.csv")
            qual_df = pd.read_csv(settings.tables_dir / "feature_quality_report.csv")
            
            report_title = "Extraction Validation" if settings.is_paired_validation else "Cross-Dataset Comparison"
            lines = [f"# ECG-Deli {report_title} Conclusions\n",
                     "## Overview",
                     f"Comparing {settings.dataset2_name} against {settings.dataset1_name}.\n",
                     f"## 1. Missingness Concerns ({settings.dataset2_name})"]
            
            high_missing = m_target[m_target['Missing_Percent'] > 10].sort_values('Missing_Percent', ascending=False)
            if not high_missing.empty:
                lines.append(f"Found {len(high_missing)} features with >10% missing data in {settings.dataset2_name}.")
                for _, r in high_missing.head(5).iterrows():
                    lines.append(f"- **{r['Feature']}**: {r['Missing_Percent']:.1f}% missing")
            else:
                lines.append(f"No features have concerning levels (>10%) of missing data in {settings.dataset2_name}.")
                
            lines.extend([f"\n## 2. Physiological Plausibility ({settings.dataset2_name} Outside {settings.dataset1_name} Bounds)"])
            high_plaus = plaus_df[plaus_df['Pct_Outside_Range'] > 5].sort_values('Pct_Outside_Range', ascending=False)
            if not high_plaus.empty:
                lines.append(f"Found {len(high_plaus)} features where >5% of {settings.dataset2_name} values fall outside {settings.dataset1_name} bounds.")
                for _, r in high_plaus.head(5).iterrows():
                    lines.append(f"- **{r['Feature']}**: {r['Pct_Outside_Range']:.1f}% outside reference bounds")
            else:
                lines.append(f"All {settings.dataset2_name} features generally conform to {settings.dataset1_name} bounds.")
                
            lines.extend([f"\n## 3. Distribution Similarity ({settings.dataset1_name} vs {settings.dataset2_name})"])
            poor_sim = sim_df[sim_df['Similarity_Score'] < 60]
            if not poor_sim.empty:
                lines.append(f"Found {len(poor_sim)} features with severe distribution shifts (Score < 60/100).")
            else:
                lines.append("All features maintain a reasonable distribution similarity.")
            
            if settings.is_paired_validation:
                try:
                    paired_df = pd.read_csv(settings.tables_dir / "paired_validation_report.csv")
                    if not paired_df.empty:
                        lines.extend(["\n## 4. Paired Validation Summary (Matched Records)"])
                        lines.append(f"- **Features analyzed**: {len(paired_df)}")
                        lines.append(f"- **Mean ICC**: {paired_df['ICC'].mean():.4f}")
                        lines.append(f"- **Median ICC**: {paired_df['ICC'].median():.4f}")
                        lines.append(f"- **Mean Pearson r**: {paired_df['Pearson_r'].mean():.4f}")
                        lines.append(f"- **Mean MAE**: {paired_df['MAE'].mean():.4f}")
                        lines.append(f"- **Mean MAPE**: {paired_df['MAPE_pct'].mean():.2f}%")
                        
                        lines.append("\n### Agreement Distribution:")
                        for cat in ['Excellent', 'Good', 'Moderate', 'Fair', 'Poor']:
                            count = (paired_df['Agreement'] == cat).sum()
                            if count > 0:
                                lines.append(f"- **{cat}**: {count} features")
                        
                        poor_features = paired_df[paired_df['Agreement'].isin(['Fair', 'Poor'])]
                        if not poor_features.empty:
                            lines.append("\n### Features with Fair/Poor Agreement:")
                            for _, r in poor_features.head(10).iterrows():
                                lines.append(f"- **{r['Feature']}**: ICC={r['ICC']:.4f}, MAPE={r['MAPE_pct']:.1f}%")
                except FileNotFoundError:
                    pass
            
            lines.extend(["\n## Final Feature Quality Ratings"])
            for cat, count in qual_df['Category'].value_counts().items():
                lines.append(f"- **{cat}**: {count} features")
                
            with open(settings.report_dir / "automated_conclusions.md", "w") as f:
                f.write("\n".join(lines))
        except Exception as e:
            logger.warning(f"Error generating conclusions: {e}")
