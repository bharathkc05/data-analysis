import argparse
from pathlib import Path
from core.config import settings
from core.logger import logger
from data.data_manager import DataManager
from analysis.univariate import UnivariateAnalysis
from analysis.multivariate import MultivariateAnalysis
from analysis.comparative import ComparativeAnalysis
from reporting.report_generator import ReportGenerator
from visualization.plots import Plotter

def main():
    parser = argparse.ArgumentParser(description="ECG-Deli Comprehensive Analysis Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    # The config singleton automatically loads settings when imported,
    # but we could reload it if a different path is provided.
    if args.config != "config.yaml":
        settings.load_config(args.config)
        
    logger.info("=" * 60)
    logger.info(f"STARTING ECG-DELI ANALYSIS PIPELINE")
    logger.info(f"Mode: {settings.mode}")
    logger.info(f"Comparing: {settings.dataset1_name} vs {settings.dataset2_name}")
    logger.info("=" * 60)
    
    Plotter.setup_style()
    
    # 1. Data Loading & Preprocessing
    dm = DataManager()
    dm.load_and_intersect()
    
    # 2. Setup Analyzers
    univariate = UnivariateAnalysis(dm)
    multivariate = MultivariateAnalysis(dm)
    comparative = ComparativeAnalysis(dm)
    reporter = ReportGenerator(dm)
    
    # 3. Basic Overview
    reporter.generate_overview()
    
    # 4. Univariate Analysis
    univariate.analyze_missingness()
    univariate.analyze_distributions(generate_plots=True)
    univariate.analyze_plausibility()
    univariate.extract_extremes(n=settings.thresholds.get('extreme_n', 20))
    univariate.assess_normality()
    
    # 5. Multivariate Analysis
    multivariate.detect_outliers()
    multivariate.analyze_correlations()
    multivariate.detect_redundancy()
    multivariate.run_dimensionality()
    multivariate.analyze_relationships()
    multivariate.detect_batch_effects()
    
    # 6. Comparative Analysis
    comparative.analyze_comparisons(generate_plots=True)
    comparative.analyze_similarity()
    
    # 7. Paired Validation (Conditional based on mode)
    if settings.is_paired_validation:
        comparative.analyze_paired_validation(generate_plots=True)
    else:
        logger.info(f"Skipping Paired Validation (Mode: {settings.mode})")
    
    # 8. Reporting & Export
    reporter.calculate_quality_score()
    reporter.generate_html_report()
    reporter.generate_conclusions()
    reporter.export_publication_tables()
    
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE. Reports generated in: {settings.report_dir}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
