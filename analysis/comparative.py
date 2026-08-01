import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import List, Tuple
from core.config import settings
from core.logger import logger
from data.data_manager import DataManager
from visualization.plots import Plotter
from pathlib import Path

class ComparativeAnalysis:
    """Handles comparison between datasets (Part 10, 11) and paired validation."""
    
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager

    @staticmethod
    def compute_cohens_d(x: pd.Series, y: pd.Series) -> float:
        n1, n2 = len(x), len(y)
        var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        return (np.mean(x) - np.mean(y)) / np.sqrt(pooled_var) if pooled_var > 0 else 0.0

    @staticmethod
    def compute_overlap_coefficient(x: pd.Series, y: pd.Series, bins: int = 100) -> float:
        if len(x) == 0 or len(y) == 0: return 0.0
        min_val = min(np.min(x), np.min(y))
        max_val = max(np.max(x), np.max(y))
        if min_val == max_val: return 1.0
        
        hist_x, _ = np.histogram(x, bins=bins, range=(min_val, max_val), density=True)
        hist_y, _ = np.histogram(y, bins=bins, range=(min_val, max_val), density=True)
        
        bin_width = (max_val - min_val) / bins
        area_x = hist_x * bin_width
        area_y = hist_y * bin_width
        
        return float(np.sum(np.minimum(area_x, area_y)))

    @staticmethod
    def jensen_shannon_divergence(p: pd.Series, q: pd.Series, bins: int = 100) -> float:
        if len(p) == 0 or len(q) == 0: return 1.0
        min_val = min(np.min(p), np.min(q))
        max_val = max(np.max(p), np.max(q))
        if min_val == max_val: return 0.0
        
        p_hist, _ = np.histogram(p, bins=bins, range=(min_val, max_val), density=True)
        q_hist, _ = np.histogram(q, bins=bins, range=(min_val, max_val), density=True)
        
        epsilon = 1e-10
        p_hist = p_hist + epsilon
        q_hist = q_hist + epsilon
        
        p_hist = p_hist / np.sum(p_hist)
        q_hist = q_hist / np.sum(q_hist)
        
        m = 0.5 * (p_hist + q_hist)
        jsd = 0.5 * np.sum(p_hist * np.log(p_hist / m)) + 0.5 * np.sum(q_hist * np.log(q_hist / m))
        return float(jsd)

    @staticmethod
    def compute_icc(x: np.ndarray, y: np.ndarray) -> float:
        """Computes ICC(3,1) — two-way mixed, single measures, consistency."""
        n = len(x)
        if n < 3:
            return np.nan
        
        # Stack into matrix: n subjects x 2 raters
        data = np.column_stack([x, y])
        k = 2  # number of raters
        
        grand_mean = np.mean(data)
        row_means = np.mean(data, axis=1)
        col_means = np.mean(data, axis=0)
        
        # Sum of squares
        ss_total = np.sum((data - grand_mean) ** 2)
        ss_rows = k * np.sum((row_means - grand_mean) ** 2)  # between subjects
        ss_cols = n * np.sum((col_means - grand_mean) ** 2)  # between raters
        ss_residual = ss_total - ss_rows - ss_cols
        
        # Mean squares
        ms_rows = ss_rows / (n - 1)
        ms_residual = ss_residual / ((n - 1) * (k - 1))
        
        # ICC(3,1) — two-way mixed, consistency
        denom = ms_rows + (k - 1) * ms_residual
        if denom == 0:
            return np.nan
        icc = (ms_rows - ms_residual) / denom
        return float(icc)

    def analyze_comparisons(self, generate_plots: bool = True) -> pd.DataFrame:
        """Runs comparative statistics (Part 10)."""
        logger.info("Running Comparative: Dataset Comparisons...")
        results = []
        
        for feature in self.dm.numeric_features:
            data1 = self.dm.df1[feature].dropna()
            data2 = self.dm.df2[feature].dropna()
            
            if len(data1) < 10 or len(data2) < 10: continue
            
            mean_diff = np.mean(data1) - np.mean(data2)
            median_diff = np.median(data1) - np.median(data2)
            
            ks_stat, ks_pval = stats.ks_2samp(data1, data2)
            mw_stat, mw_pval = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            cohens_d = self.compute_cohens_d(data1, data2)
            overlap = self.compute_overlap_coefficient(data1, data2)
            
            similarity = 'Similar' if overlap > 0.8 else ('Substantially Different' if overlap < 0.5 else 'Moderately Different')
            
            results.append({
                'Feature': feature, 'Mean_Diff': mean_diff, 'Median_Diff': median_diff,
                'Cohen_d': cohens_d, 'Overlap_Coefficient': overlap, 'KS_Stat': ks_stat,
                'KS_pvalue': ks_pval, 'MW_Stat': mw_stat, 'MW_pvalue': mw_pval, 'Similarity': similarity
            })
            
            if generate_plots:
                Plotter.plot_detailed_comparison(data1, data2, feature)
                
        comp_df = pd.DataFrame(results)
        comp_df.to_csv(settings.tables_dir / "comparison_report.csv", index=False)
        return comp_df

    def analyze_similarity(self) -> pd.DataFrame:
        """Computes JSD and composite similarity score (Part 11)."""
        logger.info("Running Comparative: Similarity Score...")
        results = []
        
        for feature in self.dm.numeric_features:
            data1 = self.dm.df1[feature].dropna()
            data2 = self.dm.df2[feature].dropna()
            
            if len(data1) < 10 or len(data2) < 10: continue
            
            ks_stat, ks_pval = stats.ks_2samp(data1, data2)
            cohens_d = self.compute_cohens_d(data1, data2)
            js_div = self.jensen_shannon_divergence(data1, data2)
            overlap = self.compute_overlap_coefficient(data1, data2)
            
            quantiles = np.linspace(0.01, 0.99, 100)
            q1 = np.quantile(data1, quantiles)
            q2 = np.quantile(data2, quantiles)
            
            # Add tiny noise to avoid constant array issues in pearsonr
            if np.std(q1) == 0: q1 += np.random.normal(0, 1e-10, len(q1))
            if np.std(q2) == 0: q2 += np.random.normal(0, 1e-10, len(q2))
            
            corr, _ = stats.pearsonr(q1, q2)
            
            sim_score = ((1 - min(js_div, 1.0)) * 0.4 + (overlap) * 0.4 + (1 - ks_stat) * 0.2) * 100
            
            results.append({
                'Feature': feature, 'KS_Statistic': ks_stat, 'KS_pvalue': ks_pval,
                'Cohens_d': cohens_d, 'JS_Divergence': js_div, 'Quantile_Correlation': corr,
                'Similarity_Score': sim_score
            })
            
        sim_df = pd.DataFrame(results).sort_values('Similarity_Score', ascending=False)
        sim_df.to_csv(settings.tables_dir / "similarity_report.csv", index=False)
        return sim_df

    def analyze_paired_validation(self, generate_plots: bool = True) -> pd.DataFrame:
        """Runs paired validation analyses on matched records."""
        if not settings.is_paired_validation:
            return pd.DataFrame()
            
        logger.info("=" * 60)
        logger.info(f"Running Paired Validation: {settings.dataset1_name} vs {settings.dataset2_name}")
        logger.info("=" * 60)
        
        if self.dm.df1_matched is None or self.dm.df2_matched is None:
            logger.warning("No matched records available. Skipping paired validation.")
            return pd.DataFrame()
        
        n_matched = len(self.dm.df1_matched)
        logger.info(f"Using {n_matched} matched ECG records for paired validation.")
        
        results = []
        
        for feature in self.dm.numeric_features:
            if feature not in self.dm.df1_matched.columns or feature not in self.dm.df2_matched.columns:
                continue
            
            vals1 = self.dm.df1_matched[feature]
            vals2 = self.dm.df2_matched[feature]
            
            # Drop pairs where either is NaN
            valid_mask = vals1.notna() & vals2.notna()
            valid1 = vals1[valid_mask].values
            valid2 = vals2[valid_mask].values
            n_valid = len(valid1)
            
            if n_valid < 10:
                continue
            
            # --- Per-record error metrics ---
            diff = valid2 - valid1
            abs_diff = np.abs(diff)
            
            mae = np.mean(abs_diff)
            rmse = np.sqrt(np.mean(diff ** 2))
            
            # MAPE — only where valid1 != 0
            nonzero_mask = valid1 != 0
            if np.sum(nonzero_mask) > 0:
                mape = np.mean(np.abs(diff[nonzero_mask] / valid1[nonzero_mask])) * 100
            else:
                mape = np.nan
            
            # Mean and median difference
            mean_diff = np.mean(diff)
            median_diff = np.median(diff)
            
            # --- Paired statistical tests ---
            # Paired t-test
            try:
                t_stat, t_pval = stats.ttest_rel(valid1, valid2)
            except Exception:
                t_stat, t_pval = np.nan, np.nan
            
            # Wilcoxon signed-rank test
            try:
                w_stat, w_pval = stats.wilcoxon(valid1, valid2)
            except Exception:
                w_stat, w_pval = np.nan, np.nan
            
            # --- Paired correlation ---
            try:
                pearson_r, pearson_p = stats.pearsonr(valid1, valid2)
            except Exception:
                pearson_r, pearson_p = np.nan, np.nan
            
            try:
                spearman_r, spearman_p = stats.spearmanr(valid1, valid2)
            except Exception:
                spearman_r, spearman_p = np.nan, np.nan
            
            # --- ICC ---
            icc = self.compute_icc(valid1, valid2)
            
            # --- Bland-Altman metrics ---
            ba_mean = (valid1 + valid2) / 2
            ba_diff = valid2 - valid1
            ba_mean_diff = np.mean(ba_diff)
            ba_sd_diff = np.std(ba_diff, ddof=1)
            ba_loa_upper = ba_mean_diff + 1.96 * ba_sd_diff
            ba_loa_lower = ba_mean_diff - 1.96 * ba_sd_diff
            
            # --- Agreement classification ---
            if icc >= 0.95 and mape < 5:
                agreement = "Excellent"
            elif icc >= 0.90 and mape < 10:
                agreement = "Good"
            elif icc >= 0.75 and mape < 20:
                agreement = "Moderate"
            elif icc >= 0.50:
                agreement = "Fair"
            else:
                agreement = "Poor"
            
            results.append({
                'Feature': feature,
                'N_Paired': n_valid,
                'Mean_Diff': round(mean_diff, 6),
                'Median_Diff': round(median_diff, 6),
                'MAE': round(mae, 6),
                'RMSE': round(rmse, 6),
                'MAPE_pct': round(mape, 4) if not np.isnan(mape) else np.nan,
                'Paired_t_stat': round(t_stat, 4) if not np.isnan(t_stat) else np.nan,
                'Paired_t_pvalue': t_pval,
                'Wilcoxon_stat': round(w_stat, 4) if not np.isnan(w_stat) else np.nan,
                'Wilcoxon_pvalue': w_pval,
                'Pearson_r': round(pearson_r, 6) if not np.isnan(pearson_r) else np.nan,
                'Pearson_pvalue': pearson_p,
                'Spearman_r': round(spearman_r, 6) if not np.isnan(spearman_r) else np.nan,
                'Spearman_pvalue': spearman_p,
                'ICC': round(icc, 6) if not np.isnan(icc) else np.nan,
                'BlandAltman_MeanDiff': round(ba_mean_diff, 6),
                'BlandAltman_SD': round(ba_sd_diff, 6),
                'BlandAltman_LoA_Upper': round(ba_loa_upper, 6),
                'BlandAltman_LoA_Lower': round(ba_loa_lower, 6),
                'Agreement': agreement
            })
            
            # --- Paired validation plots ---
            if generate_plots:
                Plotter.plot_bland_altman(ba_mean, ba_diff, ba_mean_diff, ba_loa_upper, ba_loa_lower, feature)
                Plotter.plot_paired_scatter(valid1, valid2, feature, pearson_r)
        
        paired_df = pd.DataFrame(results).sort_values('ICC', ascending=False)
        paired_df.to_csv(settings.tables_dir / "paired_validation_report.csv", index=False)
        
        # Log summary statistics
        if not paired_df.empty:
            logger.info("=" * 60)
            logger.info("PAIRED VALIDATION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"  Features analyzed:    {len(paired_df)}")
            logger.info(f"  Mean ICC:             {paired_df['ICC'].mean():.4f}")
            logger.info(f"  Median ICC:           {paired_df['ICC'].median():.4f}")
            logger.info(f"  Mean Pearson r:       {paired_df['Pearson_r'].mean():.4f}")
            logger.info(f"  Mean MAE:             {paired_df['MAE'].mean():.4f}")
            logger.info(f"  Mean MAPE:            {paired_df['MAPE_pct'].mean():.2f}%")
            for cat in ['Excellent', 'Good', 'Moderate', 'Fair', 'Poor']:
                count = (paired_df['Agreement'] == cat).sum()
                if count > 0:
                    logger.info(f"  {cat} agreement:  {count} features")
            logger.info("=" * 60)
        
        return paired_df
