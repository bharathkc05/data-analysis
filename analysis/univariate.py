import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import List, Tuple
from core.config import settings
from core.logger import logger
from data.data_manager import DataManager
from visualization.plots import Plotter
from pathlib import Path

class UnivariateAnalysis:
    """Handles 1D statistical analysis: Missingness, Distribution, Plausibility, Extremes, Normality."""
    
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
        
    def analyze_missingness(self) -> pd.DataFrame:
        """Analyzes missing data (Part 2)."""
        logger.info("Running Univariate: Missingness Analysis...")
        
        results = []
        for dataset_name, df in [(settings.dataset1_name, self.dm.df1), (settings.dataset2_name, self.dm.df2)]:
            missing_counts = df.isnull().sum()
            missing_pct = (missing_counts / len(df)) * 100
            
            missing_df = pd.DataFrame({
                'Dataset': dataset_name,
                'Feature': missing_counts.index,
                'Missing_Count': missing_counts.values,
                'Missing_Percent': missing_pct.values
            })
            results.append(missing_df)
            
            # Plot
            cols_with_missing = missing_counts[missing_counts > 0].index.tolist()
            Plotter.plot_missingness(df, dataset_name, cols_with_missing)
            
        combined_df = pd.concat(results, ignore_index=True)
        combined_df.to_csv(settings.tables_dir / "missingness_report.csv", index=False)
        return combined_df
        
    def _compute_distribution_stats(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        stats_list = []
        for col in self.dm.numeric_features:
            data = df[col].dropna()
            if len(data) == 0: continue
            
            stats_list.append({
                'Dataset': dataset_name,
                'Feature': col,
                'Mean': np.mean(data),
                'Median': np.median(data),
                'SD': np.std(data),
                'Min': np.min(data),
                'Max': np.max(data),
                'IQR': stats.iqr(data),
                'MAD': stats.median_abs_deviation(data) if len(data) > 0 else np.nan,
                'Variance': np.var(data),
                'Skewness': stats.skew(data),
                'Kurtosis': stats.kurtosis(data),
                'p01': np.percentile(data, 1),
                'p05': np.percentile(data, 5),
                'p25': np.percentile(data, 25),
                'p50': np.percentile(data, 50),
                'p75': np.percentile(data, 75),
                'p95': np.percentile(data, 95),
                'p99': np.percentile(data, 99)
            })
        return pd.DataFrame(stats_list)

    def analyze_distributions(self, generate_plots: bool = True) -> pd.DataFrame:
        """Computes basic distribution statistics and plots (Part 3)."""
        logger.info("Running Univariate: Distribution Analysis...")
        
        stats1 = self._compute_distribution_stats(self.dm.df1, settings.dataset1_name)
        stats2 = self._compute_distribution_stats(self.dm.df2, settings.dataset2_name)
        
        combined_stats = pd.concat([stats1, stats2], ignore_index=True)
        combined_stats.to_csv(settings.tables_dir / "distribution_statistics.csv", index=False)
        
        if generate_plots:
            logger.info("Generating comparative distribution plots...")
            for feature in self.dm.numeric_features:
                data1 = self.dm.df1[feature].dropna()
                data2 = self.dm.df2[feature].dropna()
                Plotter.plot_comparative_distribution(data1, data2, feature)
                
        return combined_stats

    def analyze_plausibility(self) -> pd.DataFrame:
        """Checks target values against reference bounds (Part 4)."""
        logger.info("Running Univariate: Plausibility Analysis...")
        if settings.is_paired_validation:
            logger.info(f"NOTE: Reference bounds are from {settings.dataset1_name}. {settings.dataset2_name} values outside these bounds indicate extraction differences.")
        else:
            logger.info(f"NOTE: Reference bounds are from {settings.dataset2_name}. {settings.dataset1_name} values outside these bounds may reflect population differences.")
        
        # 1. Define bounds from reference dataset
        ref_df = self.dm.df1 if settings.is_paired_validation else self.dm.df2
        ref_name = settings.dataset1_name if settings.is_paired_validation else settings.dataset2_name
        
        lower_pct = settings.thresholds.get('plausibility_lower_pct', 1.0)
        upper_pct = settings.thresholds.get('plausibility_upper_pct', 99.0)
        bounds = {}
        for feature in self.dm.numeric_features:
            data = ref_df[feature].dropna()
            if len(data) > 0:
                bounds[feature] = (np.percentile(data, lower_pct), np.percentile(data, upper_pct))
                
        # 2. Check target against bounds
        target_df = self.dm.df2 if settings.is_paired_validation else self.dm.df1
        target_name = settings.dataset2_name if settings.is_paired_validation else settings.dataset1_name
        id_col = self.dm.get_id_column(target_name)
        
        results = []
        for feature, (lower, upper) in bounds.items():
            data = target_df[feature]
            outside_mask = (data < lower) | (data > upper)
            num_outside = outside_mask.sum()
            valid_len = len(data.dropna())
            pct_outside = (num_outside / valid_len * 100) if valid_len > 0 else 0
            
            extreme_ids = target_df.loc[outside_mask, id_col].dropna().astype(str).head(5).tolist()
            
            results.append({
                'Feature': feature,
                f'Lower_Bound_{ref_name}': lower,
                f'Upper_Bound_{ref_name}': upper,
                'Num_Outside_Range': num_outside,
                'Pct_Outside_Range': round(pct_outside, 2),
                'Extreme_ECG_ID_Examples': ", ".join(extreme_ids) if extreme_ids else "None",
                'Note': f'Bounds from {ref_name}'
            })
            
        report_df = pd.DataFrame(results)
        report_df.to_csv(settings.tables_dir / "plausibility_report.csv", index=False)
        return report_df

    def extract_extremes(self, n: int = 20) -> pd.DataFrame:
        """Extracts top N and bottom N values for each feature (Part 6)."""
        logger.info("Running Univariate: Extreme Value Extraction...")
        results = []
        
        for dataset_name, df in [(settings.dataset1_name, self.dm.df1), (settings.dataset2_name, self.dm.df2)]:
            id_col = self.dm.get_id_column(dataset_name)
            
            for feature in self.dm.numeric_features:
                valid_data = df[[id_col, feature]].dropna()
                if valid_data.empty: continue
                
                largest = valid_data.nlargest(n, feature)
                smallest = valid_data.nsmallest(n, feature)
                
                results.append({
                    'Dataset': dataset_name, 'Feature': feature, 'Extreme_Type': f'Top {n}',
                    'Values': ", ".join([str(round(v, 4)) for v in largest[feature]]),
                    'ECG_IDs': ", ".join([str(v) for v in largest[id_col]])
                })
                
                results.append({
                    'Dataset': dataset_name, 'Feature': feature, 'Extreme_Type': f'Bottom {n}',
                    'Values': ", ".join([str(round(v, 4)) for v in smallest[feature]]),
                    'ECG_IDs': ", ".join([str(v) for v in smallest[id_col]])
                })
                
        res_df = pd.DataFrame(results)
        res_df.to_csv(settings.tables_dir / "feature_extremes.csv", index=False)
        return res_df

    def assess_normality(self) -> pd.DataFrame:
        """Assesses normality using statistical tests (Part 13)."""
        logger.info("Running Univariate: Normality Assessment...")
        results = []
        
        for dataset_name, df in [(settings.dataset1_name, self.dm.df1), (settings.dataset2_name, self.dm.df2)]:
            for feature in self.dm.numeric_features:
                data = df[feature].dropna()
                if len(data) < 20: continue
                
                # Shapiro-Wilk (Subsampled max 5000)
                shapiro_data = data.sample(n=min(5000, len(data)), random_state=42)
                sw_stat, sw_pval = stats.shapiro(shapiro_data)
                
                # Anderson-Darling
                try:
                    ad_result = stats.anderson(data, dist='norm')
                    ad_stat = ad_result.statistic
                    ad_gaussian = ad_stat < ad_result.critical_values[2] # 5% level
                except Exception:
                    ad_stat, ad_gaussian = np.nan, False
                    
                # D'Agostino
                try:
                    k2_stat, k2_pval = stats.normaltest(data)
                except Exception:
                    k2_stat, k2_pval = np.nan, np.nan
                    
                # Practical normality: skewness near 0, kurtosis near 0 (excess)
                skew_threshold = settings.thresholds.get('normality_skew_threshold', 2.0)
                kurt_threshold = settings.thresholds.get('normality_kurtosis_threshold', 7.0)
                alpha = settings.thresholds.get('normality_alpha', 0.05)
                
                skew_val = stats.skew(data)
                kurt_val = stats.kurtosis(data)
                is_practically_normal = (abs(skew_val) < skew_threshold) and (abs(kurt_val) < kurt_threshold)
                
                # Formal test consensus (all must agree)
                formal_consensus = (sw_pval > alpha) and ad_gaussian and (k2_pval > alpha if not np.isnan(k2_pval) else False)
                
                # Use practical criterion for large samples, formal for small
                is_gaussian = is_practically_normal if len(data) > 5000 else formal_consensus
                
                results.append({
                    'Dataset': dataset_name, 'Feature': feature,
                    'N': len(data),
                    'Skewness': skew_val, 'Kurtosis': kurt_val,
                    'Shapiro_Wilk_Stat': sw_stat, 'Shapiro_Wilk_pvalue': sw_pval,
                    'Anderson_Darling_Stat': ad_stat, 'AD_Is_Gaussian_5pct': ad_gaussian,
                    'DAgostino_K2_Stat': k2_stat, 'DAgostino_K2_pvalue': k2_pval,
                    'Practically_Normal': is_practically_normal,
                    'Formal_Consensus': formal_consensus,
                    'Approx_Gaussian': is_gaussian
                })
                
                Plotter.plot_qq(data, dataset_name, feature)
                
        res_df = pd.DataFrame(results)
        res_df.to_csv(settings.tables_dir / "normality_report.csv", index=False)
        return res_df
