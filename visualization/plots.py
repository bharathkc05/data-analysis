import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import missingno as msno
import os
from typing import List, Optional
from core.config import settings
from core.logger import logger
import scipy.stats as stats

class Plotter:
    """Centralized plotting utilities to avoid redundancy."""
    
    @staticmethod
    def setup_style():
        sns.set_context("paper", font_scale=1.5)
        sns.set_style("whitegrid")
        
    @staticmethod
    def plot_missingness(df: pd.DataFrame, dataset_name: str, cols_with_missing: List[str]):
        """Generates missingness plots for a dataset."""
        logger.info(f"Generating missingness plots for {dataset_name}...")
        
        plt.figure(figsize=(20, 10))
        msno.matrix(df, sparkline=False)
        plt.title(f"{dataset_name} Missingness Matrix", fontsize=20)
        plt.savefig(settings.sub_fig_dirs["missingness"] / f"missing_matrix_{dataset_name}.{settings.plot_format}", 
                    dpi=settings.plot_dpi, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(20, 10))
        msno.bar(df)
        plt.title(f"{dataset_name} Missingness Bar Plot", fontsize=20)
        plt.savefig(settings.sub_fig_dirs["missingness"] / f"missing_bar_{dataset_name}.{settings.plot_format}", 
                    dpi=settings.plot_dpi, bbox_inches='tight')
        plt.close()
        
        if len(cols_with_missing) > 1:
            plt.figure(figsize=(15, 12))
            msno.heatmap(df[cols_with_missing])
            plt.title(f"{dataset_name} Missingness Heatmap (Correlation)", fontsize=20)
            plt.savefig(settings.sub_fig_dirs["missingness"] / f"missing_heatmap_{dataset_name}.{settings.plot_format}", 
                        dpi=settings.plot_dpi, bbox_inches='tight')
            plt.close()

    @staticmethod
    def plot_comparative_distribution(data1: pd.Series, data2: pd.Series, feature: str):
        """Generates a 4-panel distribution comparison for a feature."""
        if len(data1) == 0 and len(data2) == 0:
            return
            
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f'Distribution Analysis: {feature}', fontsize=24, y=1.02)
        gs = fig.add_gridspec(2, 2)
        
        # Plot 1: Histogram & KDE overlay
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(data1, color=settings.colors[settings.dataset1_name], label=settings.dataset1_name, kde=True, stat="density", ax=ax1, alpha=0.5, bins=50)
        sns.histplot(data2, color=settings.colors[settings.dataset2_name], label=settings.dataset2_name, kde=True, stat="density", ax=ax1, alpha=0.5, bins=50)
        ax1.set_title("Histogram & KDE Overlay")
        ax1.legend()
        
        df_plot = pd.concat([
            pd.DataFrame({feature: data1, 'Dataset': settings.dataset1_name}),
            pd.DataFrame({feature: data2, 'Dataset': settings.dataset2_name})
        ], ignore_index=True)
        
        # Plot 2: Boxplot
        ax2 = fig.add_subplot(gs[0, 1])
        sns.boxplot(x='Dataset', y=feature, data=df_plot, hue='Dataset', palette=settings.colors, legend=False, ax=ax2)
        ax2.set_title("Boxplot")
        
        # Plot 3: Violinplot
        ax3 = fig.add_subplot(gs[1, 0])
        sns.violinplot(x='Dataset', y=feature, data=df_plot, hue='Dataset', palette=settings.colors, legend=False, ax=ax3, cut=0)
        ax3.set_title("Violin Plot")
        
        # Plot 4: ECDF
        ax4 = fig.add_subplot(gs[1, 1])
        sns.ecdfplot(data1, color=settings.colors[settings.dataset1_name], label=settings.dataset1_name, ax=ax4)
        sns.ecdfplot(data2, color=settings.colors[settings.dataset2_name], label=settings.dataset2_name, ax=ax4)
        ax4.set_title("Empirical Cumulative Distribution Function (ECDF)")
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(settings.sub_fig_dirs["distributions"] / f"{feature}_distribution.{settings.plot_format}", 
                    dpi=settings.plot_dpi, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def plot_qq(data: pd.Series, dataset_name: str, feature: str):
        """Generates a Q-Q plot for normality assessment."""
        fig = plt.figure(figsize=(8, 6))
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f"{dataset_name} Q-Q Plot: {feature}")
        plt.tight_layout()
        plt.savefig(settings.sub_fig_dirs["normality"] / f"{dataset_name}_{feature}_qq.{settings.plot_format}", 
                    dpi=settings.plot_dpi)
        plt.close(fig)
        
    @staticmethod
    def plot_pca_outliers(X_pca: np.ndarray, out_iso: pd.Series, out_lof: pd.Series, out_mah: pd.Series, dataset_name: str):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'{dataset_name}: PCA Visualization of Multivariate Outliers', fontsize=18)
        
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=out_iso, ax=axes[0], palette=['#1f77b4', '#d62728'], alpha=0.5)
        axes[0].set_title('Isolation Forest Outliers')
        
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=out_lof, ax=axes[1], palette=['#1f77b4', '#d62728'], alpha=0.5)
        axes[1].set_title('LOF Outliers')
        
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=out_mah, ax=axes[2], palette=['#1f77b4', '#d62728'], alpha=0.5)
        axes[2].set_title('Mahalanobis Outliers')
        
        plt.tight_layout()
        plt.savefig(settings.sub_fig_dirs["outliers"] / f"pca_outliers_{dataset_name}.{settings.plot_format}", 
                    dpi=settings.plot_dpi)
        plt.close()

    @staticmethod
    def plot_correlation_heatmap(corr_matrix: pd.DataFrame, dataset_name: str, method: str):
        plt.figure(figsize=(24, 20))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title(f"{dataset_name} {method.capitalize()} Correlation Matrix", fontsize=24)
        plt.tight_layout()
        plt.savefig(settings.sub_fig_dirs["correlations"] / f"{dataset_name}_{method}_heatmap.{settings.plot_format}", 
                    dpi=settings.plot_dpi)
        plt.close()

    @staticmethod
    def plot_dimensionality(explained_var_ratio: np.ndarray, cum_explained_var: np.ndarray, dataset_name: str,
                            X_pca: np.ndarray, pca_explained: List[float], X_umap: Optional[np.ndarray]):
        # Variance Plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_var_ratio) + 1), explained_var_ratio, alpha=0.7, align='center', label='Individual explained variance')
        plt.step(range(1, len(cum_explained_var) + 1), cum_explained_var, where='mid', label='Cumulative explained variance')
        plt.ylabel('Explained Variance Ratio')
        plt.xlabel('Principal Components')
        plt.title(f'{dataset_name} - PCA Explained Variance')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(settings.sub_fig_dirs["dimensionality"] / f"{dataset_name}_pca_variance.{settings.plot_format}", 
                    dpi=settings.plot_dpi)
        plt.close()
        
        # PCA 2D Plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], alpha=0.5, color=settings.colors.get(dataset_name, 'blue'))
        plt.title(f'{dataset_name} - PCA 2D Projection', fontsize=16)
        plt.xlabel(f'PC1 ({pca_explained[0]:.2%} var)')
        plt.ylabel(f'PC2 ({pca_explained[1]:.2%} var)')
        plt.tight_layout()
        plt.savefig(settings.sub_fig_dirs["dimensionality"] / f"{dataset_name}_pca_2d.{settings.plot_format}", 
                    dpi=settings.plot_dpi)
        plt.close()
        
        # UMAP 2D Plot
        if X_umap is not None:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], alpha=0.5, color=settings.colors.get(dataset_name, 'blue'))
            plt.title(f'{dataset_name} - UMAP 2D Projection', fontsize=16)
            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')
            plt.tight_layout()
            plt.savefig(settings.sub_fig_dirs["dimensionality"] / f"{dataset_name}_umap_2d.{settings.plot_format}", 
                        dpi=settings.plot_dpi)
            plt.close()

    @staticmethod
    def plot_batch_effects(X_pca: np.ndarray, X_umap: np.ndarray, y_sample: pd.Series):
        title_suffix = f"({settings.dataset1_name} vs {settings.dataset2_name})"
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_sample, palette=settings.colors, alpha=0.3)
        plt.title(f'Combined PCA {title_suffix}', fontsize=16)
        plt.tight_layout()
        plt.savefig(settings.sub_fig_dirs["batch_effects"] / f"combined_pca.{settings.plot_format}", 
                    dpi=settings.plot_dpi)
        plt.close()
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y_sample, palette=settings.colors, alpha=0.3)
        plt.title(f'Combined UMAP {title_suffix}', fontsize=16)
        plt.tight_layout()
        plt.savefig(settings.sub_fig_dirs["batch_effects"] / f"combined_umap.{settings.plot_format}", 
                    dpi=settings.plot_dpi)
        plt.close()
        
    @staticmethod
    def plot_relationship(df: pd.DataFrame, dataset_name: str, x_col: str, y_col: str):
        data = df[[x_col, y_col]].dropna()
        if len(data) < 10: return
        
        plt.figure(figsize=(10, 8))
        sns.regplot(
            data=data, 
            x=x_col, 
            y=y_col, 
            scatter_kws={'alpha': 0.1, 'color': settings.colors.get(dataset_name, 'blue')},
            line_kws={'color': 'red', 'linewidth': 2}
        )
        plt.title(f"{dataset_name} Relationship: {y_col} vs {x_col}", fontsize=16)
        plt.tight_layout()
        plt.savefig(settings.sub_fig_dirs["relationships"] / f"{dataset_name}_{y_col}_vs_{x_col}.{settings.plot_format}", 
                    dpi=settings.plot_dpi)
        plt.close()

    @staticmethod
    def plot_detailed_comparison(data1: pd.Series, data2: pd.Series, feature: str):
        fig = plt.figure(figsize=(24, 18))
        title_suffix = f"({settings.dataset1_name} vs {settings.dataset2_name})"
        fig.suptitle(f'Comparative Analysis: {feature} {title_suffix}', fontsize=24, y=1.02)
        gs = fig.add_gridspec(3, 2)
        
        # Plot 1: Overlaid Histograms & KDE
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(data1, color=settings.colors[settings.dataset1_name], label=settings.dataset1_name, kde=True, stat="density", ax=ax1, alpha=0.5, bins=50)
        sns.histplot(data2, color=settings.colors[settings.dataset2_name], label=settings.dataset2_name, kde=True, stat="density", ax=ax1, alpha=0.5, bins=50)
        ax1.set_title("Overlaid Histograms & KDE")
        ax1.legend()
        
        # Plot 2: QQ Plot
        ax2 = fig.add_subplot(gs[0, 1])
        quantiles = np.linspace(0.01, 0.99, 100)
        q1 = np.quantile(data1, quantiles)
        q2 = np.quantile(data2, quantiles)
        ax2.scatter(q1, q2, color='#2ca02c')
        
        min_q = min(min(q1), min(q2))
        max_q = max(max(q1), max(q2))
        ax2.plot([min_q, max_q], [min_q, max_q], 'k--')
        ax2.set_xlabel(f"{settings.dataset1_name} Quantiles")
        ax2.set_ylabel(f"{settings.dataset2_name} Quantiles")
        ax2.set_title(f"Q-Q Plot {title_suffix}")
        
        # Plot 3: ECDF
        ax3 = fig.add_subplot(gs[1, 0])
        sns.ecdfplot(data1, color=settings.colors[settings.dataset1_name], label=settings.dataset1_name, ax=ax3)
        sns.ecdfplot(data2, color=settings.colors[settings.dataset2_name], label=settings.dataset2_name, ax=ax3)
        ax3.set_title("ECDF Comparison")
        ax3.legend()
        
        df_plot = pd.concat([
            pd.DataFrame({feature: data1, 'Dataset': settings.dataset1_name}),
            pd.DataFrame({feature: data2, 'Dataset': settings.dataset2_name})
        ], ignore_index=True)
        
        # Plot 4: Violin Plot
        ax4 = fig.add_subplot(gs[1, 1])
        sns.violinplot(x='Dataset', y=feature, data=df_plot, hue='Dataset', palette=settings.colors, legend=False, ax=ax4, cut=0)
        ax4.set_title("Violin Plot")
        
        # Plot 5: Boxplot
        ax5 = fig.add_subplot(gs[2, :])
        sns.boxplot(x='Dataset', y=feature, data=df_plot, hue='Dataset', palette=settings.colors, legend=False, ax=ax5, orient='v')
        ax5.set_title("Comparative Boxplot")
        
        plt.tight_layout()
        plt.savefig(settings.sub_fig_dirs["comparisons"] / f"{feature}_comparison.{settings.plot_format}", 
                    dpi=settings.plot_dpi, bbox_inches='tight')
        plt.close(fig)

    # ==========================================
    # Paired Validation Plots
    # ==========================================
    
    @staticmethod
    def plot_bland_altman(ba_mean: np.ndarray, ba_diff: np.ndarray, mean_diff: float, 
                          loa_upper: float, loa_lower: float, feature: str):
        """Generates a Bland-Altman plot for agreement analysis."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.scatter(ba_mean, ba_diff, alpha=0.15, color='#2ca02c', s=10)
        ax.axhline(y=mean_diff, color='#d62728', linestyle='-', linewidth=2, label=f'Mean Diff: {mean_diff:.4f}')
        ax.axhline(y=loa_upper, color='#ff7f0e', linestyle='--', linewidth=1.5, label=f'+1.96 SD: {loa_upper:.4f}')
        ax.axhline(y=loa_lower, color='#ff7f0e', linestyle='--', linewidth=1.5, label=f'-1.96 SD: {loa_lower:.4f}')
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_xlabel(f'Mean of {settings.dataset1_name} & {settings.dataset2_name}', fontsize=14)
        ax.set_ylabel(f'Difference ({settings.dataset2_name} − {settings.dataset1_name})', fontsize=14)
        ax.set_title(f'Bland-Altman Plot: {feature}', fontsize=18)
        ax.legend(loc='best', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(settings.sub_fig_dirs["paired_validation"] / f"{feature}_bland_altman.{settings.plot_format}",
                    dpi=settings.plot_dpi, bbox_inches='tight')
        plt.close(fig)
    
    @staticmethod
    def plot_paired_scatter(valid1: np.ndarray, valid2: np.ndarray, feature: str, pearson_r: float):
        """Generates a scatter plot of Original vs Re-extracted values with identity line."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.scatter(valid1, valid2, alpha=0.15, color='#1f77b4', s=10)
        
        # Identity line
        all_vals = np.concatenate([valid1, valid2])
        min_val, max_val = np.min(all_vals), np.max(all_vals)
        margin = (max_val - min_val) * 0.05
        ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 
                'k--', linewidth=1.5, label='Identity (y=x)')
        
        ax.set_xlabel(settings.dataset1_name, fontsize=14)
        ax.set_ylabel(settings.dataset2_name, fontsize=14)
        r_str = f'{pearson_r:.4f}' if not np.isnan(pearson_r) else 'N/A'
        ax.set_title(f'Paired Agreement: {feature}\n(Pearson r = {r_str})', fontsize=16)
        ax.legend(loc='best', fontsize=11)
        ax.set_aspect('equal', adjustable='datalim')
        
        plt.tight_layout()
        plt.savefig(settings.sub_fig_dirs["paired_validation"] / f"{feature}_paired_scatter.{settings.plot_format}",
                    dpi=settings.plot_dpi, bbox_inches='tight')
        plt.close(fig)
