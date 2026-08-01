import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import List, Tuple
from core.config import settings
from core.logger import logger
from data.data_manager import DataManager
from visualization.plots import Plotter
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import umap
from statsmodels.stats.outliers_influence import variance_inflation_factor

class MultivariateAnalysis:
    """Handles N-D statistical analysis: Outliers, Correlation, Redundancy, Dimensionality, Batch Effect."""
    
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager

    def detect_outliers(self) -> pd.DataFrame:
        """Computes univariate and multivariate outliers (Part 5)."""
        logger.info("Running Multivariate: Outlier Detection...")
        results = []
        
        for dataset_name, df in [(settings.dataset1_name, self.dm.df1), (settings.dataset2_name, self.dm.df2)]:
            # Multivariate
            df_imputed = df[self.dm.numeric_features].fillna(df[self.dm.numeric_features].median())
            if df_imputed.empty or len(df_imputed) < 50:
                continue
                
            X_scaled = StandardScaler().fit_transform(df_imputed)
            
            # Isolation Forest
            iso = IsolationForest(contamination=settings.thresholds.get('outlier_contamination', 0.01), random_state=42)
            out_iso = pd.Series(iso.fit_predict(X_scaled) == -1, index=df.index)
            
            # LOF
            lof = LocalOutlierFactor(contamination=settings.thresholds.get('outlier_contamination', 0.01))
            out_lof = pd.Series(lof.fit_predict(X_scaled) == -1, index=df.index)
            
            # Mahalanobis
            out_mah = pd.Series(False, index=df.index)
            try:
                cov_matrix = np.cov(X_scaled, rowvar=False)
                inv_cov_matrix = np.linalg.pinv(cov_matrix)
                mean_vec = np.mean(X_scaled, axis=0)
                diff = X_scaled - mean_vec
                md = np.sqrt(np.sum(np.dot(diff, inv_cov_matrix) * diff, axis=1))
                threshold = np.sqrt(stats.chi2.ppf(1 - settings.thresholds.get('mahalanobis_alpha', 0.001), df=X_scaled.shape[1]))
                out_mah = pd.Series(md > threshold, index=df.index)
            except np.linalg.LinAlgError:
                logger.warning(f"Mahalanobis calculation failed for {dataset_name} due to LinAlgError.")
            
            # Record-level summary
            results.extend([
                {'Dataset': dataset_name, 'Analysis_Type': 'Multivariate', 'Feature_or_Method': 'All', 'Method': 'Isolation Forest', 'Pct_Outliers': round((out_iso.sum() / len(df)) * 100, 2)},
                {'Dataset': dataset_name, 'Analysis_Type': 'Multivariate', 'Feature_or_Method': 'All', 'Method': 'LOF', 'Pct_Outliers': round((out_lof.sum() / len(df)) * 100, 2)},
                {'Dataset': dataset_name, 'Analysis_Type': 'Multivariate', 'Feature_or_Method': 'All', 'Method': 'Mahalanobis', 'Pct_Outliers': round((out_mah.sum() / len(df)) * 100, 2)}
            ])
            
            # PCA Visuals for Outliers
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            Plotter.plot_pca_outliers(X_pca, out_iso, out_lof, out_mah, dataset_name)
            
            # Univariate Outliers
            for feature in self.dm.numeric_features:
                data = df[feature]
                Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
                IQR = Q3 - Q1
                out_iqr = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
                
                z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
                out_z = z_scores > 3
                
                median = data.median()
                mad = stats.median_abs_deviation(data.dropna())
                out_mad = (np.abs(0.6745 * (data - median) / mad) > 3.5) if mad > 0 else pd.Series(False, index=data.index)
                
                results.extend([
                    {'Dataset': dataset_name, 'Analysis_Type': 'Univariate', 'Feature_or_Method': feature, 'Method': 'IQR', 'Pct_Outliers': round((out_iqr.sum() / len(df)) * 100, 2)},
                    {'Dataset': dataset_name, 'Analysis_Type': 'Univariate', 'Feature_or_Method': feature, 'Method': 'Z-Score', 'Pct_Outliers': round((out_z.sum() / len(df)) * 100, 2)},
                    {'Dataset': dataset_name, 'Analysis_Type': 'Univariate', 'Feature_or_Method': feature, 'Method': 'MAD', 'Pct_Outliers': round((out_mad.sum() / len(df)) * 100, 2)}
                ])
                
        out_df = pd.DataFrame(results)
        out_df.to_csv(settings.tables_dir / "outlier_summary.csv", index=False)
        return out_df

    def analyze_correlations(self) -> pd.DataFrame:
        """Computes correlation matrices and finds highly correlated pairs (Part 7)."""
        logger.info("Running Multivariate: Correlation Analysis...")
        methods = ['pearson', 'spearman', 'kendall']
        high_corr_list = []
        
        for dataset_name, df in [(settings.dataset1_name, self.dm.df1), (settings.dataset2_name, self.dm.df2)]:
            numeric_df = df[self.dm.numeric_features]
            if numeric_df.empty: continue
            
            for method in methods:
                corr_matrix = numeric_df.corr(method=method)
                Plotter.plot_correlation_heatmap(corr_matrix, dataset_name, method)
                
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                s = upper.unstack().dropna()
                
                for threshold in settings.thresholds.get('correlation_high', [0.90, 0.95, 0.99]):
                    high_pairs = s[s.abs() > threshold]
                    for (feat1, feat2), val in high_pairs.items():
                        high_corr_list.append({
                            'Dataset': dataset_name, 'Method': method.capitalize(), 'Threshold': f"> {threshold}",
                            'Feature_1': feat1, 'Feature_2': feat2, 'Correlation': val
                        })
                        
        hc_df = pd.DataFrame(high_corr_list).drop_duplicates()
        hc_df.to_csv(settings.tables_dir / "high_correlation_pairs.csv", index=False)
        return hc_df

    def detect_redundancy(self) -> pd.DataFrame:
        """Calculates VIF and constant features (Part 8)."""
        logger.info("Running Multivariate: Redundancy Analysis...")
        results = []
        
        for dataset_name, df in [(settings.dataset1_name, self.dm.df1), (settings.dataset2_name, self.dm.df2)]:
            numeric_df = df[self.dm.numeric_features]
            df_imputed = numeric_df.fillna(numeric_df.median())
            
            constant_cols = [col for col in numeric_df.columns if numeric_df[col].nunique() <= 1]
            near_constant_cols = [col for col in numeric_df.columns if col not in constant_cols and (numeric_df[col].var() < 1e-4 or numeric_df[col].value_counts(normalize=True).max() >= 0.99)]
            
            df_vif = df_imputed.drop(columns=constant_cols)
            vif_data = pd.DataFrame({'Feature': df_vif.columns})
            
            try:
                vif_data["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(len(df_vif.columns))]
            except Exception as e:
                logger.warning(f"VIF calculation failed for {dataset_name}: {e}")
                vif_data["VIF"] = np.nan
            
            # Cap infinite VIF values and flag them
            vif_data['VIF_Is_Infinite'] = np.isinf(vif_data['VIF'])
            vif_data['VIF'] = vif_data['VIF'].replace([np.inf, -np.inf], 9999.0)
            if vif_data['VIF_Is_Infinite'].any():
                inf_count = vif_data['VIF_Is_Infinite'].sum()
                logger.warning(f"{dataset_name}: {inf_count} features have infinite VIF (perfect multicollinearity). Capped at 9999.")
                
            for col in constant_cols:
                vif_data = pd.concat([vif_data, pd.DataFrame({"Feature": [col], "VIF": [np.nan]})], ignore_index=True)
                
            vif_data['Dataset'] = dataset_name
            vif_data['Is_Constant'] = vif_data['Feature'].isin(constant_cols)
            vif_data['Is_Near_Constant'] = vif_data['Feature'].isin(near_constant_cols)
            
            def recommend(row):
                if row['Is_Constant']: return "Remove (Constant)"
                if row['Is_Near_Constant']: return "Review for Removal (Near Constant)"
                if pd.notna(row['VIF']) and row['VIF'] > settings.thresholds.get('vif_high', 10.0): return "Review for Removal (High VIF)"
                return "Keep"
                
            vif_data['Recommendation'] = vif_data.apply(recommend, axis=1)
            results.append(vif_data)
            
        combined = pd.concat(results, ignore_index=True)
        combined.to_csv(settings.tables_dir / "feature_redundancy.csv", index=False)
        return combined

    def run_dimensionality(self):
        """Runs PCA and UMAP (Part 9)."""
        logger.info("Running Multivariate: Dimensionality Analysis...")
        for dataset_name, df in [(settings.dataset1_name, self.dm.df1), (settings.dataset2_name, self.dm.df2)]:
            df_imputed = df[self.dm.numeric_features].fillna(df[self.dm.numeric_features].median())
            if df_imputed.empty or len(df_imputed) < 10: continue
            
            X_scaled = StandardScaler().fit_transform(df_imputed)
            
            pca = PCA(n_components=min(10, len(df_imputed.columns)))
            X_pca = pca.fit_transform(X_scaled)
            
            # Loadings
            loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_), columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=df_imputed.columns)
            loadings.to_csv(settings.tables_dir / f"{dataset_name}_pca_loadings.csv")
            
            # UMAP
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            X_umap = reducer.fit_transform(X_scaled)
            
            explained_var_ratio = pca.explained_variance_ratio_
            cum_explained_var = np.cumsum(explained_var_ratio)
            
            Plotter.plot_dimensionality(explained_var_ratio, cum_explained_var, dataset_name, X_pca, list(explained_var_ratio), X_umap)

    def analyze_relationships(self):
        """Plots specific clinical relationships (Part 14)."""
        logger.info("Running Multivariate: Feature Relationships...")
        target_pairs = [
            ('RR_Mean_Global', 'QRS_Dur_Global'),
            ('S_Amp_V1', 'R_Amp_V5'),
            ('ST_Elev_V2', 'ST_Elev_V3'),
            ('QRS_Dur_Global', 'R_Amp_I'),
            ('Q_Amp_III', 'T_Amp_III'),
        ]
        
        for dataset_name, df in [(settings.dataset1_name, self.dm.df1), (settings.dataset2_name, self.dm.df2)]:
            for x_col, y_col in target_pairs:
                if x_col in df.columns and y_col in df.columns:
                    Plotter.plot_relationship(df, dataset_name, x_col, y_col)

    def detect_batch_effects(self):
        """Detects batch effects using PCA, UMAP, and Random Forest (Part 12)."""
        logger.info("Running Multivariate: Batch Effect Detection...")
        X_scaled, y_combined = self.dm.get_combined_scaled_data()
        
        # Subsample for visuals
        df_full = pd.DataFrame(X_scaled, columns=self.dm.numeric_features)
        df_full['Dataset'] = y_combined.values
        
        sample1 = df_full[df_full['Dataset'] == settings.dataset1_name].sample(n=min(10000, sum(y_combined == settings.dataset1_name)), random_state=42)
        sample2 = df_full[df_full['Dataset'] == settings.dataset2_name].sample(n=min(10000, sum(y_combined == settings.dataset2_name)), random_state=42)
        
        X_sample = pd.concat([sample1, sample2]).drop(columns=['Dataset']).values
        y_sample = pd.concat([sample1['Dataset'], sample2['Dataset']])
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_sample)
        
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X_sample)
        
        Plotter.plot_batch_effects(X_pca, X_umap, y_sample)
        
        # XGBoost Classification (using CUDA)
        y_encoded = (y_combined == settings.dataset1_name).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
        
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42, 
                tree_method="hist", 
                device="cuda",
                eval_metric="logloss"
            )
            logger.info("Training XGBoost Classifier (CUDA enabled) for batch effect detection...")
        except ImportError:
            logger.warning("XGBoost not found or CUDA not supported. Falling back to sklearn RandomForest (CPU).")
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        feat_imp = pd.DataFrame({'Feature': self.dm.numeric_features, 'Importance': clf.feature_importances_}).sort_values('Importance', ascending=False)
        feat_imp.to_csv(settings.tables_dir / "batch_effect_importance.csv", index=False)
        
        report_path = settings.tables_dir / "batch_effect_report.txt"
        with open(report_path, "w") as f:
            f.write("Batch Effect / Dataset Difference Detection Report\n")
            f.write("=====================================================\n")
            f.write(f"Classifier Accuracy (predicting {settings.dataset1_name} vs {settings.dataset2_name}): {acc:.4f}\n")
            f.write("Note: An accuracy close to 0.50 means the datasets are indistinguishable.\n")
            f.write("An accuracy close to 1.00 indicates systematic differences.\n\n")
            
            if settings.is_paired_validation:
                f.write("In extraction validation, high accuracy indicates differences in the extraction process.\n\n")
            else:
                f.write("IMPORTANT CONTEXT: High accuracy is expected here because the datasets differ\n")
                f.write("in patient population, recording hardware, and clinical context.\n\n")
                
            f.write("Top 10 Most Distinguishing Features:\n")
            for _, row in feat_imp.head(10).iterrows():
                f.write(f"- {row['Feature']}: {row['Importance']:.4f}\n")
                
        logger.info(f"Batch effect accuracy: {acc:.4f}")
