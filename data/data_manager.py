import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from core.config import settings
from core.logger import logger
from typing import Tuple, List, Optional

class DataManager:
    """Manages loading, joining, and preprocessing of datasets."""
    
    def __init__(self):
        self.df1: Optional[pd.DataFrame] = None
        self.df2: Optional[pd.DataFrame] = None
        # Matched (inner-joined) versions for paired analyses
        self.df1_matched: Optional[pd.DataFrame] = None
        self.df2_matched: Optional[pd.DataFrame] = None
        self.features: List[str] = []
        self.numeric_features: List[str] = []
        
        # Cached preprocessed data
        self._df_combined_imputed: Optional[pd.DataFrame] = None
        self._X_scaled: Optional[np.ndarray] = None
        self._y_combined: Optional[pd.Series] = None
        
    def load_and_intersect(self) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Loads both datasets, optionally matches records by ID, and filters to overlapping columns."""
        logger.info(f"Loading {settings.dataset1_name} dataset from: {settings.dataset1_path}")
        self.df1 = pd.read_csv(settings.dataset1_path)
        logger.info(f"{settings.dataset1_name} Initial Shape: {self.df1.shape}")
        
        logger.info(f"Loading {settings.dataset2_name} dataset from: {settings.dataset2_path}")
        self.df2 = pd.read_csv(settings.dataset2_path)
        logger.info(f"{settings.dataset2_name} Initial Shape: {self.df2.shape}")
        
        # Normalize ID columns to a common 'ecg_id' for Re-extracted dataset if needed
        if 'SubjectID' in self.df2.columns and 'ecg_id' not in self.df2.columns:
            self.df2 = self.df2.rename(columns={'SubjectID': 'ecg_id'})
            logger.info("Renamed 'SubjectID' to 'ecg_id' in dataset2 for matching.")
            
        if 'SubjectID' in self.df1.columns and 'ecg_id' not in self.df1.columns:
            self.df1 = self.df1.rename(columns={'SubjectID': 'ecg_id'})
            logger.info("Renamed 'SubjectID' to 'ecg_id' in dataset1 for matching.")
        
        # Identify overlapping columns
        cols1 = set(self.df1.columns)
        cols2 = set(self.df2.columns)
        self.features = sorted(cols1.intersection(cols2))
        logger.info(f"Found {len(self.features)} overlapping features.")
        
        # Determine ID column to keep
        id_col_1 = self.get_id_column(settings.dataset1_name)
        id_col_2 = self.get_id_column(settings.dataset2_name)
        
        # Match records via inner join if in extraction validation mode
        if settings.is_paired_validation:
            self._match_records()
        
        # Keep only overlapping features + ID column
        keep_cols1 = list(set(self.features + [id_col_1]))
        keep_cols2 = list(set(self.features + [id_col_2]))
        
        self.df1 = self.df1[[c for c in keep_cols1 if c in self.df1.columns]].copy()
        self.df2 = self.df2[[c for c in keep_cols2 if c in self.df2.columns]].copy()
        
        # Drop exact duplicate rows
        df1_dups = self.df1.duplicated().sum()
        df2_dups = self.df2.duplicated().sum()
        if df1_dups > 0:
            logger.warning(f"{settings.dataset1_name}: Dropping {df1_dups} exact duplicate rows.")
            self.df1 = self.df1.drop_duplicates().reset_index(drop=True)
        if df2_dups > 0:
            logger.warning(f"{settings.dataset2_name}: Dropping {df2_dups} exact duplicate rows.")
            self.df2 = self.df2.drop_duplicates().reset_index(drop=True)
        
        # Identify numeric features (exclude ID columns)
        feature_cols = [f for f in self.features if f not in [id_col_1, id_col_2]]
        self.numeric_features = [f for f in feature_cols if pd.api.types.is_numeric_dtype(self.df1[f])]
        
        return self.df1, self.df2, self.features

    def _match_records(self):
        """Matches records between datasets using ecg_id."""
        logger.info("=" * 60)
        logger.info("RECORD MATCHING REPORT")
        logger.info("=" * 60)
        
        id_col = 'ecg_id'
        
        n_df1 = len(self.df1)
        n_df2 = len(self.df2)
        
        # Check for duplicates
        dup_df1 = self.df1[id_col].duplicated().sum() if id_col in self.df1.columns else 0
        dup_df2 = self.df2[id_col].duplicated().sum() if id_col in self.df2.columns else 0
        
        # Perform inner join to find matched records
        df1_ids = set(self.df1[id_col].dropna()) if id_col in self.df1.columns else set()
        df2_ids = set(self.df2[id_col].dropna()) if id_col in self.df2.columns else set()
        
        common_ids = df1_ids.intersection(df2_ids)
        only_df1 = df1_ids - df2_ids
        only_df2 = df2_ids - df1_ids
        
        match_pct = (len(common_ids) / len(df1_ids) * 100) if len(df1_ids) > 0 else 0
        
        logger.info(f"  Records in {settings.dataset1_name}:        {n_df1}")
        logger.info(f"  Records in {settings.dataset2_name}:        {n_df2}")
        logger.info(f"  Successfully matched:              {len(common_ids)}")
        logger.info(f"  Present only in {settings.dataset1_name}:   {len(only_df1)}")
        logger.info(f"  Present only in {settings.dataset2_name}:   {len(only_df2)}")
        logger.info(f"  Duplicate IDs in {settings.dataset1_name}:  {dup_df1}")
        logger.info(f"  Duplicate IDs in {settings.dataset2_name}:  {dup_df2}")
        logger.info(f"  Match percentage:                  {match_pct:.2f}%")
        logger.info("=" * 60)
        
        # Create matched DataFrames via inner join
        feature_cols_for_join = [c for c in self.features if c != id_col]
        
        df_orig_for_join = self.df1[[id_col] + [c for c in feature_cols_for_join if c in self.df1.columns]].copy()
        df_reext_for_join = self.df2[[id_col] + [c for c in feature_cols_for_join if c in self.df2.columns]].copy()
        
        # Drop duplicates on ID before joining (keep first)
        df_orig_for_join = df_orig_for_join.drop_duplicates(subset=[id_col], keep='first')
        df_reext_for_join = df_reext_for_join.drop_duplicates(subset=[id_col], keep='first')
        
        # Inner join
        merged = pd.merge(df_orig_for_join, df_reext_for_join, on=id_col, suffixes=('_dataset1', '_dataset2'))
        
        # Split back into matched DataFrames with aligned rows
        cols1 = [id_col] + [c + '_dataset1' for c in feature_cols_for_join if c + '_dataset1' in merged.columns]
        cols2 = [id_col] + [c + '_dataset2' for c in feature_cols_for_join if c + '_dataset2' in merged.columns]
        
        self.df1_matched = merged[cols1].copy()
        self.df1_matched.columns = [id_col] + [c.replace('_dataset1', '') for c in cols1 if c != id_col]
        
        self.df2_matched = merged[cols2].copy()
        self.df2_matched.columns = [id_col] + [c.replace('_dataset2', '') for c in cols2 if c != id_col]
        
        logger.info(f"Matched {settings.dataset1_name} shape: {self.df1_matched.shape}")
        logger.info(f"Matched {settings.dataset2_name} shape: {self.df2_matched.shape}")
        
        # Save matching report
        matching_report = pd.DataFrame([{
            'Metric': f'Records in {settings.dataset1_name}', 'Value': n_df1,
        }, {
            'Metric': f'Records in {settings.dataset2_name}', 'Value': n_df2,
        }, {
            'Metric': 'Successfully matched', 'Value': len(common_ids),
        }, {
            'Metric': f'Present only in {settings.dataset1_name}', 'Value': len(only_df1),
        }, {
            'Metric': f'Present only in {settings.dataset2_name}', 'Value': len(only_df2),
        }, {
            'Metric': f'Duplicate IDs in {settings.dataset1_name}', 'Value': dup_df1,
        }, {
            'Metric': f'Duplicate IDs in {settings.dataset2_name}', 'Value': dup_df2,
        }, {
            'Metric': 'Match percentage (%)', 'Value': round(match_pct, 2),
        }])
        matching_report.to_csv(settings.tables_dir / "record_matching_report.csv", index=False)
        logger.info(f"Matching report saved to {settings.tables_dir / 'record_matching_report.csv'}")

    def get_id_column(self, dataset_name: str) -> str:
        """Determines the ID column for a given dataset."""
        df = self.df1 if dataset_name == settings.dataset1_name else self.df2
        if 'ecg_id' in df.columns:
            return 'ecg_id'
        elif 'SubjectID' in df.columns:
            return 'SubjectID'
        else:
            return df.columns[0]

    def get_combined_imputed_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns median-imputed combined numeric dataset and dataset labels."""
        if self._df_combined_imputed is None:
            logger.info("Computing combined imputed dataset...")
            df1_sub = self.df1[self.numeric_features].copy()
            df2_sub = self.df2[self.numeric_features].copy()
            df_combined = pd.concat([df1_sub, df2_sub], ignore_index=True)
            self._y_combined = pd.Series([settings.dataset1_name]*len(df1_sub) + [settings.dataset2_name]*len(df2_sub))
            
            missing_before = df_combined.isnull().sum()
            total_missing = missing_before.sum()
            if total_missing > 0:
                logger.info(f"Median imputation: filling {total_missing} missing values across {(missing_before > 0).sum()} features.")
            else:
                logger.info("No missing values found; no imputation needed.")
            
            self._df_combined_imputed = df_combined.fillna(df_combined.median())
            
        return self._df_combined_imputed, self._y_combined
        
    def get_combined_scaled_data(self) -> Tuple[np.ndarray, pd.Series]:
        """Returns standard scaled combined imputed data."""
        if self._X_scaled is None:
            df_imputed, y = self.get_combined_imputed_data()
            logger.info("Computing combined scaled dataset...")
            self._X_scaled = StandardScaler().fit_transform(df_imputed)
        return self._X_scaled, self._y_combined
