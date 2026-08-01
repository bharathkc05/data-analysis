import os
from pathlib import Path
import yaml
import logging

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.config_path = self.base_dir / config_path
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)
            
        # Pipeline Mode
        self.mode = self._config.get("mode", "extraction_validation")
        
        # Plotting & Thresholds
        self.plot_dpi = self._config['plotting']['dpi']
        self.plot_format = self._config['plotting']['format']
        all_colors = self._config['plotting']['colors']
        self.thresholds = self._config.get('thresholds', {})
        
        if self.mode == "extraction_validation":
            self.dataset1_name = "PTB-XL Original"
            self.dataset2_name = "PTB-XL Re-extracted"
            self.dataset1_path = Path(self._config['paths']['ptbxl_original_csv'])
            self.dataset2_path = Path(self._config['paths']['ptbxl_reextracted_csv'])
            self.is_paired_validation = True
            output_subdir = "validation"
        elif self.mode == "cross_dataset_comparison":
            self.dataset1_name = "MIMIC-IV"
            self.dataset2_name = "PTB-XL Original"
            self.dataset1_path = Path(self._config['paths']['mimic_csv'])
            self.dataset2_path = Path(self._config['paths']['ptbxl_original_csv'])
            self.is_paired_validation = False
            output_subdir = "comparison"
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        self.colors = {
            self.dataset1_name: all_colors.get(self.dataset1_name, "#1f77b4"),
            self.dataset2_name: all_colors.get(self.dataset2_name, "#ff7f0e")
        }
        
        # Directories
        self.output_dir = self.base_dir / "outputs"
        self.tables_dir = self.output_dir / "tables" / output_subdir
        self.fig_dir = self.output_dir / "figures"
        self.report_dir = self.output_dir / "report" / output_subdir
        self.pub_dir = self.output_dir / "publication" / output_subdir
        
        self.sub_fig_dirs = {
            "missingness": self.fig_dir / "missingness" / output_subdir,
            "distributions": self.fig_dir / "distributions" / output_subdir,
            "outliers": self.fig_dir / "outliers" / output_subdir,
            "correlations": self.fig_dir / "correlations" / output_subdir,
            "dimensionality": self.fig_dir / "dimensionality" / output_subdir,
            "comparisons": self.fig_dir / "comparisons" / output_subdir,
            "batch_effects": self.fig_dir / "batch_effects" / output_subdir,
            "normality": self.fig_dir / "normality" / output_subdir,
            "relationships": self.fig_dir / "relationships" / output_subdir,
        }
        if self.is_paired_validation:
            self.sub_fig_dirs["paired_validation"] = self.fig_dir / "paired_validation"
        
    def create_directories(self):
        """Creates all necessary output directories."""
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.pub_dir.mkdir(parents=True, exist_ok=True)
        for d in self.sub_fig_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
            
# Instantiate a global config object
settings = Config()
settings.create_directories()
