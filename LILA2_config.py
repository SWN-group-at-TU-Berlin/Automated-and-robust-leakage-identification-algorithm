
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ellasteins

LILA Configuration Class

This module defines all parameters and initialized dataset-information used by the automated and robust leak identification algorithm.
"""

#%% imports

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import pandas as pd

#%% CONFIGURATION CLASS 
@dataclass
class LILAConfig:
    """
    Configuration class for automated and robust LILA
    
    All default parameters are defined.
    
    Specific to dataset: Start_date and end_date of the dataset, the pressure and flow files, 
    backgorund of the leakages (start and end times, sensor combinations)
    
    Usage:
        # Use defaults
        config = LILAConfig()
        
        # Customize parameters
        config = LILAConfig(cusum_threshold=60, training_days=10)
        
        # Access values
        print(config.cusum_threshold)  # 60
    """
    
    # ========== CUSUM Parameters ==========
    cusum_threshold: float = 52.0
    """CUSUM threshold for adaptive nonparametric method (h_adn)"""
    
    cusum_threshold_corr: float = 30.0
    """CUSUM threshold for correlation-based method (h_corr)"""
    
    ic_arl: int = 30
    """In-control Average Run Length for CUSUM (IC_ARL)
    Options: 30, 200, 300, 400, 500, 800, 1000"""
    
    delta_0: float = 0.7
    """Minimum magnitude of interest for early detection"""
    
    m_cusum: int = 2
    """Memory parameter for CUSUM statistic initialization"""
    
    # ========== Time Periods ==========
    training_days: int = 7
    """Initial training period duration in days"""
    
    detection_days: int = 28
    """Detection period duration in days"""
    
    verification_days: int = 7
    """Verification period duration in days"""
    
    adjustment_days_fp: int = 5
    """Days to subtract from leak start after false positive detection"""
    
    adjustment_days_fn: int = 10
    """Days to advance after false negative detection"""
    
    adjustment_hours_tp: int = 1
    """Hours to advance after true positive (successful) detection"""
    
    adjustment_hours_verification: int = 1
    """Hours to subtract in verification when alarm is raised"""
    
    plot_extension_hours: int = 3
    """Hours to extend plot beyond detection time for visualization"""
    
    # ========== Data Parameters ==========
    data_frequency: str = "15min"
    """Time resolution of SCADA data"""
    
    start_date: str = "2022-01-01 00:00:00"
    """Default start date for analysis"""
    
    end_date: str = "2022-11-08 23:45:00"
    """Default end date for analysis"""
    
    # ========== File Paths ==========
    data_dir: str = "data/"
    """Directory containing input data files"""
    
    pressure_file: str = "pressure.csv"
    """Filename for pressure measurements"""
    
    flow_file: str = "flow.csv"
    """Filename for flow measurements"""
    
    output_dir: str = "results/"
    """Directory for saving results and plots"""
    
    # ========== CSV Loading Parameters ==========
    csv_delimiter: str = ","
    """Delimiter for CSV files"""
    
    csv_decimal: str = "."
    """Decimal separator for CSV files"""
    
    # ========== Regression Parameters ==========
    regression_fit_intercept: bool = True
    """Whether to fit intercept in linear regression models"""
    
    # ========== Ground Truth ==========
    ground_truth: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        "L1": ("2022-01-19 12:45:00", "2022-02-01 12:45:00"),
        "L2": ("2022-02-19 20:15:00", "2022-02-21 20:00:00"),
        "L3": ("2022-03-15 13:30:00", "2022-04-24 13:15:00"),
        "L4": ("2022-05-10 15:30:00", "2022-06-15 15:30:00"),
        "L5": ("2022-07-03 22:30:00", "2022-07-23 22:15:00"),
        "L6": ("2022-08-12 17:15:00", "2022-08-14 17:00:00"),
        "L7": ("2022-09-01 08:00:00", "2022-09-04 08:00:00"),
        "L8": ("2022-09-21 19:30:00", "2022-09-25 19:30:00"),
        "L9": ("2022-10-14 11:30:00", "2022-11-08 23:45:00")
    })
    """Dictionary of true leak events: {leak_id: (start_time, end_time)}"""
    
    sensor_combinations: List[Tuple[str, str]] = field(default_factory=lambda: [
        ('p1', 'p2'),
        ('p2', 'p3'),
        ('p3', 'p2')
    ])
    """List of (MAS_node, closest_node) sensor combinations to test. The clostest node is the node closest to the respective
    MAS_node based on a shortest path algorithm. """
    
    # ========== Polynomial Coefficients for CUSUM ==========
    polynomial_coeffs: Dict[int, List[float]] = field(default_factory=lambda: {
        30: [13.54844243, -278.74909693, 953.99916442, -1436.40494624,
             1161.01248598, -535.16425988, 141.95524256, -24.57009576, 5.02083018],
        # Add coefficients for other IC_ARL values as needed:
        # 200: [...],
        # 300: [...],
    })
    """Polynomial coefficients for h(k) function, keyed by IC_ARL value"""
    
    # ========== Plotting Parameters ==========
    plot_figsize: Tuple[int, int] = (18, 7)
    """Figure size for CUSUM plots (width, height)"""
    
    plot_dpi: int = 300
    """DPI for saved plot images"""
    
    plot_title_fontsize: int = 16
    """Font size for plot titles"""
    
    plot_label_fontsize: int = 16
    """Font size for axis labels"""
    
    plot_tick_fontsize: int = 14
    """Font size for tick labels"""
    
    plot_legend_fontsize: str = "x-large"
    """Font size for plot legends"""
    
    plot_text_fontsize: int = 15
    """Font size for annotations"""
    
    # ========== Helper Methods ==========
    
    def get_training_timedelta(self) -> pd.Timedelta:
        """Get training period as pandas Timedelta"""
        return pd.Timedelta(days=self.training_days)
    
    def get_detection_timedelta(self) -> pd.Timedelta:
        """Get detection period as pandas Timedelta"""
        return pd.Timedelta(days=self.detection_days)
    
    def get_verification_timedelta(self) -> pd.Timedelta:
        """Get verification period as pandas Timedelta"""
        return pd.Timedelta(days=self.verification_days)
    
    def get_pressure_path(self) -> str:
        """Get full path to pressure data file"""
        import os
        return os.path.join(self.data_dir, self.pressure_file)
    
    def get_flow_path(self) -> str:
        """Get full path to flow data file"""
        import os
        return os.path.join(self.data_dir, self.flow_file)
    
    def get_timestamps(self):
        """Generate timestamp range based on start/end dates"""
        return pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=self.data_frequency
        )
    
    def validate(self):
        """Validate configuration parameters"""
        errors = []
        
        if self.cusum_threshold <= 0:
            errors.append("cusum_threshold must be positive")
        
        if self.training_days <= 0:
            errors.append("training_days must be positive")
        
        if self.detection_days <= 0:
            errors.append("detection_days must be positive")
        
        if self.ic_arl not in self.polynomial_coeffs:
            errors.append(
                f"ic_arl {self.ic_arl} not supported. "
                f"Available: {list(self.polynomial_coeffs.keys())}"
            )
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary (useful for saving)"""
        return {
            'cusum_threshold': self.cusum_threshold,
            'cusum_threshold_corr': self.cusum_threshold_corr,
            'ic_arl': self.ic_arl,
            'delta_0': self.delta_0,
            'training_days': self.training_days,
            'detection_days': self.detection_days,
            'data_frequency': self.data_frequency,
            'start_date': self.start_date,
            'end_date': self.end_date,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create configuration from dictionary"""
        return cls(**config_dict)

#%% Example Usage


if __name__ == "__main__":
    # Example 1: Use default configuration
    print("=" * 60)
    print("Example 1: Default Configuration")
    print("=" * 60)
    config = LILAConfig()
    print(f"CUSUM Threshold: {config.cusum_threshold}")
    print(f"Training Days: {config.training_days}")
    print(f"Detection Days: {config.detection_days}")
    print(f"Data Frequency: {config.data_frequency}")
    
    # Example 2: Custom configuration
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)
    custom_config = LILAConfig(
        cusum_threshold=60.0,
        training_days=10,
        detection_days=21,
        data_dir="./my_data/"
    )
    print(f"CUSUM Threshold: {custom_config.cusum_threshold}")
    print(f"Training Days: {custom_config.training_days}")
    print(f"Pressure Path: {custom_config.get_pressure_path()}")
    
    # Example 3: Validate configuration
    print("\n" + "=" * 60)
    print("Example 3: Validation")
    print("=" * 60)
    try:
        config.validate()
        print("✓ Configuration is valid")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
    
    # Example 4: Using helper methods
    print("\n" + "=" * 60)
    print("Example 4: Helper Methods")
    print("=" * 60)
    print(f"Training Timedelta: {config.get_training_timedelta()}")
    print(f"Detection Timedelta: {config.get_detection_timedelta()}")
    print(f"Number of timestamps: {len(config.get_timestamps())}")
    
    # Example 5: Save/load configuration
    print("\n" + "=" * 60)
    print("Example 5: Save/Load Configuration")
    print("=" * 60)
    config_dict = config.to_dict()
    print(f"Config as dict: {config_dict}")
    
    restored_config = LILAConfig.from_dict(config_dict)
    print(f"Restored CUSUM threshold: {restored_config.cusum_threshold}")
#%%