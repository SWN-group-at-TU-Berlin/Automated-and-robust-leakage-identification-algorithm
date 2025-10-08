
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ellasteins

LILA main script

This is the main script for running the automated and robust leak identification algorithm.
"""
#%% imports
import sys
sys.path.append('..')
from LILA2_config import *
from LILA2_utils import *

import pandas as pd
#%% FUNCTION FOR LOADING THE PRESSURE AND FLOW DATA

def load_data(config):
    """
    Load pressure and flow data using paths from config.
    
    Args:
        config: LILAConfig object containing file paths and parameters
        
    Returns:
        df_scada_pressure: DataFrame with pressure measurements
        df_scada_flows: DataFrame with flow measurements
    """
    timestamps = config.get_timestamps()
    
    # Load Pressure
    path_scada_pressure = config.get_pressure_path()
    df_scada_pressure = pd.read_csv(
        path_scada_pressure,
        delimiter=config.csv_delimiter,
        decimal=config.csv_decimal
    )
    
    if len(df_scada_pressure) == len(timestamps):
        df_scada_pressure.index = timestamps
        df_scada_pressure.index.name = "Timestamp"
    else:
        print(f"Warning: Pressure CSV has {len(df_scada_pressure)} rows, expected {len(timestamps)}")
    
    # Load Flow
    path_scada_flows = config.get_flow_path()
    df_scada_flows = pd.read_csv(path_scada_flows)
    
    if len(df_scada_flows) == len(timestamps):
        df_scada_flows.index = timestamps
        df_scada_flows.index.name = "Timestamp"
    else:
        print(f"Warning: Flow CSV has {len(df_scada_flows)} rows, expected {len(timestamps)}")
    
    return df_scada_pressure, df_scada_flows

#%% FUNCTION TO RUN LILA: flowchart steps 3-6
def run_lila_detection(df_scada_pressure, df_scada_flows, config):
    """
    Run LILA detection algorithm.
    
    Args:
        df_scada_pressure: DataFrame with pressure data
        df_scada_flows: DataFrame with flow data
        config: LILAConfig object with all parameters
        
    Returns:
        detection: Dictionary with detection results
    """
    detection = {}
    start_date_set = config.start_date
    old_start_date = config.start_date
    training_length_set = str(config.get_training_timedelta())
    
    # Main detection loop: Repeat until start time is after last leakage
    i = 0
    while pd.Timestamp(config.ground_truth['L9'][0]) - pd.Timestamp(start_date_set) > pd.Timedelta(0):
        i = i+1
        print("ITERATION "+str(i))
        
        # FLOWCHART STEP 3 & 4: Setting of training time, and closest leak start, Verification step
        training_length_set, true_leak_start_set = f_verification_MRE(
            df_scada_pressure,
            df_scada_flows,
            old_start_date,
            start_date_set,
            training_length_set,
            config
            # will use three config-paramaters:
            # config.ground_truth, config.sensor_combinations, config.cusum_threshold 
        )
        
        # FLOWCHART STEP 5 & 6, 7-8: Run LILA detection (training of regression models and change point detection with CUSUM, set new start time)
        start_date_set_new, det_leak, MAS_signaled, df_C, pipe_id, TTD, TTD_times = f_LILA2(
            df_scada_pressure,
            df_scada_flows,
            config,
            true_leak_start = true_leak_start_set,
            training_length=training_length_set,
            start_date=start_date_set
            # will use three config-parameters:
            # config.sensor_combinations, config.ground_truth, config.cusum_threshold
        )
            
        print("- Leakage to be detected: "+str(pipe_id))
        print("- Start time: "+str(start_date_set)+" with training duration: "+str(training_length_set))
        if det_leak=="FP":
            print("- Detection result: FP")
        elif det_leak=="FN":
            print("- Detection result: FN")
        else:
            print("- Detection result: TP with TTD = "+str(det_leak))
            
            
        # Plot CUSUM detection results
        f = f_plot_S_adn(
            df_C,
            MAS_signaled,
            det_leak,
            pipe_id,
            config
            # will use two config-parameters:
            # config.ground_truth, config.cusum_threshold
        )
        
        # Save detection result
        detection[pipe_id, start_date_set] = (det_leak, MAS_signaled, training_length_set)
        old_start_date = start_date_set
        start_date_set = start_date_set_new
    
    return detection

#%% MAIN 

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("INITIALIZATION:")
    print("="*80)
    # ========== METHOD 1: Use Default Configuration ==========
    print("Running with default configuration.")
    # FLOWCHART STEP 1 & 2: setting of initial start time, leakage information, threshold paramater
    config = LILAConfig()
    
    # ========== METHOD 2: Customize Specific Parameters ==========
    # Uncomment to use custom parameters:
    # config = LILAConfig(
    #     cusum_threshold=60.0,
    #     training_days=10,
    #     detection_days=21,
    #     data_dir="../data/"
    # )
    
    # ========== METHOD 3: Load from External Source ==========
    # You could also load from a YAML or JSON file:
    # import yaml
    # with open('config.yaml', 'r') as f:
    #     config_dict = yaml.safe_load(f)
    # config = LILAConfig(**config_dict)
    
    # Validate configuration
    try:
        config.validate()
        print("✓ Configuration validated successfully")
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    print("\n")
    print(f"CUSUM Threshold: {config.cusum_threshold}")
    print(f"Number of Leakages: {len(config.ground_truth)}")
    print(f"Number of Sensor Combinations: {len(config.sensor_combinations)}")
    print(f"Initial start date: {config.start_date}")
    print("\n")
    print(f"Training Period: {config.training_days} days")
    print(f"Detection Period: {config.detection_days} days")
    print(f"IC ARL: {config.ic_arl}")
    print(f"Data Directory: {config.data_dir}")
    # print("="*80 + "\n")
    
    # Load data using config
    df_scada_pressure, df_scada_flows = load_data(config)
    
    # Run detection
    print("\n" + "="*80)
    print("ITERATIVE RETARINING TIMES AND DETECTION:")
    print("="*80)
    detection = run_lila_detection(df_scada_pressure, df_scada_flows, config)
    
    # Display results
    print("\n" + "="*80)
    print("SUMMARY OF DETECTION RESULTS:")
    print("="*80)
    for key, value in detection.items():
        print(f"{key}: {value}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
    
    #%%