
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ellasteins

LILA utility script

This script contains all functions used by the automated and robust leak identification algorithm, including:
- ITERATVE RETRAINING TIMES:
    - FLOWCHART STEPS 3 & 4: setting retraining time with verification step (f_verification_MRE)
- ITERATIVE DETECTION:
    – FLOWCHART STEPS 5 - 9: Wrapper function for iterative part (f_LILA)
        - STEP 5 – Linear regression: Regression and MRE computation (leak_analysis_individual),
                                      Class for storing regression models (class State)
        - STEP 6 – CUSUM method: wrapper function (f_CUSUM),
                                 adaptive and nonparametric CUSUM method (f_CUSUM_adn), 
                                 helper functions (interpret_ttd, smallest_positive_timedelta, extract_max_fp_timedelta),
                                 plotting the CUSUM statistic (f_plot_S_adn)
"""

#%% imports
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from datetime import timedelta


#%% VERIFICATION MODULE 

def f_verification_MRE(df_scada_pressure, df_scada_flows, old_start_date, 
                       new_start_date, training_length_old, config):
    
    """
    Verification Module: MRE for the new training duration is created with regression models of previous detection round,
    and tested if a alarm is raised with CUSUM. Based of current start time (new_start_date), the closest leak start is chosen
    
    Args:
        df_scada_pressure: Pressure data
        df_scada_flows: Flow data
        old_start_date: Previous start date
        new_start_date: Current start date
        training_length_old: Duration of the previous training period
        config: LILAConfig object
        
    Returns:
        training_length: str, Duration of training period
        true_leak_start: str, Closest leak start time to current start date (new start date)
    """
    
    # used parameters of Configuration objects
    ground_truth_sensors = config.sensor_combinations
    ground_truth = config.ground_truth
    h_adn = config.cusum_threshold
    
    # identify closest leak start ( the leakage that happens first after the current start time of the decetion)
    list_of_true_leaks = [v[0] for v in ground_truth.values()]
    time_diff = []
    for leak_start in list_of_true_leaks:
        time_diff.append(pd.Timestamp(leak_start) - pd.Timestamp(new_start_date))
    smallest_timedelta = smallest_positive_timedelta(time_diff)
    true_leak_start = pd.Timestamp(new_start_date) + smallest_timedelta
        
    # in the first round, no previous regression model exists, hence default training duration = '7 days'
    if old_start_date == new_start_date:
        training_length = str(config.get_training_timedelta())
        
    # starting from the second iteration, the previous regression models are used for testing the new
    # potential 7-days training time
    else:
        time_index = pd.date_range(
            start=new_start_date,
            end=str(pd.Timestamp(new_start_date) + config.get_verification_timedelta()),
            freq=config.data_frequency
        )
        
        df_C = pd.DataFrame(index=time_index)
        TTD = []
        TTD_times = []
        
        # Time frame list for linear regression 
        time_frame_list = [[
            old_start_date,  # start date of old detection period
            old_start_date,  # start date of old training period
            str(pd.Timestamp(old_start_date) + pd.Timedelta(training_length_old)), # end date of old training period
            str(pd.Timestamp(new_start_date) + pd.Timedelta(days=8)) # end date of new training period + 1 day
        ]]
        
        A = len(ground_truth_sensors)
        
        # Loop through sensor combinations
        for j in range(A):
            node_MAS = ground_truth_sensors[j][0]
            closest_nodes = [ground_truth_sensors[j][1]]
            
            # getting the MRE for the full time frame
            MRE = leak_analysis_individual(
                df_scada_pressure, df_scada_flows, node_MAS, 
                closest_nodes, time_frame_list, config
            ) 
            
            # getting the MRE of the current node_MAS for the new potential 7-day detection period
            MRE_new = MRE.loc[
                new_start_date:str(pd.Timestamp(new_start_date) + 
                                   config.get_verification_timedelta())
            ]
            
            # perform CUSUM method on MRE 
            detection, C, TTD_time = f_CUSUM(MRE_new, true_leak_start, config)
            
            # append results
            df_C[node_MAS] = C
            TTD.append(detection)
            TTD_times.append(TTD_time)
        
        # get overall detection results based of above CUSUM-detection-results
        result_identification, index = interpret_ttd(TTD)
        
        # if no alarm, use full 7-day training period,otherwise only up to 1 hour before raised alarm
        if result_identification == 'FN':
            training_length = str(config.get_training_timedelta())
        else:
            training_length = str(
                (pd.Timestamp(true_leak_start) + pd.Timedelta(TTD_times[index])) - 
                pd.Timedelta(hours=config.adjustment_hours_verification) - 
                pd.Timestamp(new_start_date)
            )
    
    return (training_length, true_leak_start)

#%% WRAPPER FUNCTION FOR ITERATIVE PART

def f_LILA2(pressure, flow, config, true_leak_start, training_length, start_date=None):
    """
    LILA detection using configuration object.
    
    Args:
        pressure: pd.DataFrame with pressure measurements
        flow: pd.DataFrame with flow measurements
        config: LILAConfig object containing all parameters
        training_length: str or pd.Timedelta for training duration -> given by verification module
        start_date: str or None, start date for analysis
        
    Returns:
        Same as before...
    """
    # Access parameters from config:
    ground_truth = config.ground_truth # leakage start and end times
    ground_truth_sensors = config.sensor_combinations # all possible sensor combinations
    h_adn = config.cusum_threshold # CUSUM threshold parameter
    
    # get the Leakage ID (pipe_id) of the leakage that is searched for
    true_leak_start = pd.Timestamp(true_leak_start)  
    pipe_id = [k for k, v in ground_truth.items() if v[0] == str(true_leak_start)][0]
    
    # Time frame list for linear regression 
    time_frame_list = [
        [
            start_date, # start date of detection period
            start_date, # start date of training period
            str(pd.Timestamp(start_date) + pd.Timedelta(training_length)), # end date of training period
            str(pd.Timestamp(start_date) + config.get_detection_timedelta())  # end date of detection period
        ]
    ]
    
    # Initialize
    A = len(ground_truth_sensors) 
    TTD = []
    TTD_times = []
    time_index = pd.date_range( start=time_frame_list[0][0], end=time_frame_list[0][-1],freq=config.data_frequency)
    df_C = pd.DataFrame(index=time_index)
    df_MRE = pd.DataFrame(index=time_index)
    
    # Loop through sensor combinations
    for j in range(A):
        # get current MAS and closest sensor
        node_MAS = ground_truth_sensors[j][0]
        closest_nodes = [ground_truth_sensors[j][1]]
        
        # perform linear regression
        MRE = leak_analysis_individual(pressure, flow, node_MAS, closest_nodes, time_frame_list, config)
        
        # perform change-point-detection with CUSUM
        detection, C, TTD_time = f_CUSUM(MRE, true_leak_start, config)
        
        # append results
        TTD.append(detection)
        TTD_times.append(TTD_time)
        df_C[node_MAS] = C
        df_MRE[node_MAS] = MRE[MRE.columns[0]].to_numpy()
        
    # get overall detection results based of above CUSUM-detection-results
    result_identification, index = interpret_ttd(TTD)
    
    MAS_signaled = ground_truth_sensors[index][0] if index is not False else None
    
    # Set new start date according to FLOWCHART STEPS 7 - 9:
    if result_identification == 'FP': # Step 8
        TTD_FP = extract_max_fp_timedelta(TTD, TTD_times)
        start_date_set_new = str(
            pd.Timestamp(ground_truth[pipe_id][0]) - TTD_FP + 
            pd.Timedelta(days=config.adjustment_days_fp)  
        )
    elif result_identification == 'FN': # Step 9
        start_date_set_new = str(
            pd.Timestamp(start_date) + 
            pd.Timedelta(days=config.adjustment_days_fn) 
        )
    else: # Step 7
        start_date_set_new = str(
            pd.Timestamp(ground_truth[pipe_id][1]) + 
            pd.Timedelta(hours=config.adjustment_hours_tp)  
        )
    
    return start_date_set_new, result_identification, MAS_signaled, df_C, pipe_id, TTD, TTD_times


#%% STATE CLASS FOR REGRESSION

class State:
    """
    State object for storing time windows and regression models.
    
    Attributes:
        _start: str
            Start time of the full analysis window
        _end: str
            End time of the full analysis window
        _cor_start: str
            Start time of the correlation/training window
        _cor_end: str
            End time of the correlation/training window
        _models_Reg: list or None
            List of trained regression models
    """
    
    def __init__(self, start, end, cor_start, cor_end, models_Reg=None):
        self._models_Reg = models_Reg
        self._start = start
        self._end = end
        self._cor_start = cor_start
        self._cor_end = cor_end
    
    def set_models(self, models):
        """Set the regression models for this state"""
        self._models_Reg = models
    
    def get_models(self):
        """Get the regression models for this state"""
        return self._models_Reg
    
    def __repr__(self):
        """String representation for debugging"""
        return (f"State(start={self._start}, end={self._end}, "
                f"cor_start={self._cor_start}, cor_end={self._cor_end}, "
                f"models={'loaded' if self._models_Reg else 'None'})")

#%% REGRESSION FUNCTION 

def leak_analysis_individual(df_scada_pressure, df_scada_flows, node_MAS, 
                            closest_nodes, time_frame_list, config):
    """
    Perform individual leak analysis by training linear regression models on training periods and 
    computing residuals (Mean Residual Error, MRE) for the specified closest nodes.

    Linear regression models are trained for each pressure node using the MAS node pressure and pump 
    flow as input features over a defined healthy period. Residuals are then computed for the analysis period.
    leak analysis using config parameters.
    
    Args:
        df_scada_pressure: Pressure measurements
        df_scada_flows: Flow measurements
        node_MAS: Main sensor node
        closest_nodes: List of closest nodes
        time_frame_list: Time frames for analysis
        config: LILAConfig object
        
    Returns:
        pd.DataFrame: Residual time series (MRE)
    """
    
    def old_style_linear_regression(X, y, fit_intercept=True):
        """Custom linear regression implementation"""
        if fit_intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        intercept = coef[0] if fit_intercept else 0.0
        coefficients = coef[1:] if fit_intercept else coef
        return intercept, coefficients
    
    # Create State objects
    states = []
    for time_frame in time_frame_list:
        states.append(State(
            start=time_frame[0],
            end=time_frame[3],
            cor_start=time_frame[1],
            cor_end=time_frame[2]
        ))
    
    column_names = df_scada_pressure.columns[1:].to_numpy()
    
    # Train models
    for state in states:
        models = []
        cor_time_frame = [state._cor_start, state._cor_end]
        
        # Build input features
        X_tr = np.concatenate([
            df_scada_pressure[node_MAS].loc[cor_time_frame[0]:cor_time_frame[1]].to_numpy().reshape(-1, 1),
            df_scada_flows['pump'].loc[cor_time_frame[0]:cor_time_frame[1]].to_numpy().reshape(-1, 1)
        ], axis=1)
        
        # Fit models using config parameter
        for node_cor in column_names:
            y_tr = df_scada_pressure[node_cor].loc[cor_time_frame[0]:cor_time_frame[1]].to_numpy().reshape(-1, 1)
            intercept, coef = old_style_linear_regression(
                X_tr, y_tr, 
                fit_intercept=config.regression_fit_intercept  # From config
            )
            model_dict = {"intercept": intercept, "coef": coef}
            models.append(model_dict)
        
        state.set_models(models)
    
    # Prepare DataFrame for residuals using config frequency
    timestamps = pd.date_range(
        start=time_frame_list[0][0],
        end=time_frame_list[0][-1],
        freq=config.data_frequency  # Instead of "15min"
    )
    
    df_state = pd.DataFrame(
        np.zeros((
            df_scada_pressure.loc[time_frame_list[0][0]:time_frame_list[0][-1]].shape[0],
            int(df_scada_pressure.shape[1] - 1)
        )),
        columns=column_names,
        index=timestamps
    )
    
    # Evaluate prediction errors
    for i_ in range(len(states)):
        models = states[i_]._models_Reg
        start = states[i_]._start
        end = states[i_]._end
        
        for node_cor in closest_nodes:
            df_error = pd.DataFrame(index=timestamps)
            
            y_test = df_scada_pressure[node_cor].loc[time_frame_list[0][0]:time_frame_list[0][-1]].to_numpy().reshape(-1, 1)
            
            X_test = np.concatenate([
                df_scada_pressure[node_MAS].loc[time_frame_list[0][0]:time_frame_list[0][-1]].to_numpy().reshape(-1, 1),
                df_scada_flows['pump'].loc[time_frame_list[0][0]:time_frame_list[0][-1]].to_numpy().reshape(-1, 1)
            ], axis=1)
            
            model = models[df_scada_pressure.loc[time_frame_list[0][0]:time_frame_list[0][-1]].columns[1:].tolist().index(node_cor)]
            y_pred = X_test @ model['coef'] + model['intercept']
            df_error['e'] = y_test - y_pred
            df_state.loc[start:end, node_cor] = df_error['e'].loc[start:end]
    
    return df_state[closest_nodes]

#%% CUSUSM WRAPPER FUNCTION 

def f_CUSUM(df, true_leak_start, config):
    """
    Run CUSUM leak detection on a residual time series and return detection results.
    If other/additional CUSUM methods than f_CUSUM_adn should be employed, 
    they can be combined with this function.
    
    Args:
        df: pd.DataFrame
            MRE time series for one or multiple sensors
        true_leak_start: str or pd.Timestamp
            Timestamp of the actual leak start used to calculate time-to-detection
        config: LILAConfig
            Configuration object containing CUSUM parameters
    
    Returns:
        detection: str or pd.Timedelta
            Overall detection result. Returns the time-to-detection as a timedelta if detected, 
            'FP' if false positive, or 'FN' if no detection.
        C: np.ndarray
            The computed CUSUM statistic over time for each sensor
        TTD_leak: pd.Timedelta or str
            Time-to-detection relative to the true leak start, or 'none' if no detection
    """
    
    # Call f_CUSUM_adn
    leak_det, det, C = f_CUSUM_adn(df, config)
    
    if det == 0:
        detection = "FN"
        TTD_leak = "none"
    else:
        TTD_leak = (leak_det - pd.Timestamp(true_leak_start))
        TTD_leak_seconds = TTD_leak.total_seconds()
        if TTD_leak_seconds >= 0:
            detection = TTD_leak
        else:
            detection = "FP"
    
    return (detection, C, TTD_leak)

#%% ADAPATIVE AND NONPARAMTERIC CUSUM FUNCTION 

def f_CUSUM_adn(df, config):
    """
    Adaptive and non-parametric CUSUM per Liu, Tsang, and Zhang. 
    Adaptive nonparametric CUSUM scheme for detecting unknown shifts in location. 2014.
    
    Args:
        df: pd.DataFrame
            Data to analyze (MRE time series)
        config: LILAConfig
            Configuration object containing:
            - cusum_threshold: threshold value (h_thr)
            - ic_arl: IC ARL value
            - delta_0: minimum magnitude of interest
            - m_cusum: memory parameter
            - polynomial_coeffs: coefficients for h(k) function
    
    Returns:
        leak_det: pd.Timestamp or pd.Series
            Timestamp of leak detection if detected
        det: int
            1 if leak detected, 0 otherwise
        S: np.ndarray
            CUSUM statistic values over time
    
    Notes:
        IC_ARL: IC ARL, choose IC_ARL = {30, 200, 300, 400, 500, 800, 1000}
        Y_t: one-step ahead estimate of shift in R_t
        delta_0: minimum magnitude of interest for early detection
        Rst_t: standardised sequential rank
        R_t: sequential rank 
        h(k): operating function, which denotes the control limit with reference value k
    """
    
    # Get parameters from config
    h_thr_alter = config.cusum_threshold
    IC_ARL = config.ic_arl
    delta_0 = config.delta_0
    m = config.m_cusum
    
    def f_hk(IC_ARL, k):
        """Calculate h(k) using polynomial coefficients from config"""
        if IC_ARL not in config.polynomial_coeffs:
            raise ValueError(
                f"IC_ARL {IC_ARL} not supported. "
                f"Available values: {list(config.polynomial_coeffs.keys())}"
            )
        
        a = np.array(config.polynomial_coeffs[IC_ARL])
        h_k = 0
        for i in range(len(a)):
            h_k = h_k + a[i] * k**(len(a)-1-i)
        
        return h_k
    
    def f_Yt(Rst_t, Rst_tminus1):
        """Calculate one-step ahead forecast"""
        Y_t = (Rst_t + Rst_tminus1) / 2
        return Y_t
    
    def f_Rst(x):
        """Calculate standardized sequential rank"""
        t = x.size
        x_t = x[-1]
        R_t = np.sum(x_t >= x)
        E = (t + 1) / 2
        V = ((t + 1) * (t - 1)) / 12
        Rst_t = (R_t - E) / np.sqrt(V)
        return Rst_t
    
    # Process each column in the dataframe
    for i, col in enumerate(df.columns):
        traj_ = df[col].copy()
    
    X = traj_.to_numpy()
    
    # Initialize CUSUM statistic
    S = np.zeros(df.shape)
    S[0, :] = 0
    Rst = np.zeros(df.shape[0])
    Yt = 0
    
    # Compute CUSUM statistic
    for t in range(m + 1, df.shape[0]):
        Rst[t] = f_Rst(X[0:t+1])
        Yt = f_Yt(Rst[t], Rst[t-1])
        delta_t = max(Yt, delta_0)
        k = delta_t / 2
        hk = f_hk(IC_ARL, k)
        z = S[t-1, :] + ((Rst[t] - k) / hk)
        S[t, :] = max(0, z)
    
    # Create DataFrame with CUSUM statistics
    df_S = pd.DataFrame(S, columns=df.columns, index=df.index)
    leak_det = pd.Series(dtype=object)
    
    # Check if threshold is exceeded
    det = 0
    for i, pipe in enumerate(df_S):
        hthr = h_thr_alter
        if any(df_S[pipe] > hthr):
            leak_det[pipe] = df_S.index[(df_S[pipe] > hthr).values][0]
            det = 1
            leak_det = leak_det.iloc[0]  # Get first detection
            break  # Exit loop once detection found
    
    return (leak_det, det, S)


#%% CUSUM - HELPER FUNCTIONS 

def interpret_ttd(ttd):
    """
    Interpret the time-to-detection (TTD) results from multiple sensors to identify the overall detection outcome.
    
    Args:
        ttd: list
            List of detection results for each sensor combination. 
            Each entry can be a timedelta (time to detection), 'FP' (false positive), or 'FN' (false negative).
    
    Returns:
        result_identification: str
            Overall detection result: 'FP' if any false positives exist, 'FN' if only false negatives, 
            otherwise the shortest positive TTD as a string.
        shortest_idx: int or bool
            Index of the sensor combination corresponding to the shortest positive TTD.
            Returns False if result_identification is 'FP' or 'FN'.
    """
    if 'FP' in ttd:
        shortest_idx = False
        return 'FP', shortest_idx  # If FP in list, the result is FP
    elif set(ttd) == {'FN'}:
        shortest_idx = False
        return 'FN', shortest_idx  # If only FNs in list, the result is FN
    else:
        # Filter out 'FN' and parse the remaining durations as timedeltas
        durations = [
            (i, pd.to_timedelta(val)) 
            for i, val in enumerate(ttd) 
            if val != 'FN'
        ]
        
        if not durations:
            # Edge case: all are FN but set check didn't catch it
            shortest_idx = False
            return 'FN', shortest_idx
        
        # TTD is the shortest TTD, index is important to know which sensor it was
        shortest_idx, shortest_val = min(durations, key=lambda x: x[1])
        return str(shortest_val), shortest_idx


def smallest_positive_timedelta(timedeltas):
    """
    Finds the smallest positive timedelta in a list of timedeltas.
    
    Args:
        timedeltas: list
            A list of datetime.timedelta or pd.Timedelta objects.
    
    Returns:
        datetime.timedelta or pd.Timedelta or None
            The smallest positive timedelta, or None if no positive timedeltas exist.
    
    Example:
        >>> from datetime import timedelta
        >>> deltas = [timedelta(days=-5), timedelta(days=3), timedelta(days=1)]
        >>> smallest_positive_timedelta(deltas)
        datetime.timedelta(days=1)
    """
    positive_timedeltas = [delta for delta in timedeltas if delta > timedelta(0)]
    
    if not positive_timedeltas:
        return None
    
    # Use built-in min for cleaner code
    smallest_delta = min(positive_timedeltas)
    
    return smallest_delta


def extract_max_fp_timedelta(TTD, TTD_times):
    """
    Extract the maximum false positive (FP) detection timedelta from a list of detection results.
    
    This function identifies all occurrences labeled as 'FP' in TTD and returns the corresponding 
    timedelta with the largest absolute value from TTD_times. Useful for adjusting start dates or 
    evaluating the worst-case FP scenario.
    
    Args:
        TTD: list
            List of detection results for multiple sensors, containing 'FP', 'FN', or timedeltas.
        TTD_times: list
            List of timedeltas corresponding to the TTD entries. Should have same length as TTD.
    
    Returns:
        pd.Timedelta or None
            The maximum absolute timedelta among all false positive detections, 
            or None if no 'FP' exists.
    
    Raises:
        ValueError: If TTD and TTD_times have different lengths.
    
    Example:
        >>> import pandas as pd
        >>> TTD = ['FP', 'FN', 'FP', pd.Timedelta(hours=2)]
        >>> TTD_times = [pd.Timedelta(hours=-3), 'none', pd.Timedelta(hours=-5), pd.Timedelta(hours=2)]
        >>> extract_max_fp_timedelta(TTD, TTD_times)
        Timedelta('-1 days +19:00:00')  # -5 hours has larger absolute value than -3 hours
    """
    if len(TTD) != len(TTD_times):
        raise ValueError(
            f"TTD and TTD_times must have same length. "
            f"Got {len(TTD)} and {len(TTD_times)}"
        )
    
    # Collect all Timedeltas corresponding to 'FP'
    fp_times = [td for label, td in zip(TTD, TTD_times) if label == 'FP']
    
    if not fp_times:
        return None  # No false positives found
    
    # Return the one with the maximum absolute value
    return max(fp_times, key=lambda x: abs(x))


#%% CUSUM PLOTTING FUNCTION 

def f_plot_S_adn(df_C, MAS_signaled, det_leak, pipe_id, config):
    """
    Plot CUSUM statistics using config parameters.
    
    Args:
        df_C: DataFrame with CUSUM statistics
        MAS_signaled: MAS node that detected leak
        det_leak: Detection result
        pipe_id: Pipe identifier
        config: LILAConfig object
        
    Returns:
        matplotlib.figure.Figure
    """
    ground_truth = config.ground_truth
    true_leak_start = ground_truth[pipe_id][0]
    true_leak_fix = ground_truth[pipe_id][1]
    
    h = config.cusum_threshold  # Instead of h_adn parameter
    
    # Determine plot end time
    if det_leak == "FP" or det_leak == "FN":
        if det_leak == "FP":
            end_time = pd.Timestamp(true_leak_start) + pd.Timedelta(
                hours=config.plot_extension_hours  
            )
        else:
            end_time = pd.Timestamp(true_leak_fix)
        A = 0
        filtered_df_C = df_C[df_C.index <= end_time]
    else:
        end_time = pd.Timestamp(true_leak_start) + pd.Timedelta(det_leak) + pd.Timedelta(
            hours=config.plot_extension_hours 
        )
        A = 1
        filtered_df_C = df_C[MAS_signaled][df_C.index <= end_time]
        filtered_df_C_all = df_C[df_C.index <= end_time]
    
    # Create plot 
    plt.rcParams['text.usetex'] = False
    f, ax = plt.subplots(
        1, 
        sharex=True, 
        sharey=True, 
        figsize=config.plot_figsize 
    )
    
    plt.title(
        f"Leak ID: {pipe_id}", 
        fontsize=config.plot_title_fontsize 
    )
    
    ax.axvline(pd.Timestamp(true_leak_start), linestyle='--', color='gray', lw=2)
    plt.text(
        pd.Timestamp(true_leak_start) - pd.Timedelta(hours=7),
        0.5,
        "true leak start",
        rotation=90,
        transform=ax.get_xaxis_transform(),
        verticalalignment='bottom',
        fontsize=config.plot_text_fontsize, 
        color='black'
    )
    
    if A == 1: #TP
        ax.axvline(
            pd.Timestamp(true_leak_start) + pd.Timedelta(det_leak),
            linestyle='--',
            color='C1',
            lw=2
        )
        plt.plot(filtered_df_C, lw=2, linestyle="-", color="C2", label=f"MAS = {MAS_signaled}")
        plt.plot(filtered_df_C_all, linestyle="--", color="C2", alpha=0.2)
        plt.legend(fontsize=config.plot_legend_fontsize)  # Instead of "x-large"
    else: #FN or FP
        plt.plot(filtered_df_C, linestyle="--", color="C2")
    
    ax.axhline(h, linestyle="-", color='red', lw=0.5, alpha=0.8)
    plt.text(0.05, h+0.5, "threshold", transform=ax.get_yaxis_transform(), 
             fontsize=config.plot_text_fontsize)
    
    plt.xticks(fontsize=config.plot_tick_fontsize) 
    plt.yticks(fontsize=config.plot_tick_fontsize)  
    plt.xlabel("time [date]", fontsize=config.plot_label_fontsize)  
    plt.ylabel("$C^+$ statistic [-]", fontsize=config.plot_label_fontsize)  
    plt.show()
    
    return f
#%%
