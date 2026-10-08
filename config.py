"""
Global configuration for the real-time and offline fNIRS processing toolbox:
UI defaults, preprocessing parameter defaults, data folder paths, non-signal
column names, and per-experiment behavioral alignment profiles.
"""
import os

# Experiment profile pre-selected in the UI at startup (see EXPERIMENT_PROFILES below).
# Read-only default: the selected experiment is passed explicitly to the
# processing code, never written back here.
DEFAULT_EXPERIMENT = "new_experiment"


# Default values used to populate the preprocessing fields of the offline and
# neurofeedback tabs.
DEFAULT_PREPROC_CONFIG = {
    # Pipeline Step Toggles
    "butterworth_filter": True,
    "detrend": True,
    "averaging": True,
    "block_avg": True,

    # Quality Assessment & MBLL Conversion
    "enable_sci": False,
    "enable_sqa": True,
    "sqa_cv_threshold": 7.5,            # Max Coefficient of Variation per channel (%)
    "sci_correlation_min": 0.5,         # Min SCI (wavelength-pair correlation) to keep a channel
    "enable_mbll": True,                # Modified Beer-Lambert Law conversion

    # Bandpass Filter
    "filter_low_band": 0.05,            # Low cut-off frequency (Hz)
    "filter_high_band": 0.5,            # High cut-off frequency (Hz)
    "butterworth_order": 5,             # Butterworth filter order (applied zero-phase)

    # Detrending
    "detrend_batches": 10,              # Number of batches for piecewise linear detrending

    # Feature Scaling ('standard' or 'minmax')
    "scaling": True,
    "scaler_type": "standard",

    # Epoch Window, relative to each trial onset
    "peak_onset_time_buffer": 4,        # Window start after trial onset (seconds)
    "peak_offset_time_buffer": 9,       # Window end after trial onset (seconds)

    # Sampling
    "sampling_rate": 6.6,               # Native instrument sampling frequency (Hz)

    # Optical Settings
    "mbll_wavelengths": [735, 850],     # Emitter wavelengths used for MBLL (nm)
    # Extinction coefficients from:
    # Cope, M., et al. "Data analysis methods for near-infrared spectroscopy of tissue: problems in determining the relative cytochrome aa3 concentration." Time-Resolved Spectroscopy and Imaging of Tissues. Vol. 1431. SPIE, 1991.
    # Cope, M., van der Zee, P., Essenpreis, M., Delpy, D. T., Reynolds, E. O. R., & Wyatt, J. S. (1989). "Data analysis methods for near infrared spectroscopy of tissue." Proceedings of SPIE, Time-Resolved Spectroscopy and Imaging of Tissues, 1431, 251-263. DOI: 10.1117/12.44199
    "extinction_coefficients": {        # Per wavelength (nm, string key): [e_HbO, e_HbR]
        "690": [0.9554, 4.9318],
        "730": [1.1564, 3.8437],
        "735": [1.1977, 3.7323],  # Main for LUMO
        "760": [1.4866, 3.8437],
        "780": [1.7909, 2.9482],
        "805": [2.1416, 2.0534],  # Isosbestic point (approx equal absorption)
        "830": [2.4214, 1.7946],
        "850": [2.5264, 1.7983]   # Main for LUMO
    },

    # Plotting
    "max_plot_trial_traces": 5          # Max individual trial traces overlaid in plots
}

# Minimum decoder probability for a live trial to be added to the
# co-adaptation training set in the neurofeedback tab.
COADAPTATION_ACCEPTANCE_CRITERIA = 0.8


# Default Folder Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_RAW_RECORDINGS = os.path.join(BASE_DIR, "data", "raw", "recordings")
DEFAULT_RAW_BEHAVIORAL = os.path.join(BASE_DIR, "data", "raw", "behavs")
DEFAULT_OUT_SESSIONS   = os.path.join(BASE_DIR, "data", "preprocessed", "grouped_sessions")
DEFAULT_OUT_BEHAVIORAL = os.path.join(BASE_DIR, "data", "preprocessed", "grouped_behavs")


# Non-signal (structural / marker / label) columns.
# Single source of truth for which DataFrame columns are NOT signal channels,
# so every consumer (offline pipeline, decoder training, test harness)
# agrees on what counts as a feature. Experiment-specific behavioral
# columns are added on top of this via get_non_signal_columns(experiment).
NON_SIGNAL_COLUMNS = [
    "timestamps", "trial_onset", "trial_offset", "run_onset", "run_offset",
    "run", "target", "trial_index", "peak_onset", "peak_offset",
    "correct", "response", "index", "block_id"
]


# Experiment profiles.
# Describes how each paradigm's behavioral log is aligned onto the signal
# during the offline 'epoch_prep' step:
#   columns:          every column the behavioral log may contain
#   target_column:    behavioral column copied into 'target' (the decoder label);
#                     None -> 'target' defaults to 1 when absent
#   carry_columns:    behavioral columns copied onto each trial-onset row and
#                     forward-filled across the trial
#   keep_trials_where: {column: value} - only trials matching are kept
#                     (e.g. correct responses only); {} keeps every trial
#   behavior_required: True -> the pipeline refuses to run without a behavioral log
EXPERIMENT_PROFILES = {
    "WM_exp": {
        "columns": [
            'WM_load', 'correct', 'is_target', 'letter', 'live_row', 'response',
            'run_onset_time', 'time_logger', 'trial_index', 'time_trial_inline',
            'trial_onset_time', 'timestamp', 'run'
        ],
        "target_column": "WM_load",
        "carry_columns": ['trial_index', 'correct', 'response'],
        "keep_trials_where": {"correct": True},
        "behavior_required": True,
    },
    "Visual_perception_exp": {
        "columns": ['trial_index', 'condition', 'duration', 'timestamp', 'run'],
        "target_column": "condition",
        "carry_columns": ['trial_index', 'duration', 'timestamp'],
        "keep_trials_where": {},
        "behavior_required": False,
    },
    "new_experiment": {
        "columns": [],
        "target_column": None,
        "carry_columns": [],
        "keep_trials_where": {},
        "behavior_required": False,
    },
}


def get_experiment_profile(name):
    """Returns the named experiment profile, or the empty 'new_experiment' profile if unknown."""
    return EXPERIMENT_PROFILES.get(name, EXPERIMENT_PROFILES["new_experiment"])


def get_behavioral_columns(name):
    """Returns the behavioral log columns expected for the named experiment."""
    return list(get_experiment_profile(name)["columns"])


def get_non_signal_columns(experiment=None):
    """
    Columns to treat as non-signal. With an experiment name, also includes that
    profile's behavioral columns, which 'epoch_prep' merges into the session
    and must never be filtered, scaled, or averaged as if they were channels.
    """
    cols = list(NON_SIGNAL_COLUMNS)
    if experiment is not None:
        cols += [c for c in get_behavioral_columns(experiment) if c not in cols]
    return cols


def register_experiment_profile(name, columns, target_column=None, carry_columns=None,
                                keep_trials_where=None, behavior_required=False):
    """
    Adds (or replaces) a custom experiment profile for this session. When no
    target column is given, a 'condition' column is used as the label if the
    profile has one; carry_columns defaults to every other column except the
    target and 'run' (which the session merge already assigns).

    Returns the stored profile dict.
    """
    if target_column is None and "condition" in columns:
        target_column = "condition"
    if carry_columns is None:
        carry_columns = [c for c in columns if c not in (target_column, "run")]
    EXPERIMENT_PROFILES[name] = {
        "columns": list(columns),
        "target_column": target_column,
        "carry_columns": list(carry_columns),
        "keep_trials_where": dict(keep_trials_where or {}),
        "behavior_required": behavior_required,
    }
    return EXPERIMENT_PROFILES[name]
