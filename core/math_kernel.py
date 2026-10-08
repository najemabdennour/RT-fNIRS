# core/math_kernel.py
"""Low-level numeric routines: bandpass filtering, channel quality metrics, and MBLL inversion."""
import numpy as np
from scipy.signal import butter, sosfiltfilt


def butter_bandpass_filter(data, lowcut=0.01, highcut=0.3, fs=6.6, order=5):
    """
    Applies a zero-phase (forward-backward) Butterworth bandpass filter along
    axis 0, using second-order sections for numerical stability.
    """
    sos = butter(order, [lowcut, highcut], fs=fs, btype='bandpass', output="sos")
    return sosfiltfilt(sos, data, axis=0)


def bandpass_min_samples(lowcut=0.01, highcut=0.3, fs=6.6, order=5):
    """
    Minimum number of samples butter_bandpass_filter can process: sosfiltfilt's
    default edge padding (3 * ntaps) must be strictly shorter than the signal.
    Mirrors scipy's own padlen computation so callers can check up front.
    """
    sos = butter(order, [lowcut, highcut], fs=fs, btype='bandpass', output="sos")
    ntaps = 2 * sos.shape[0] + 1
    ntaps -= min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    return 3 * ntaps + 1


def calculate_scalp_coupling_index(w1_series, w2_series):
    """
    Scalp coupling index: Pearson correlation between the two wavelength
    signals of one optode pair (good skin contact -> highly correlated).
    Computed on the signals as given (no cardiac-band prefiltering).
    Returns 0.0 if either signal is flat.
    """
    if w1_series.std() > 0 and w2_series.std() > 0:
        return np.corrcoef(w1_series, w2_series)[0, 1]
    return 0.0


def calculate_coefficient_of_variation(series):
    """
    Coefficient of Variation (std / mean, in %) of one channel, used by SQA to
    flag noisy channels. Returns inf if the mean is zero.
    """
    mean_val = series.mean()
    if mean_val != 0:
        return (series.std() / mean_val) * 100.0
    return float('inf')


def apply_mbll_inversion(od_w1, od_w2, e_inv):
    """
    Converts optical density changes at two wavelengths into concentration
    changes via the Modified Beer-Lambert Law.

    e_inv is the inverse of the 2x2 extinction matrix whose rows are
    [e_HbO, e_HbR] per wavelength. Returns a 2 x N array: row 0 = delta HbO,
    row 1 = delta HbR. No pathlength/DPF scaling is applied, so values are
    relative (concentration x pathlength), not absolute concentrations.
    """
    return e_inv @ np.vstack([od_w1, od_w2])