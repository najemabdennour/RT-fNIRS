# core/signal_pipeline.py
"""
Channel-level signal preprocessing steps shared by every processing context.

Steps only need a DataFrame of channel columns and a params dict, with no
awareness of trials, runs, or behavioral data, so the same code serves both
the offline batch pipeline (core/offline_processor.py) and the real-time
neurofeedback pipeline (ui/neurofeedback_tab.py).

Every step has the signature step(data, params, excluded_cols) -> DataFrame.
Columns in excluded_cols (timestamps, markers, labels) are never treated as
channels. Paired-wavelength channels are expected to be named
'<base>_<wavelength>' (e.g. 'S(1)_D(2)_735'); see core.lsl_utils.
"""
import warnings

import numpy as np
import pandas as pd

from core.math_kernel import (
    butter_bandpass_filter,
    bandpass_min_samples,
    calculate_scalp_coupling_index,
    calculate_coefficient_of_variation,
    apply_mbll_inversion,
    optical_density,
    short_channel_regression,
)

# Documents the order that makes physical sense by default (OD conversion ->
# manual overrides -> quality control -> short-channel correction and removal
# -> concentration conversion -> filtering/detrending -> scaling). Callers are
# free to run any subset, in any order they configure.
SIGNAL_STEP_ORDER = ['optical_density', 'channel_filter', 'sci', 'sqa', 'short_channel_regression',
                     'short_channel_rejection', 'mbll', 'filter', 'detrend', 'scaling']

# DataFrame.attrs key recording what the signal columns hold: absent = raw
# intensity, 'od' = optical density, 'hb' = HbO/HbR concentrations. Lets later
# steps adapt to an earlier 'optical_density' step (pandas propagates attrs
# through the operations the pipeline uses, but not through a CSV round-trip).
SIGNAL_TYPE_ATTR = 'fnirs_signal'

def _with_columns(data, cols, values):
    """
    Returns a copy of data with the given columns replaced by the 2-D array
    `values` (n_samples x len(cols)), keeping the original column order and
    attrs. Building the frame in one block is far faster than
    `data[cols] = values` on thousands of channels, and never modifies `data`.
    """
    replaced = pd.DataFrame(values, columns=cols, index=data.index)
    kept = data.drop(columns=cols)
    out = pd.concat([kept, replaced], axis=1)[list(data.columns)]
    out.attrs = dict(data.attrs)
    return out


class SignalPreprocessor:
    """Executes the shared channel-level signal processing steps."""

    def __init__(self, log_callback=None, warning_callback=None):
        """
        Args:
            log_callback: callable(str) for routine log lines (default: print).
            warning_callback: optional callable(str) for non-fatal problems
                (e.g. a step skipped on a too-short buffer), so callers can
                surface them separately from INFO lines.
        """
        self.log_callback = log_callback
        self.warning_callback = warning_callback
        # {'channel', 'reason', 'metric'} dicts appended by the SCI, SQA and short-channel rejection steps.
        # Callers can export them (core.channel_lists) and re-apply them via
        # 'channel_filter', so short/noisy real-time buffers don't re-decide them.
        self.rejected_channels = []
        self._step_handlers = {
            'optical_density': self._step_optical_density,
            'channel_filter': self._step_channel_filter,
            'sci': self._step_sci,
            'sqa': self._step_sqa,
            'short_channel_regression': self._step_short_channel_regression,
            'short_channel_rejection': self._step_short_channel_rejection,
            'mbll': self._step_mbll,
            'filter': self._step_filter,
            'detrend': self._step_detrend,
            'scaling': self._step_scaling,
            'averaging (feature_vector)':self._step_reduce_to_feature_vector,
        }

    def log(self, message):
        """Send a message to log_callback, or print it if none was given."""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def warn(self, message):
        """
        Reports a non-fatal problem: via warning_callback if set, else through
        the log callback with a [WARNING] prefix, else as a Python RuntimeWarning.
        """
        if self.warning_callback:
            self.warning_callback(message)
        elif self.log_callback:
            self.log_callback(f"[WARNING] {message}")
        else:
            warnings.warn(message, RuntimeWarning, stacklevel=2)

    def available_steps(self):
        """Returns the set of step ids this engine knows how to execute."""
        return set(self._step_handlers.keys())

    def reset_rejections(self):
        """Clears the SCI/SQA/short-channel rejection log - call before each fresh run."""
        self.rejected_channels = []

    def get_rejected_channel_names(self):
        """
        Unique, sorted list of channel/base names SCI, SQA or short-channel
        rejection has dropped so far this run. SCI and short-channel
        rejections are stored as the shared base name (e.g. 'S(1)_D(2)') since
        they drop both wavelength columns of a pair; SQA rejections are stored
        as the exact column name it dropped, since SQA can reject a single
        wavelength channel independently of its pair. Both forms are matched
        correctly by _step_channel_filter.
        """
        return sorted({r['channel'] for r in self.rejected_channels})

    def run_steps(self, data, pipeline_steps, params, excluded_cols=None):
        """
        Executes an ordered list of {'id': str, 'run': bool} step dicts against `data`.
        Step ids this engine doesn't recognize are ignored, so callers can pass a
        mixed list (e.g. offline structural steps interleaved with signal steps) safely.
        'run' defaults to True. Returns the processed DataFrame.
        """
        excluded_cols = excluded_cols or []
        for step in pipeline_steps:
            step_id = step.get('id')
            if step_id not in self._step_handlers:
                continue
            if not step.get('run', True):
                self.log(f"[Signal Engine] Skipping step: {step_id.upper()}")
                continue
            self.log(f"[Signal Engine] Executing Processing Stage: {step_id.upper()}...")
            data = self._step_handlers[step_id](data, params, excluded_cols)
        return data

    # Step implementations

    def _step_optical_density(self, data, params, excluded_cols):
        """
        Converts raw intensity signal columns to optical density:
        OD = -ln(|I| / mean(|I|)) per channel over the given buffer. If any
        intensity is <= 0, values are replaced by their absolute value and
        clipped at the smallest non-zero channel minimum (with a warning), so
        the log is always defined.

        Reads no params. Column names are kept, and the result is marked as OD
        (data.attrs[SIGNAL_TYPE_ATTR] = 'od') so later steps adapt: 'sqa'
        evaluates the equivalent intensity CV and 'mbll' skips its own OD
        conversion. Data that is already OD (or HbO/HbR) is returned unchanged
        with a warning.
        """
        signal_type = data.attrs.get(SIGNAL_TYPE_ATTR)
        if signal_type in ('od', 'hb'):
            self.warn(f"Optical density conversion SKIPPED - the data is already "
                      f"{'optical density' if signal_type == 'od' else 'HbO/HbR concentration'}.")
            return data

        signal_cols = [c for c in data.columns if c not in excluded_cols]
        if not signal_cols:
            return data

        self.log(f"[OD Engine] Converting {len(signal_cols)} raw intensity channel(s) to optical density...")
        od, n_non_positive = optical_density(data[signal_cols].values)
        if n_non_positive:
            self.warn(f"Non-positive intensities encountered ({n_non_positive} sample(s)) during optical "
                      f"density conversion - set to abs(x) and clipped at the smallest non-zero channel "
                      f"minimum. Likely hardware or movement artifacts.")
        data = _with_columns(data, signal_cols, od)
        data.attrs[SIGNAL_TYPE_ATTR] = 'od'
        return data

    def _step_channel_filter(self, data, params, excluded_cols):
        """
        Manual, name-based channel keep/exclude filter.

        Independent of SCI/SQA's own live thresholds - this applies a fixed
        list of channel names (typically loaded from a file saved via
        core.channel_lists, e.g. the SCI/SQA rejections from a prior offline
        run) so the same channels can be excluded consistently across
        contexts/trials instead of being re-decided per buffer.

        params:
          'manual_exclude_channels' (default []): names to drop.
          'manual_keep_channels' (default []): names to keep; every other
              signal column is dropped. Only used if the exclude list is empty.
        With both lists empty, data is returned unchanged.

        Names are matched against both the exact column name (useful for a
        single-wavelength SQA-style rejection) and the column's base name -
        i.e. the part before a trailing '_<suffix>' (useful for a SCI-style
        pair rejection, and for matching a pre-MBLL wavelength column against
        a base name saved before MBLL ran).
        """
        exclude_list = set(params.get('manual_exclude_channels', []) or [])
        keep_list = set(params.get('manual_keep_channels', []) or [])
        if not exclude_list and not keep_list:
            return data

        signal_cols = [c for c in data.columns if c not in excluded_cols]
        if not signal_cols:
            return data

        mode = 'exclude' if exclude_list else 'keep'
        active_list = exclude_list if mode == 'exclude' else keep_list
        self.log(f"[Channel Filter] Applying manual {mode} list ({len(active_list)} entries)...")

        cols_to_drop = []
        for col in signal_cols:
            base = col.rsplit('_', 1)[0] if '_' in col else col
            in_list = (col in active_list) or (base in active_list)
            drop = in_list if mode == 'exclude' else not in_list
            if drop:
                cols_to_drop.append(col)

        if cols_to_drop:
            self.log(f" -> Dropping {len(cols_to_drop)} channel column(s): {cols_to_drop}")
            data = data.drop(columns=cols_to_drop, errors='ignore')
        else:
            self.log(" -> No channels in the current buffer matched the filter list.")
        return data

    def _step_sci(self, data, params, excluded_cols):
        """
        Scalp Coupling Index check: drops both wavelength columns of any optode
        pair whose two wavelengths correlate below the threshold.

        params:
          'sci_threshold' (default 0.5): minimum Pearson correlation.
          'mbll_wavelengths' (default [735, 850]): the two wavelength suffixes
              used to find '<base>_<wl1>' / '<base>_<wl2>' pairs.
        Pairs missing either column are left untouched. Rejected base names
        are recorded in self.rejected_channels. Run before 'mbll'.
        """
        sci_threshold = params.get('sci_threshold', 0.5)
        wavelengths = params.get('mbll_wavelengths', [735, 850])
        signal_cols = [c for c in data.columns if c not in excluded_cols]

        self.log(f"[SCI Engine] Evaluating optical scalp coupling contact thresholds (Limit: {sci_threshold})...")
        base_channels = dict.fromkeys(c.rsplit('_', 1)[0] for c in signal_cols if '_' in c)

        # Dropped once at the end: dropping inside the loop copies the whole frame per pair.
        cols_to_drop = []
        for base in base_channels:
            ch_w1, ch_w2 = f"{base}_{wavelengths[0]}", f"{base}_{wavelengths[1]}"
            if ch_w1 in data.columns and ch_w2 in data.columns:
                r_val = calculate_scalp_coupling_index(data[ch_w1], data[ch_w2])
                if r_val < sci_threshold:
                    self.log(f" -> [REJECTED] Base Channel '{base}' dropped via poor skin contact metrics. SCI = {r_val:.3f}")
                    self.rejected_channels.append({'channel': base, 'reason': 'sci', 'metric': round(float(r_val), 4)})
                    cols_to_drop += [ch_w1, ch_w2]
                else:
                    self.log(f" -> [PASSED] Base Channel '{base}' verified clean. SCI = {r_val:.3f}")
        if cols_to_drop:
            data = data.drop(columns=cols_to_drop, errors='ignore')
        return data

    def _step_sqa(self, data, params, excluded_cols):
        """
        Signal Quality Assessment: drops each signal column whose absolute
        Coefficient of Variation exceeds the threshold.

        params:
          'cv_threshold' (default 7.5): maximum |CV| in percent.
        Rejected column names are recorded in self.rejected_channels.

        The CV is an intensity measure. On optical density data (after the
        'optical_density' step) it is computed on exp(-OD) = I / mean(I),
        which has exactly the CV of the original intensity, so results match
        running SQA before the OD conversion. Don't run it after 'mbll'.
        """
        cv_threshold = params.get('cv_threshold', 7.5)
        signal_cols = [c for c in data.columns if c not in excluded_cols]
        is_od = data.attrs.get(SIGNAL_TYPE_ATTR) == 'od'

        self.log(f"[SQA Engine] Evaluating channel coupling efficiency (Threshold: {cv_threshold}%)"
                 f"{' on the intensity equivalent exp(-OD)' if is_od else ''}...")
        if not signal_cols:
            return data
        # All channels at once (vectorized), and dropped once at the end.
        cvs = calculate_coefficient_of_variation(np.exp(-data[signal_cols]) if is_od else data[signal_cols])
        cols_to_drop = []
        for col, cv in cvs.items():
            if abs(cv) > cv_threshold:
                self.log(f" -> [REJECTED] Channel '{col}' exceeded noise limits. CV = {cv:.2f}%")
                self.rejected_channels.append({'channel': col, 'reason': 'sqa', 'metric': round(float(cv), 4)})
                cols_to_drop.append(col)
            else:
                self.log(f" -> [PASSED] Channel '{col}' verified clean. CV = {cv:.2f}%")
        if cols_to_drop:
            data = data.drop(columns=cols_to_drop)
        return data

    # Short-separation channels: both steps read the same geometry params.
    #   'channel_lengths_mm' {pair key: source-detector distance (mm)} and
    #   'channel_midpoints' {pair key: 3D midpoint (mm)}, keyed like
    #   core.montage.pair_key (the column name without its trailing
    #   '_<wavelength>' / '_HbO' part), typically from
    #   core.montage.channel_geometry(montage); and
    #   'short_channel_max_distance_mm' (default 8.4, see config.py).

    def _split_short_long(self, data, params, excluded_cols):
        """
        Splits the signal columns into short-separation and long channels.

        Returns:
            (short_cols, long_cols, unknown_cols): channels whose pair's
            distance is at most the threshold, those above it, and those with
            no distance in 'channel_lengths_mm'.
        """
        lengths = params.get('channel_lengths_mm') or {}
        threshold = float(params.get('short_channel_max_distance_mm', 8.4))
        short, long_, unknown = [], [], []
        for col in data.columns:
            if col in excluded_cols:
                continue
            length = lengths.get(col.rsplit('_', 1)[0] if '_' in col else col)
            if length is None:
                unknown.append(col)
            elif length <= threshold:
                short.append(col)
            else:
                long_.append(col)
        return short, long_, unknown

    def _step_short_channel_regression(self, data, params, excluded_cols):
        """
        Short-separation channel regression: removes from every long channel
        the scalp (systemic) signal it shares with its nearest short channel,
        A_l' = A_l - alpha * A_s with alpha = <A_s, A_l> / <A_s, A_s>
        (Scholkmann et al. 2014, Eqs. 26-27; see math_kernel). Short channels
        themselves are left unchanged - run 'short_channel_rejection' after
        this step to drop them.

        params:
          'channel_lengths_mm', 'channel_midpoints',
          'short_channel_max_distance_mm' (default 8.4): see above.
        The nearest short channel is the one of the same wavelength (or, after
        'mbll', the same chromophore) whose 3D midpoint is closest to the long
        channel's; without midpoints the mean of all short channels of that
        wavelength is the regressor. Requires optical density or HbO/HbR data
        (run 'optical_density' first). Warns and returns data unchanged on raw
        intensity, without distances, or without short or long channels.
        """
        signal_type = data.attrs.get(SIGNAL_TYPE_ATTR)
        if signal_type not in ('od', 'hb'):
            self.warn("Short-channel regression SKIPPED - it needs optical density (or HbO/HbR) data. "
                      "Run the 'optical_density' step earlier in the pipeline.")
            return data
        if not params.get('channel_lengths_mm'):
            self.warn("Short-channel regression SKIPPED - no channel distances available (a channel layout "
                      "table, or stream metadata with source-detector distances, is needed).")
            return data

        short_cols, long_cols, unknown = self._split_short_long(data, params, excluded_cols)
        threshold = float(params.get('short_channel_max_distance_mm', 8.4))
        if unknown:
            self.warn(f"Short-channel regression: {len(unknown)} channel(s) have no known distance and are left as is.")
        if not short_cols:
            self.warn(f"Short-channel regression SKIPPED - no channel is {threshold:g} mm or shorter. "
                      f"Raise 'short_channel_max_distance_mm' to match the montage's shortest pairs.")
            return data
        if not long_cols:
            self.warn("Short-channel regression SKIPPED - every channel with a known distance is a short channel.")
            return data

        midpoints = params.get('channel_midpoints') or {}
        suffix = lambda col: col.rsplit('_', 1)[1] if '_' in col else ''
        pair = lambda col: col.rsplit('_', 1)[0] if '_' in col else col
        short_by_suffix = {}
        for col in short_cols:
            short_by_suffix.setdefault(suffix(col), []).append(col)
        self.log(f"[SSR Engine] Regressing {len(short_cols)} short channel(s) (<= {threshold:g} mm) out of "
                 f"{len(long_cols)} long channel(s)...")

        regressors, corrected, nearest_used, unpaired = {}, {}, 0, []
        for col in long_cols:
            candidates = short_by_suffix.get(suffix(col))
            if not candidates:
                unpaired.append(col)
                continue
            located = [c for c in candidates if pair(c) in midpoints]
            if located and pair(col) in midpoints:
                chosen = min(located, key=lambda c: np.linalg.norm(midpoints[pair(c)] - midpoints[pair(col)]))
                regressor = data[chosen].to_numpy(dtype=float)
                nearest_used += 1
            else:
                key = tuple(candidates)
                if key not in regressors:
                    regressors[key] = data[list(candidates)].mean(axis=1).to_numpy(dtype=float)
                regressor = regressors[key]
            corrected[col], _ = short_channel_regression(data[col].to_numpy(dtype=float), regressor)

        if corrected:
            data = _with_columns(data, list(corrected), np.column_stack(list(corrected.values())))
        self.log(f" -> Corrected {len(corrected)} long channel(s): {nearest_used} with their nearest short channel, "
                 f"{len(corrected) - nearest_used} with the mean of the short channels of the same wavelength.")
        if unpaired:
            self.warn(f"Short-channel regression left {len(unpaired)} long channel(s) uncorrected - no short channel "
                      f"of the same wavelength: {unpaired[:8]}{' ...' if len(unpaired) > 8 else ''}")
        return data

    def _step_short_channel_rejection(self, data, params, excluded_cols):
        """
        Drops every short-separation channel (source-detector distance at
        most 'short_channel_max_distance_mm'), so channels that sample the
        scalp rather than the brain never become decoder features. Meant to
        run after 'short_channel_regression', which still needs them.

        params: 'channel_lengths_mm', 'short_channel_max_distance_mm' (see
        above). Rejected pairs are recorded in self.rejected_channels with
        reason 'short'. Warns and returns data unchanged without distances.
        """
        if not params.get('channel_lengths_mm'):
            self.warn("Short-channel rejection SKIPPED - no channel distances available (a channel layout "
                      "table, or stream metadata with source-detector distances, is needed).")
            return data
        short_cols, _, unknown = self._split_short_long(data, params, excluded_cols)
        threshold = float(params.get('short_channel_max_distance_mm', 8.4))
        lengths = params['channel_lengths_mm']
        if unknown:
            self.warn(f"Short-channel rejection: {len(unknown)} channel(s) have no known distance and are kept.")
        if not short_cols:
            self.log(f"[Short Channels] No channel is {threshold:g} mm or shorter - nothing removed.")
            return data
        pairs = dict.fromkeys(c.rsplit('_', 1)[0] if '_' in c else c for c in short_cols)
        for key in pairs:
            self.rejected_channels.append({'channel': key, 'reason': 'short', 'metric': round(float(lengths[key]), 4)})
        self.log(f"[Short Channels] Removing {len(short_cols)} short-separation channel column(s) "
                 f"({len(pairs)} optode pair(s) at or below {threshold:g} mm).")
        return data.drop(columns=short_cols)

    def _step_mbll(self, data, params, excluded_cols):
        """
        Modified Beer-Lambert Law: converts paired raw intensity columns into
        '<base>_HbO' and '<base>_HbR' concentration-change columns.

        params:
          'mbll_wavelengths' (default [735, 850]): the two wavelengths to pair.
          'extinction_coefficients' (default {}): {str(wavelength): [e_HbO, e_HbR]}.
        Input may be raw intensity, converted here to optical density as
        -log(I / mean(I)) over the given buffer, or optical density from an
        earlier 'optical_density' step, used as is. No DPF or source-detector
        distance is applied. Raw-intensity input containing non-positive values
        triggers a warning, since it is probably OD data whose marker was lost
        (e.g. a checkpoint CSV saved after the OD step). If at least one pair converts,
        the result holds only the excluded (non-signal) columns plus the new
        HbO/HbR columns - unpaired channels are dropped. If the coefficients
        are missing or no pair is found, data is returned unchanged.
        """
        wavelengths = params.get('mbll_wavelengths', [735, 850])
        ext_coeffs = params.get('extinction_coefficients', {})
        signal_cols = [c for c in data.columns if c not in excluded_cols]

        is_od = data.attrs.get(SIGNAL_TYPE_ATTR) == 'od'
        self.log(f"[MBLL Engine] Converting {'Optical Density' if is_od else 'Raw Intensity'} to Concentration Matrix...")
        # Channel order follows the input (not a set), so the HbO/HbR column order -
        # and thus a decoder's feature order - is the same in every run.
        base_channels = dict.fromkeys(c.rsplit('_', 1)[0] for c in signal_cols if '_' in c)
        mbll_df = data[[c for c in excluded_cols if c in data.columns]].copy()
        hb_columns = {}
        non_positive = 0

        if len(wavelengths) >= 2 and str(wavelengths[0]) in ext_coeffs and str(wavelengths[1]) in ext_coeffs:
            E_inv = np.linalg.inv(np.array([ext_coeffs[str(wavelengths[0])], ext_coeffs[str(wavelengths[1])]]))
            converted = False
            for base in base_channels:
                ch_w1, ch_w2 = f"{base}_{wavelengths[0]}", f"{base}_{wavelengths[1]}"
                if ch_w1 in data.columns and ch_w2 in data.columns:
                    if is_od:
                        od_w1, od_w2 = data[ch_w1], data[ch_w2]
                    else:
                        non_positive += int((data[ch_w1] <= 0).any()) + int((data[ch_w2] <= 0).any())
                        od_w1 = -np.log(data[ch_w1] / data[ch_w1].mean())
                        od_w2 = -np.log(data[ch_w2] / data[ch_w2].mean())
                    conc = apply_mbll_inversion(od_w1, od_w2, E_inv)
                    hb_columns[f"{base}_HbO"] = conc[0, :]
                    hb_columns[f"{base}_HbR"] = conc[1, :]
                    converted = True
            if non_positive:
                self.warn(f"MBLL input treated as raw intensity contains non-positive values in {non_positive} "
                          f"channel(s) - it may already be optical density (e.g. a checkpoint saved after the "
                          f"OD step). Their HbO/HbR values will contain NaN; enable 'optical_density' in this "
                          f"pipeline instead of reloading OD data.")
            if converted:
                # One concat instead of a column insert per channel (avoids a fragmented frame).
                attrs = dict(mbll_df.attrs)
                mbll_df = pd.concat([mbll_df, pd.DataFrame(hb_columns, index=data.index)], axis=1)
                mbll_df.attrs.update(attrs)
                mbll_df.attrs[SIGNAL_TYPE_ATTR] = 'hb'
                data = mbll_df
        return data

    def _step_filter(self, data, params, excluded_cols):
        """
        Zero-phase Butterworth bandpass filter applied to every signal column.

        params:
          'low_cut' (default 0.05), 'high_cut' (default 0.5): band edges (Hz).
          'fs' (default 6.6): sampling rate (Hz).
          'butterworth_order' (default 5): filter order.
        Returns a filtered copy; if the parameters are invalid, the buffer is
        too short for filtfilt padding, or filtering fails, warns and returns
        data unfiltered.
        """
        low_cut = params.get('low_cut', 0.05)
        high_cut = params.get('high_cut', 0.5)
        fs = params.get('fs', 6.6)
        b_order = params.get('butterworth_order', 5)
        signal_cols = [c for c in data.columns if c not in excluded_cols]

        if signal_cols:
            try:
                min_samples = bandpass_min_samples(low_cut, high_cut, fs, b_order)
            except Exception as e:
                self.warn(
                    f"Butterworth filter pass SKIPPED - invalid filter parameters "
                    f"({low_cut}-{high_cut} Hz, fs={fs} Hz, order={b_order}): {e}. "
                    f"Data passed through UNFILTERED."
                )
                return data

            # Short real-time buffers can be too small for filtfilt's padding
            # requirements - skip rather than crash the pipeline, but say so.
            n_samples = len(data)
            if n_samples < min_samples:
                self.warn(
                    f"Butterworth filter pass SKIPPED - buffer too short: {n_samples} sample(s) "
                    f"available, at least {min_samples} needed (~{min_samples / fs:.1f} s at "
                    f"fs={fs} Hz) for an order-{b_order} {low_cut}-{high_cut} Hz bandpass. "
                    f"Data passed through UNFILTERED."
                )
                return data

            try:
                self.log(f"[Pipeline Engine] Applying Butterworth Filter ({low_cut}-{high_cut} Hz)...")
                # One filter design applied to all channels at once (axis 0), not per column.
                filtered = butter_bandpass_filter(data[signal_cols].values, low_cut, high_cut, fs, b_order)
                data = _with_columns(data, signal_cols, filtered)
            except Exception as e:
                self.warn(f"Butterworth filter pass SKIPPED: {e}. Data passed through UNFILTERED.")
        return data

    def _step_detrend(self, data, params, excluded_cols):
        """
        Piecewise linear detrending of every signal column (nilearn's _detrend).

        params:
          'detrend_batches' (default 10): number of batches, capped at the
              number of rows.
        Returns a detrended copy. Skipped (with a log message)
        if nilearn is not installed or detrending fails.
        """
        d_batches = params.get('detrend_batches', 10)
        signal_cols = [c for c in data.columns if c not in excluded_cols]

        if signal_cols:
            try:
                from nilearn.signal import _detrend
            except ImportError:
                self.log("[WARNING] 'nilearn' library not discovered. Skipping detrend pass.")
                return data
            try:
                # Clamp batch count to buffer length so short real-time trial
                # buffers don't blow up on a batch size sized for full sessions.
                n_batches = min(d_batches, max(1, len(data)))
                self.log(f"[Pipeline Engine] Executing linear batch detrending passes...")
                detrended_signals = _detrend(data[signal_cols], inplace=False, type='linear', n_batches=n_batches)
                data = _with_columns(data, signal_cols, detrended_signals)
            except Exception as e:
                self.log(f"[WARNING] Detrend pass skipped: {e}")
        return data

    def _step_scaling(self, data, params, excluded_cols):
        """
        Scales every signal column, fitting the scaler on this data alone.

        params:
          'scaler_type' (default 'standard'): 'minmax' for MinMaxScaler,
              anything else for StandardScaler (z-score).
        Returns a scaled copy.
        """
        signal_cols = [c for c in data.columns if c not in excluded_cols]
        if signal_cols:
            scaler_type = params.get('scaler_type', 'standard')
            self.log(f"[Pipeline Engine] Applying Signal Amplitude Transformation ({scaler_type.upper()} Scaling)...")
            if scaler_type == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            else:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()

            data = _with_columns(data, signal_cols, scaler.fit_transform(data[signal_cols]))
        return data

    def _step_reduce_to_feature_vector(self,data,params,exculed_cols):
        """
        Collapses a processed time-series buffer into the single-row mean feature
        vector a trial-averaged decoder expects (the live, single-trial
        counterpart of the offline 'averaging' step).

        Reads no params. Unlike the offline step, it averages every column,
        including excluded (non-signal) ones, so callers must drop those
        first. Returns data unchanged if it is None or empty.
        """
        if data is None or data.empty:
            return data
        return data.mean(axis=0).to_frame().T