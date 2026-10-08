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

from core.math_kernel import (
    butter_bandpass_filter,
    bandpass_min_samples,
    calculate_scalp_coupling_index,
    calculate_coefficient_of_variation,
    apply_mbll_inversion
)

# Documents the order that makes physical sense by default (manual overrides
# -> quality control -> concentration conversion -> filtering/detrending ->
# scaling). Callers are free to run any subset, in any order they configure.
SIGNAL_STEP_ORDER = ['channel_filter', 'sci', 'sqa', 'mbll', 'filter', 'detrend', 'scaling']


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
        # {'channel', 'reason', 'metric'} dicts appended by _step_sci/_step_sqa.
        # Callers can export them (core.channel_lists) and re-apply them via
        # 'channel_filter', so short/noisy real-time buffers don't re-decide them.
        self.rejected_channels = []
        self._step_handlers = {
            'channel_filter': self._step_channel_filter,
            'sci': self._step_sci,
            'sqa': self._step_sqa,
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
        """Clears the SCI/SQA rejection log - call before each fresh run."""
        self.rejected_channels = []

    def get_rejected_channel_names(self):
        """
        Unique, sorted list of channel/base names SCI or SQA has dropped so
        far this run. SCI rejections are stored as the shared base name
        (e.g. 'S(1)_D(2)') since it drops both wavelength columns of a pair;
        SQA rejections are stored as the exact column name it dropped, since
        SQA can reject a single wavelength channel independently of its pair.
        Both forms are matched correctly by _step_channel_filter.
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
        base_channels = set([c.rsplit('_', 1)[0] for c in signal_cols if '_' in c])

        for base in base_channels:
            ch_w1, ch_w2 = f"{base}_{wavelengths[0]}", f"{base}_{wavelengths[1]}"
            if ch_w1 in data.columns and ch_w2 in data.columns:
                r_val = calculate_scalp_coupling_index(data[ch_w1], data[ch_w2])
                if r_val < sci_threshold:
                    self.log(f" -> [REJECTED] Base Channel '{base}' dropped via poor skin contact metrics. SCI = {r_val:.3f}")
                    self.rejected_channels.append({'channel': base, 'reason': 'sci', 'metric': round(float(r_val), 4)})
                    data = data.drop(columns=[ch_w1, ch_w2], errors='ignore')
                else:
                    self.log(f" -> [PASSED] Base Channel '{base}' verified clean. SCI = {r_val:.3f}")
        return data

    def _step_sqa(self, data, params, excluded_cols):
        """
        Signal Quality Assessment: drops each signal column whose absolute
        Coefficient of Variation exceeds the threshold.

        params:
          'cv_threshold' (default 7.5): maximum |CV| in percent.
        Rejected column names are recorded in self.rejected_channels. Meant
        for raw intensities (run before 'mbll'): zero-mean signals give huge CVs.
        """
        cv_threshold = params.get('cv_threshold', 7.5)
        signal_cols = [c for c in data.columns if c not in excluded_cols]

        self.log(f"[SQA Engine] Evaluating channel coupling efficiency (Threshold: {cv_threshold}%)...")
        for col in signal_cols:
            if col not in data.columns:
                continue
            cv = calculate_coefficient_of_variation(data[col])
            if abs(cv) > cv_threshold:
                self.log(f" -> [REJECTED] Channel '{col}' exceeded noise limits. CV = {cv:.2f}%")
                self.rejected_channels.append({'channel': col, 'reason': 'sqa', 'metric': round(float(cv), 4)})
                data = data.drop(columns=[col])
            else:
                self.log(f" -> [PASSED] Channel '{col}' verified clean. CV = {cv:.2f}%")
        return data

    def _step_mbll(self, data, params, excluded_cols):
        """
        Modified Beer-Lambert Law: converts paired raw intensity columns into
        '<base>_HbO' and '<base>_HbR' concentration-change columns.

        params:
          'mbll_wavelengths' (default [735, 850]): the two wavelengths to pair.
          'extinction_coefficients' (default {}): {str(wavelength): [e_HbO, e_HbR]}.
        Optical density is -log(I / mean(I)) over the given buffer; no DPF or
        source-detector distance is applied. If at least one pair converts,
        the result holds only the excluded (non-signal) columns plus the new
        HbO/HbR columns - unpaired channels are dropped. If the coefficients
        are missing or no pair is found, data is returned unchanged.
        """
        wavelengths = params.get('mbll_wavelengths', [735, 850])
        ext_coeffs = params.get('extinction_coefficients', {})
        signal_cols = [c for c in data.columns if c not in excluded_cols]

        self.log("[MBLL Engine] Converting Raw Intensity to Concentration Matrix...")
        base_channels = set([c.rsplit('_', 1)[0] for c in signal_cols if '_' in c])
        mbll_df = data[[c for c in excluded_cols if c in data.columns]].copy()

        if len(wavelengths) >= 2 and str(wavelengths[0]) in ext_coeffs and str(wavelengths[1]) in ext_coeffs:
            E_inv = np.linalg.inv(np.array([ext_coeffs[str(wavelengths[0])], ext_coeffs[str(wavelengths[1])]]))
            converted = False
            for base in base_channels:
                ch_w1, ch_w2 = f"{base}_{wavelengths[0]}", f"{base}_{wavelengths[1]}"
                if ch_w1 in data.columns and ch_w2 in data.columns:
                    od_w1 = -np.log(data[ch_w1] / data[ch_w1].mean())
                    od_w2 = -np.log(data[ch_w2] / data[ch_w2].mean())
                    conc = apply_mbll_inversion(od_w1, od_w2, E_inv)
                    mbll_df[f"{base}_HbO"] = conc[0, :]
                    mbll_df[f"{base}_HbR"] = conc[1, :]
                    converted = True
            if converted:
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
                data_filtered = data.copy()
                data_filtered[signal_cols] = data_filtered[signal_cols].apply(
                    butter_bandpass_filter, args=(low_cut, high_cut, fs, b_order)
                )
                data = data_filtered
            except Exception as e:
                self.warn(f"Butterworth filter pass SKIPPED: {e}. Data passed through UNFILTERED.")
        return data

    def _step_detrend(self, data, params, excluded_cols):
        """
        Piecewise linear detrending of every signal column (nilearn's _detrend).

        params:
          'detrend_batches' (default 10): number of batches, capped at the
              number of rows.
        Modifies data in place and returns it. Skipped (with a log message)
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
                for i, col in enumerate(signal_cols):
                    data[col] = detrended_signals[:, i]
            except Exception as e:
                self.log(f"[WARNING] Detrend pass skipped: {e}")
        return data

    def _step_scaling(self, data, params, excluded_cols):
        """
        Scales every signal column, fitting the scaler on this data alone.

        params:
          'scaler_type' (default 'standard'): 'minmax' for MinMaxScaler,
              anything else for StandardScaler (z-score).
        Modifies data in place and returns it.
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

            data[signal_cols] = scaler.fit_transform(data[signal_cols])
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