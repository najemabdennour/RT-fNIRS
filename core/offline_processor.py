# core/offline_processor.py
"""
Offline batch processing: merging recorded runs and behavioral logs into a
session file, and running the configurable offline preprocessing pipeline.
"""
import numbers
import os
import re
from datetime import datetime
import pandas as pd

import config
from core.signal_pipeline import SignalPreprocessor
from core.io_utils import read_csv, write_csv


def _normalize_label(value):
    """Returns whole-number numeric labels as int (2.0 -> 2) and any other label unchanged."""
    if isinstance(value, numbers.Real) and not isinstance(value, bool) and float(value).is_integer():
        return int(value)
    return value


class OfflineSessionCombiner:
    """Merges per-run fNIRS recordings and per-run behavioral logs into session-level CSVs."""

    @staticmethod
    def discover_folders_by_date(recordings_root, behavioral_root, target_date=None):
        """
        Finds the recording and behavioral sub-folders whose names contain
        target_date (default: today, 'YYYY-MM-DD'; matched as a regex).

        Returns (recordings_dir, behavioral_dir), each the most recently
        modified match, or the corresponding root itself when there is no match or either root
        does not exist.
        """
        if target_date is None:
            target_date = datetime.today().strftime("%Y-%m-%d")

        if not os.path.exists(recordings_root) or not os.path.exists(behavioral_root):
            return recordings_root, behavioral_root

        data_folders = [i for i in os.listdir(recordings_root) if not i.startswith(".") and os.path.isdir(os.path.join(recordings_root, i))]
        behav_folders = [i for i in os.listdir(behavioral_root) if not i.startswith(".")]

        rec_matched = [os.path.join(recordings_root, i) for i in data_folders if re.search(target_date, i)]
        beh_matched = [os.path.join(behavioral_root, i) for i in behav_folders if re.search(target_date, i)]

        rec_matched.sort(key=lambda x: os.path.getmtime(x))
        beh_matched.sort(key=lambda x: os.path.getmtime(x))

        return rec_matched[-1] if rec_matched else recordings_root, beh_matched[-1] if beh_matched else behavioral_root

    @staticmethod
    def process_and_combine(rec_dir, beh_dir, out_rec_dir, out_beh_dir, experiment=config.DEFAULT_EXPERIMENT):
        """
        Concatenates every recording CSV in rec_dir (sorted by name) into one
        session, and the matching behavioral CSVs in beh_dir into one log.

        Each recording is trimmed to start at its first 'run_onset' marker and
        tagged with run='run<N>'; behavioral files are tagged the same way and
        only as many as there are recordings are used. Writes
        Session_<today>.csv to out_rec_dir and, if any behavioral data was
        found, behav_<today>.csv to out_beh_dir.

        Returns:
            (session_path, behav_path or "", session_shape, behav_shape)

        Raises FileNotFoundError if rec_dir holds no CSV files.
        """
        rec_files = sorted([f for f in os.listdir(rec_dir) if not f.startswith(".") and f.endswith('.csv')])

        if not rec_files:
            raise FileNotFoundError(f"No valid CSV stream files detected inside: {rec_dir}")

        today_str = datetime.today().strftime("%Y-%m-%d")
        behav_cols = config.get_behavioral_columns(experiment)

        combined_runs = []
        for idx, file_name in enumerate(rec_files):
            df = read_csv(os.path.join(rec_dir, file_name))
            first_idx = df["run_onset"].first_valid_index()
            if first_idx is not None:
                df = df[first_idx:].reset_index(drop=True)
            df["run"] = f"run{idx + 1}"
            combined_runs.append(df)

        runs_df = pd.concat(combined_runs).reset_index(drop=True)

        behav_df = pd.DataFrame(columns=behav_cols)
        if beh_dir and os.path.exists(beh_dir):
            beh_files = sorted([f for f in os.listdir(beh_dir) if not f.startswith(".") and f.endswith('.csv')])
            combined_behav = []
            for idx, file_name in enumerate(beh_files):
                if idx >= len(combined_runs): break
                df = read_csv(os.path.join(beh_dir, file_name))
                df["run"] = f"run{idx + 1}"
                combined_behav.append(df)
            if combined_behav:
                merged_b = pd.concat(combined_behav)
                valid_cols = [c for c in merged_b.columns if c in merged_b.columns]
                behav_df = merged_b[valid_cols].reset_index(drop=True)

        os.makedirs(out_rec_dir, exist_ok=True)
        session_out = os.path.join(out_rec_dir, f"Session_{today_str}.csv")
        write_csv(runs_df, session_out)

        behav_out = ""
        if out_beh_dir and not behav_df.empty:
            os.makedirs(out_beh_dir, exist_ok=True)
            behav_out = os.path.join(out_beh_dir, f"behav_{today_str}.csv")
            write_csv(behav_df, behav_out)

        return session_out, behav_out, runs_df.shape, behav_df.shape


class OfflineDataPreprocessor:
    """
    Orchestrates the full offline pipeline: channel-level signal processing
    (delegated to core.signal_pipeline.SignalPreprocessor, the same engine
    shared with the real-time neurofeedback pipeline) plus the session-
    structural steps that only make sense with a full recorded session and
    an aligned behavioral log (epoching, trial averaging, block averaging).
    """

    def __init__(self, log_callback=None, warning_callback=None):
        """Creates the shared signal engine (both callbacks passed through) and registers the structural steps."""
        self.log_callback = log_callback
        self.signal_engine = SignalPreprocessor(log_callback=log_callback, warning_callback=warning_callback)
        self._structural_handlers = {
            'epoch_prep': self._step_epoch_prep,
            'averaging': self._step_averaging,
            'block_avg': self._step_block_avg
        }

    def log(self, message):
        """Send a message to log_callback, or print it if none was given."""
        if self.log_callback: self.log_callback(message)
        else: print(message)

    def get_rejected_channels(self):
        """Passthrough to the signal engine's SCI/SQA rejection log for the last run."""
        return self.signal_engine.get_rejected_channel_names()

    def run_preprocessing_pipeline(self, session_df, behav_df, params, output_dir, today_str):
        """
        Runs the configured steps in order on a copy of session_df.

        params['pipeline_steps'] is a list of {'id', 'run', 'save'} dicts; ids
        are dispatched to SignalPreprocessor (channel-level steps) or to the
        structural steps here. Steps with run=False and unknown ids are skipped.
        After each step with save=True, the current data is written to
        output_dir/data_checkpoint_<id>_<today_str>.csv.
        params['active_experiment'] selects which behavioral columns are
        treated as non-signal.

        Returns a {step_id: checkpoint_path} dict. The final DataFrame itself
        is not returned, so only saved checkpoints persist.
        """
        data = session_df.copy()
        pipeline_steps = params.get('pipeline_steps', [])
        saved_paths = {}
        self.signal_engine.reset_rejections()

        # Shared structural columns plus this experiment's behavioral columns,
        # which epoch_prep merges in and which must never be treated as channels.
        excluded_cols = config.get_non_signal_columns(params.get('active_experiment', config.DEFAULT_EXPERIMENT))

        for step in pipeline_steps:
            step_id = step['id']
            if not step['run']:
                self.log(f"[Pipeline Engine] Skipping step sequence trace: {step_id.upper()}")
                continue

            if step_id in self.signal_engine.available_steps():
                data = self.signal_engine.run_steps(data, [step], params, excluded_cols)
            elif step_id in self._structural_handlers:
                self.log(f"[Pipeline Engine] Executing Processing Stage: {step_id.upper()}...")
                data = self._structural_handlers[step_id](data, params, excluded_cols, behav_df)
            else:
                self.log(f"[WARNING] Unknown pipeline step id '{step_id}' - skipped.")
                continue

            if step['save']:
                out_path = os.path.join(output_dir, f"data_checkpoint_{step_id}_{today_str}.csv")
                write_csv(data, out_path, log_callback=self.log)
                self.log(f" [SAVE GATE] Exported intermediate step checkpoint tracking logs -> {out_path}")
                saved_paths[step_id] = out_path

        return saved_paths

    # Session-structural steps (need a full session and, optionally, a behavioral log)

    def _step_epoch_prep(self, data, params, excluded_cols, behav_df):
        """
        Aligns the behavioral log onto trials and cuts the session into epochs.

        params:
          'peak_onset_buffer' (default 4), 'peak_offset_buffer' (default 9):
              epoch window, in seconds after each trial onset.
          'active_experiment' (default config.DEFAULT_EXPERIMENT): selects the
              experiment profile (target/carry columns, trial filter).

        Steps: marks 'peak_onset'/'peak_offset' on trial-onset rows; copies the
        profile's target column into 'target' and its carry columns onto
        trial-onset rows, in order (one behavioral row per trial), then
        forward-fills them ('target' defaults to 1); drops rows before the
        first trial onset; keeps only rows matching 'keep_trials_where'; and
        finally keeps only samples inside each epoch window, numbering epochs
        in the 'index' column (1 for all rows if no epochs were found).

        Raises ValueError if the profile requires a behavioral log (or its
        target column) and it is missing.
        """
        peak_onset_buffer = params.get('peak_onset_buffer', 4)
        peak_offset_buffer = params.get('peak_offset_buffer', 9)
        experiment = params.get('active_experiment', config.DEFAULT_EXPERIMENT)
        profile = config.get_experiment_profile(experiment)
        has_behav = behav_df is not None and not behav_df.empty

        if "trial_onset" in data.columns and data["trial_onset"].notna().any():
            self.log("[Pipeline Engine] Standardizing dataset timing boundaries...")
            mask = data["trial_onset"].notna()
            data['peak_onset'] = data[mask]["timestamps"] + peak_onset_buffer
            data['peak_offset'] = data[mask]["timestamps"] + peak_offset_buffer

        if profile['behavior_required'] and not has_behav:
            raise ValueError(f"Experiment profile '{experiment}' requires a behavioral log, but none was provided.")

        # Align the behavioral log onto trial-onset rows: one log row per trial,
        # in order. Which columns are copied is defined by the experiment profile
        # (config.EXPERIMENT_PROFILES), not hard-coded per paradigm here.
        target_col = profile['target_column']
        carry_cols = profile['carry_columns']
        if has_behav and 'trial_onset' in data.columns:
            missing = [c for c in [target_col] + carry_cols if c and c not in behav_df.columns]
            if target_col in missing and profile['behavior_required']:
                raise ValueError(f"Behavioral log is missing target column '{target_col}' required by profile '{experiment}'.")
            if missing:
                self.signal_engine.warn(f"Behavioral log is missing column(s) {missing} expected by profile '{experiment}' - not aligned.")

            trial_starts = data['trial_onset'].notna()
            n_trials = int(trial_starts.sum())
            if target_col and target_col in behav_df.columns:
                data.loc[trial_starts, 'target'] = behav_df[target_col].values[:n_trials]
            for col in carry_cols:
                if col in behav_df.columns:
                    data.loc[trial_starts, col] = behav_df[col].values[:n_trials]

        if 'target' not in data.columns: data['target'] = 1

        for track in dict.fromkeys(['target'] + carry_cols):
            if track in data.columns: data[track] = data[track].ffill()

        if 'trial_onset' in data.columns and data["trial_onset"].notna().any():
            first_idx = data["trial_onset"].first_valid_index()
            if first_idx is not None: data = data[first_idx:].reset_index(drop=True)

        for col, keep_value in profile['keep_trials_where'].items():
            if col in data.columns:
                data = data[data[col] == keep_value].reset_index(drop=True)

        has_epochs = 'peak_onset' in data.columns and data['peak_onset'].notnull().any()
        if has_epochs:
            p_on = data[data["peak_onset"].notnull()]["peak_onset"].unique().tolist()
            p_off = data[data["peak_offset"].notnull()]["peak_offset"].unique().tolist()
            slices = []
            for i, (start, end) in enumerate(zip(p_on, p_off)):
                mask = (data['timestamps'] >= start) & (data['timestamps'] <= end)
                subset = data[mask].copy()
                subset['index'] = i + 1
                slices.append(subset)
            if slices:
                data = pd.concat(slices, ignore_index=True)
        if 'index' not in data.columns: data['index'] = 1
        return data

    def _step_averaging(self, data, params, excluded_cols, behav_df):
        """
        Reduces each epoch (unique 'index' value) to one row: the mean of every
        signal column, plus the epoch's 'target' and 'run'.

        Labels keep their type, so text conditions work; whole-number floats
        (e.g. 2.0 after a CSV round-trip) are normalized to int.

        Reads no params. Requires 'target' and 'index' columns (run
        'epoch_prep' first); otherwise returns data unchanged.
        """
        signal_cols = [c for c in data.columns if c not in excluded_cols]

        if signal_cols and 'target' in data.columns and 'index' in data.columns:
            self.log("[Pipeline Engine] Synthesizing trial-level response averaging...")
            targets, groups, run_avg = [], [], pd.DataFrame()
            for trial_id in data["index"].unique():
                tranch = data[data["index"] == trial_id]
                if tranch.empty: continue
                targets.append(_normalize_label(tranch["target"].iloc[0]))
                groups.append(tranch["run"].iloc[0] if "run" in tranch.columns else "run1")
                run_avg = pd.concat([run_avg, tranch[signal_cols].mean().to_frame().T], axis=0, ignore_index=True)
            run_avg["target"] = targets
            run_avg["run"] = groups
            data = run_avg
        return data

    def _step_block_avg(self, data, params, excluded_cols, behav_df):
        """
        Averages consecutive rows sharing the same 'target' within a run into
        one row per block, returning columns 'run', 'target' and the signal
        columns. Reads no params; returns data unchanged without 'target'.
        """
        signal_cols = [c for c in data.columns if c not in excluded_cols]

        if signal_cols and 'target' in data.columns:
            self.log("[Pipeline Engine] Synthesizing block condition averaging matrix...")
            data_avg_blk = data.copy()
            if "run" not in data_avg_blk.columns: data_avg_blk["run"] = "run1"
            new_block = (data_avg_blk['target'] != data_avg_blk['target'].shift()) | (data_avg_blk['run'] != data_avg_blk['run'].shift())
            data_avg_blk['block_id'] = new_block.cumsum()

            ex_cols = [c for c in excluded_cols if c not in ['run', 'target', 'block_id']]
            signal_and_group = [c for c in data_avg_blk.columns if c not in ex_cols]
            data_avg_blk = data_avg_blk[signal_and_group].groupby(['run', 'block_id', 'target']).mean().reset_index()
            data_avg_blk = data_avg_blk.drop(columns='block_id', errors='ignore')
            data = data_avg_blk
        return data