"""
Real-time neurofeedback tab: deploys a trained decoder against live LSL
streams and broadcasts per-trial class probabilities back over LSL.
"""
import os
import time
import joblib
import pandas as pd
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QCheckBox, QGroupBox, QFileDialog,
    QMessageBox, QSplitter, QProgressBar, QTextEdit, QComboBox,
    QScrollArea, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import config
from core.lsl_utils import resolve_channel_names
from core.signal_pipeline import SignalPreprocessor
from core.decoders import DECODER_REGISTRY, decoder_selection
from core.channel_lists import load_channel_list
from core.io_utils import read_csv, find_index_columns
from ui.components import CollapsibleSection, PipelineStepTable


class NeurofeedbackWorker(QThread):
    """
    Asynchronous LSL worker handling synchronized baselines, probabilities,
    and co-adaptation.

    Buffered NIRS samples pass through the same core.signal_pipeline steps
    used offline before being handed to the deployed model.

    Co-adaptation, when enabled, loads a user-supplied base training CSV as
    the initial co-adaptation training set and starts with coadapt_model
    pointing at the loaded production model (no fit happens at startup).
    Each live trial whose target-class probability exceeds
    config.COADAPTATION_ACCEPTANCE_CRITERIA is appended to that set, and a
    brand-new decoder (core.decoders.DECODER_REGISTRY key
    coadapt_decoder_key) is fit from scratch on the accumulated data and
    saved as '<model>_coadapted.pkl'.
    """
    log_signal = pyqtSignal(str, str)
    prediction_signal = pyqtSignal(int, float)  # target_class, probability
    status_signal = pyqtSignal(str)

    def __init__(self, model_path, sig_type, marker_name, coadaptation_enabled, pipeline_steps, preproc_params,
                 coadapt_decoder_key="logisticregression", coadapt_base_data_path="", apply_coadapt_predictions=True):
        """
        Args:
            model_path: Path to the trained production model (.pkl).
            sig_type: LSL stream type of the data stream (e.g. "NIRS").
            marker_name: LSL name of the presentation marker stream.
            coadaptation_enabled: Whether to accumulate accepted trials and
                refit/save a co-adaptive decoder.
            pipeline_steps: Ordered step dicts from PipelineStepTable.compiled_steps().
            preproc_params: Parameter dict for SignalPreprocessor.run_steps().
            coadapt_decoder_key: DECODER_REGISTRY key of the co-adaptive decoder.
            coadapt_base_data_path: CSV seeding the co-adaptation training set.
            apply_coadapt_predictions: Whether live predictions use the
                co-adaptive decoder (True) or stay on the base model (False).
        """
        super().__init__()
        self.model_path = model_path
        self.sig_type = sig_type
        self.marker_name = marker_name
        self.coadaptation_enabled = coadaptation_enabled
        self.pipeline_steps = pipeline_steps
        self.preproc_params = preproc_params
        self.coadapt_decoder_key = coadapt_decoder_key
        self.coadapt_base_data_path = coadapt_base_data_path
        # Decouples training the co-adaptive decoder (gated by
        # coadaptation_enabled) from using it for live predictions. False keeps
        # predictions - and therefore trial-acceptance decisions - on the stable
        # base_model, so a co-adapted model never grades its own trials.
        self.apply_coadapt_predictions = apply_coadapt_predictions
        self.is_running = True

        self.baseline_buffer = []
        self.trial_buffer = []
        self.channel_names = []
        self.current_trial_index = "Unknown"
        self.current_target_class = 0  # Dynamically set by presentation script

        self.base_model = None
        self.coadapt_model = None

        # Same processing engine as the offline pipeline, applied to the live buffer.
        self.preprocessor = SignalPreprocessor(
            log_callback=lambda m: self.log_signal.emit("INFO", m),
            warning_callback=lambda m: self.log_signal.emit("WARNING", m)
        )

    def run(self):
        """
        Thread entry point: loads the model(s), waits for the LSL streams,
        records the synchronized baseline, then runs the marker-driven trial
        loop until stopped or a 'run_offset' marker arrives.
        """
        try:
            self.base_model = joblib.load(self.model_path)
            self.log_signal.emit("SUCCESS", f"Loaded production model: {os.path.basename(self.model_path)}")

            # A model fit on a CSV that still carried its saved index expects an
            # 'Unnamed: N' feature the live stream never provides - flag it now
            # rather than failing on every trial's prediction.
            leaked = find_index_columns(getattr(self.base_model, 'feature_names_in_', []))
            if leaked:
                self.log_signal.emit(
                    "WARNING",
                    f"Model was trained with leaked CSV index column(s) as features: {leaked}. "
                    f"Live predictions will fail on a feature-name mismatch - retrain it in the "
                    f"Decoder Training tab (index columns are now stripped on load)."
                )

            if self.coadaptation_enabled:
                if not self.coadapt_base_data_path or not os.path.exists(self.coadapt_base_data_path):
                    raise FileNotFoundError(
                        "Co-adaptation is enabled but no valid base training data CSV was provided. "
                        "Select a file with the same feature columns your pipeline produces, plus a 'target' column."
                    )

                base_df = read_csv(self.coadapt_base_data_path, log_callback=lambda m: self.log_signal.emit("WARNING", m))
                if 'target' not in base_df.columns:
                    raise ValueError("Base training data file must contain a 'target' column.")

                drop_cols = [c for c in ('target', 'run') if c in base_df.columns]
                self.X_coadapt = base_df.drop(columns=drop_cols).reset_index(drop=True)
                self.y_coadapt = base_df['target'].tolist()

                self.coadapt_model = self.base_model
                self.log_signal.emit(
                    "SUCCESS",
                    f"Co-adaptation decoder '{self.coadapt_decoder_key}' initialized fresh and fit on "
                    f"{len(self.X_coadapt)} base samples from {os.path.basename(self.coadapt_base_data_path)}."
                )
        except Exception as e:
            self.log_signal.emit("ERROR", f"Failed to load model / initialize co-adaptation: {str(e)}")
            return

        self.status_signal.emit("Resolving Streams...")

        try:
            data_streams = self._resolve_stream_with_retry("type", self.sig_type, f"{self.sig_type}/signal")
            if data_streams is None:
                self.status_signal.emit("Stopped")
                return  # Stopped by the user while waiting - not an error.

            marker_streams = self._resolve_stream_with_retry("name", self.marker_name, "Marker")
            if marker_streams is None:
                self.status_signal.emit("Stopped")
                return  # Stopped by the user while waiting - not an error.

            data_inlet = StreamInlet(data_streams[0])
            marker_inlet = StreamInlet(marker_streams[0])

            # Channel labels let the preprocessor pair wavelengths (SCI/SQA/MBLL)
            # and must match the decoder's feature names. Read them from
            # inlet.info(): the resolve_byprop() record lacks the outlet's full
            # XML description, so it would yield generic 'Ch-i' names.
            self.channel_names = resolve_channel_names(data_inlet.info())

            info = StreamInfo('Neurofeedback_Out', 'Markers', 2, 0, 'string', 'feedback_tx_01')
            feedback_outlet = StreamOutlet(info)

            self.log_signal.emit("SUCCESS", "LSL network locked. Awaiting 'baseline_start' synchronization...")
            self.status_signal.emit("Waiting for Presentation Sync")

            # Phase 1: synchronized baseline (between baseline_start/baseline_end markers).
            in_baseline = False
            baseline_start_seen = False
            last_progress_log = time.time()
            last_data_seen = None

            while self.is_running:
                m_sample, _ = marker_inlet.pull_sample(timeout=0.0)
                if m_sample:
                    if m_sample[0] == "baseline_start":
                        in_baseline = True
                        baseline_start_seen = True
                        last_data_seen = time.time()
                        self.status_signal.emit("Recording Baseline (Sync Active)...")
                        self.log_signal.emit("INFO", "Baseline synchronization marker caught. Recording...")
                    elif m_sample[0] == "baseline_end":
                        in_baseline = False
                        n_frames = len(self.baseline_buffer)

                        if not baseline_start_seen:
                            self.log_signal.emit(
                                "ERROR",
                                "'baseline_end' marker received, but 'baseline_start' was never detected "
                                "(0 frames buffered). The presentation script most likely fired "
                                "'baseline_start' before this engine finished connecting to the LSL "
                                "streams. Start the Neurofeedback Loop first and wait for the "
                                "'LSL network locked' message above before launching the presentation "
                                "paradigm - and make sure it has a few seconds of grace/countdown "
                                "before it sends 'baseline_start'."
                            )
                        elif n_frames == 0:
                            self.log_signal.emit(
                                "WARNING",
                                "Baseline window completed but 0 NIRS frames were buffered, even though "
                                "'baseline_start' was detected. Verify the data stream is actively "
                                "publishing samples (check it in the Online Streaming Monitor tab)."
                            )
                        else:
                            self.log_signal.emit("SUCCESS", f"Baseline locked ({n_frames} frames buffered).")
                        break

                d_sample, _ = data_inlet.pull_sample(timeout=0.0)
                if d_sample and in_baseline:
                    self.baseline_buffer.append(d_sample)
                    last_data_seen = time.time()
                    if time.time() - last_progress_log > 1.0:
                        self.status_signal.emit(f"Recording Baseline... ({len(self.baseline_buffer)} frames)")
                        last_progress_log = time.time()
                elif in_baseline and last_data_seen and (time.time() - last_data_seen) > 3.0:
                    # No NIRS samples arriving for a while during an active baseline window -
                    # surface this immediately instead of only finding out at baseline_end.
                    self.log_signal.emit("WARNING", "No NIRS samples received in the last 3s during baseline. Check the data stream.")
                    last_data_seen = time.time()  # Rate-limits this warning to once per 3s

                time.sleep(0.001)  # Yield briefly so this poll loop doesn't peg a CPU core

            if not self.is_running: return
            self.status_signal.emit("Active (Awaiting Trial)")

            # Phase 2: marker-driven trial loop.
            in_trial = False

            while self.is_running:
                m_sample, _ = marker_inlet.pull_sample(timeout=0.0)
                if m_sample:
                    marker_str, marker_val = m_sample[0], m_sample[1]

                    if marker_str == "trial_index":
                        self.current_trial_index = marker_val

                    elif marker_str == "target_class":
                        self.current_target_class = int(marker_val)
                        self.log_signal.emit("INFO", f"Target Class set to {self.current_target_class} for current trial.")

                    elif marker_str == "trial_onset":
                        in_trial = True
                        self.trial_buffer.clear()
                        self.status_signal.emit(f"Collecting Trial {self.current_trial_index}...")

                    elif marker_str == "request_feedback":
                        if in_trial and len(self.trial_buffer) > 0:
                            self.status_signal.emit("Processing Feedback...")
                            self._process_and_predict(feedback_outlet)
                        else:
                            self.log_signal.emit("WARNING", "Feedback requested but trial buffer is empty.")

                    elif marker_str == "trial_offset":
                        in_trial = False
                        self.status_signal.emit("Active (Awaiting Trial)")

                    elif marker_str == "run_offset":
                        self.log_signal.emit("SYSTEM", "Run completion detected. Halting worker.")
                        break

                d_sample, _ = data_inlet.pull_sample(timeout=0.0)
                if d_sample and in_trial:
                    self.trial_buffer.append(d_sample)

                time.sleep(0.001)  # Yield briefly so this poll loop doesn't peg a CPU core

        except Exception as e:
            self.log_signal.emit("ERROR", f"Worker Exception: {str(e)}")
            self.status_signal.emit("Error/Halted")

    def _process_and_predict(self, outlet):
        """
        Preprocesses Baseline+Trial, predicts the target-class probability on
        the trial portion, pushes it to LSL and, if co-adaptation is enabled and
        the probability exceeds the acceptance criterion, refits and saves the
        co-adaptive decoder.

        Only the first row of predict_proba() is used, so the pipeline is
        expected to reduce the trial to a single feature vector.

        Args:
            outlet: StreamOutlet that receives ["probability", <value>] samples.
        """
        try:
            # Prepend the baseline so filtering/detrending has enough samples near
            # trial onset, then slice back to the trial portion. If a step shrinks
            # the output below baseline_len (e.g. averaging), all of it is kept.
            full_buffer =  pd.concat([pd.DataFrame(self.baseline_buffer,columns=self.channel_names),pd.DataFrame(self.trial_buffer,columns=self.channel_names)])
            baseline_len = len(self.baseline_buffer)

            processed = self.preprocessor.run_steps(full_buffer, self.pipeline_steps, self.preproc_params, excluded_cols=[])
            trial_processed = processed.iloc[baseline_len:] if len(processed) >= baseline_len else processed

            if trial_processed is None or trial_processed.empty:
                self.log_signal.emit("WARNING", "Preprocessed trial buffer produced no usable features.")
                return

            # coadapt_model is the base model until the first accepted refit.
            if self.coadaptation_enabled and self.apply_coadapt_predictions:
                probabilities = self.coadapt_model.predict_proba(trial_processed)[0]
            else:
                probabilities = self.base_model.predict_proba(trial_processed)[0]

            if self.current_target_class < len(probabilities):
                target_prob = probabilities[self.current_target_class]
            else:
                target_prob = 0.0  # Fallback if class index is out of bounds

            outlet.push_sample(["probability", str(target_prob)])

            self.prediction_signal.emit(self.current_target_class, target_prob)
            self.log_signal.emit("SUCCESS", f"[Trial {self.current_trial_index}] Probability of Class {self.current_target_class}: {target_prob:.2f}")

            # Co-adaptation: accept confident trials and refit from scratch.
            coadapt_acceptance_criteria = getattr(config, 'COADAPTATION_ACCEPTANCE_CRITERIA', 0.85)
            if self.coadaptation_enabled and target_prob > coadapt_acceptance_criteria:

                self.log_signal.emit("SYSTEM", f"Probability ({target_prob:.2f}) meets co-adaptation criteria. Updating auxiliary decoder...")

                combined = pd.concat([self.X_coadapt, trial_processed], axis=0, ignore_index=True)
                if combined.isnull().values.any() and not self.X_coadapt.empty:
                    self.log_signal.emit(
                        "WARNING",
                        "Feature column mismatch detected between the base training data and the live "
                        "pipeline output - co-adaptation results may be unreliable. Make sure the base "
                        "CSV's feature columns match what this pipeline configuration produces."
                    )

                self.X_coadapt = combined
                trial_labels = [self.current_target_class] * len(trial_processed)
                self.y_coadapt = self.y_coadapt + trial_labels
                if len(set(self.y_coadapt)) < 2:
                    self.log_signal.emit("WARNING", "Co-adaptation fit skipped: at least two distinct target classes are needed.")
                else:
                    save_path = self.model_path.replace(".pkl", "_coadapted.pkl")
                    self.coadapt_model = decoder_selection(selected_method=self.coadapt_decoder_key)
                    self.coadapt_model.fit(self.X_coadapt, self.y_coadapt)
                    joblib.dump(self.coadapt_model, save_path)
                    self.log_signal.emit(
                        "SUCCESS",
                        f"Co-adaptive decoder refit on {len(trial_labels)} samples and saved to: {os.path.basename(save_path)}"
                    )

        except Exception as e:
            self.log_signal.emit("ERROR", f"Prediction Math Error on Trial {self.current_trial_index}: {str(e)}")

    def stop(self):
        """Signals the worker loops to exit and blocks until the thread finishes."""
        self.is_running = False
        self.wait()

    def _resolve_stream_with_retry(self, prop, value, label, poll_timeout=2.0, log_every=5):
        """
        Polls resolve_byprop until the requested stream appears or the worker
        is stopped, so the engine can be started before the data and marker
        streams exist, in any order.

        Args:
            prop: LSL property to match ("type" or "name").
            value: Required value of that property.
            label: Human-readable stream label used in status/log messages.
            poll_timeout: Seconds per resolve_byprop attempt.
            log_every: Log a "still waiting" message every N attempts.

        Returns:
            The resolved stream list, or None if the worker was stopped while waiting.
        """
        attempt = 0
        while self.is_running:
            streams = resolve_byprop(prop, value, timeout=poll_timeout)
            if streams:
                if attempt > 0:
                    self.log_signal.emit("SUCCESS", f"{label} stream '{value}' found after waiting.")
                return streams
            attempt += 1
            self.status_signal.emit(f"Waiting for {label} stream ('{value}')...")
            if attempt == 1 or attempt % log_every == 0:
                self.log_signal.emit(
                    "INFO",
                    f"Still waiting for {label} stream '{value}' on the network "
                    f"(attempt {attempt}) - start it whenever you're ready."
                )
        return None


class NeurofeedbackTab(QWidget):
    """Command center for real-time model deployment and LSL neurofeedback casting."""
    def __init__(self, parent=None):
        """Initializes tab state and builds the UI."""
        super().__init__(parent)
        self.worker = None
        self.loaded_exclude_channels = []  # Channel names loaded from a saved list (e.g. offline SCI/SQA rejections)
        self.init_ui()

    def init_ui(self):
        """Builds the scrollable configuration/control panel and the runtime log panel."""
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setStyleSheet("background-color: transparent;")

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Stream type the worker resolves via resolve_byprop("type", ...).
        source_group = QGroupBox("Signal Source")
        source_layout = QGridLayout(source_group)
        source_layout.addWidget(QLabel("Target Signal Stream Type:"), 0, 0)
        self.combo_sig_type = QComboBox()
        self.combo_sig_type.addItems(["NIRS", "EEG", "MEG"])
        source_layout.addWidget(self.combo_sig_type, 0, 1)
        left_layout.addWidget(source_group)

        cfg_group = QGroupBox("Neurofeedback Decoder Configuration")
        cfg_layout = QGridLayout(cfg_group)

        cfg_layout.addWidget(QLabel("Deployment Model (.pkl):"), 0, 0)
        self.ent_model_path = QLineEdit()
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_model)
        cfg_layout.addWidget(self.ent_model_path, 0, 1)
        cfg_layout.addWidget(btn_browse, 0, 2)

        self.chk_coadapt = QCheckBox("Enable Co-adaptation (Generates distinct decoder *_coadapted.pkl)")
        self.chk_coadapt.setChecked(True)
        cfg_layout.addWidget(self.chk_coadapt, 1, 0, 1, 3)

        cfg_layout.addWidget(QLabel("Co-adaptation Base Decoder:"), 2, 0)
        self.combo_coadapt_decoder = QComboBox()
        self.combo_coadapt_decoder.addItems(list(DECODER_REGISTRY.keys()))
        default_decoder = "logisticregression" if "logisticregression"in DECODER_REGISTRY else list(DECODER_REGISTRY.keys())[0]
        self.combo_coadapt_decoder.setCurrentText(default_decoder)
        cfg_layout.addWidget(self.combo_coadapt_decoder, 2, 1, 1, 2)

        cfg_layout.addWidget(QLabel("Co-adaptation Base Training Data (CSV):"), 3, 0)
        self.ent_coadapt_data_path = QLineEdit()
        self.ent_coadapt_data_path.setToolTip(
            "A preprocessed CSV with the same feature columns this pipeline configuration produces, "
            "plus a 'target' column. Seeds the co-adaptation decoder's training set instead of starting empty."
        )
        btn_browse_coadapt_data = QPushButton("Browse")
        btn_browse_coadapt_data.clicked.connect(self.browse_coadapt_data)
        cfg_layout.addWidget(self.ent_coadapt_data_path, 3, 1)
        cfg_layout.addWidget(btn_browse_coadapt_data, 3, 2)

        self.chk_apply_coadapt = QCheckBox("Use Co-adapted Decoder for Live Predictions")
        self.chk_apply_coadapt.setChecked(True)
        self.chk_apply_coadapt.setToolTip(
            "Checked (default): once enough accepted trials accumulate, live predictions switch to "
            "the co-adapted decoder, same as before.\n"
            "Unchecked: predictions stay on the loaded base model for the whole session, but "
            "co-adaptation still trains/refits in the background and saves '_coadapted.pkl' as "
            "usual - so you can review or deploy it afterward without it affecting this session's "
            "live feedback (and trial-acceptance decisions stay based on the stable base model's "
            "confidence, rather than a co-adapted model grading its own trials)."
        )
        cfg_layout.addWidget(self.chk_apply_coadapt, 4, 0, 1, 3)

        self.chk_coadapt.toggled.connect(self._on_coadapt_toggled)
        self._on_coadapt_toggled(self.chk_coadapt.isChecked())

        left_layout.addWidget(cfg_group)

        # Channel exclusion list consumed by the "Manual Channel Filter" pipeline step.
        chan_filter_group = QGroupBox("Channel Filter List")
        chan_filter_layout = QGridLayout(chan_filter_group)
        chan_filter_layout.setSpacing(8)

        chan_filter_layout.addWidget(QLabel("Loaded Exclusion List:"), 0, 0)
        self.ent_channel_filter_path = QLineEdit()
        self.ent_channel_filter_path.setReadOnly(True)
        self.ent_channel_filter_path.setPlaceholderText("No list loaded - live SCI/SQA will decide rejections per-trial instead.")
        chan_filter_layout.addWidget(self.ent_channel_filter_path, 0, 1)

        btn_browse_channel_filter = QPushButton("Browse")
        btn_browse_channel_filter.clicked.connect(self.browse_channel_filter_list)
        chan_filter_layout.addWidget(btn_browse_channel_filter, 0, 2)

        self.lbl_channel_filter_count = QLabel("0 channel(s) loaded")
        self.lbl_channel_filter_count.setStyleSheet("color: #b0b0b0; font-size: 10px;")
        chan_filter_layout.addWidget(self.lbl_channel_filter_count, 1, 0, 1, 3)

        note = QLabel(
            "Load a channel list exported from the Offline tab's 'Save Rejected Channels...' "
            "button to exclude the same channels here, instead of relying on SCI/SQA to "
            "re-decide rejections on each short live buffer (which can change the feature "
            "set from trial to trial). Enable the 'Manual Channel Filter' step below to apply it."
        )
        note.setStyleSheet("color: #999999; font-size: 10px;")
        note.setWordWrap(True)
        chan_filter_layout.addWidget(note, 2, 0, 1, 3)

        left_layout.addWidget(chan_filter_group)

        defaults = config.DEFAULT_PREPROC_CONFIG
        preproc_section = CollapsibleSection("Real-Time Preprocessing Pipeline")
        preproc_section.set_expanded(False)
        preproc_layout = QVBoxLayout()
        preproc_layout.setSpacing(8)

        pipeline_steps_spec = [
            ("channel_filter", "Manual Channel Filter",
             "Drops the channels loaded above by name. A fixed exclusion list keeps the "
             "feature set consistent trial-to-trial, unlike live SCI/SQA rejection.",
             False),
            ("sci", "Scalp Coupling Index (SCI)",
             "Drops poorly-coupled optode pairs. Caution: changes the channel count "
             "trial-to-trial, which can break a model trained on a fixed feature set.",
             False),
            ("sqa", "Signal Quality Assessment (SQA)",
             "Drops noisy channels via Coefficient of Variation. Same channel-count "
             "caution as SCI applies.",
             False),
            ("mbll", "Convert to HbO/HbR (MBLL)",
             "Transforms raw wavelength intensities into concentrations. Enable only "
             "if the deployed model was trained on MBLL-converted features.",
             False),
            ("filter", "Apply Butterworth Bandpass Filter",
             "Attenuates low-frequency drift and physiological noise across the "
             "Baseline+Trial buffer.",
             False),
            ("detrend", "Apply Linear Detrending",
             "Removes linear drift. Automatically skipped if the buffer is too short "
             "for the configured batch count.",
             False),
            ("scaling", "Signal Amplitude Scaling",
             "Applies StandardScaler/MinMaxScaler across the current buffer.",
             False),
            ("averaging (feature_vector)", "Compute Trial-Level Averaging", 
             "Single-row mean feature vector (mirrors the offline 'averaging' step, just for one live trial)",
            False),
        ]
        self.step_table = PipelineStepTable(pipeline_steps_spec, show_checkpoint=False, row_height=56)
        preproc_layout.addWidget(self.step_table)

        param_grid = QGridLayout()
        param_grid.setSpacing(8)
        self.txt_cv_thresh = QLineEdit(str(defaults["sqa_cv_threshold"]))
        self.txt_sci_thresh = QLineEdit(str(defaults["sci_correlation_min"]))
        self.txt_low_cut = QLineEdit(str(defaults["filter_low_band"]))
        self.txt_high_cut = QLineEdit(str(defaults["filter_high_band"]))
        self.txt_fs = QLineEdit(str(defaults["sampling_rate"]))
        self.combo_scaler_type = QComboBox()
        self.combo_scaler_type.addItems(["standard", "minmax"])
        self.combo_scaler_type.setCurrentText(defaults.get("scaler_type", "standard"))

        param_grid.addWidget(QLabel("SQA CV Limit (%):"), 0, 0); param_grid.addWidget(self.txt_cv_thresh, 0, 1)
        param_grid.addWidget(QLabel("SCI Correlation Min:"), 0, 2); param_grid.addWidget(self.txt_sci_thresh, 0, 3)
        param_grid.addWidget(QLabel("Filter Low Cut (Hz):"), 1, 0); param_grid.addWidget(self.txt_low_cut, 1, 1)
        param_grid.addWidget(QLabel("Filter High Cut (Hz):"), 1, 2); param_grid.addWidget(self.txt_high_cut, 1, 3)
        param_grid.addWidget(QLabel("Sampling Frequency (Fs):"), 2, 0); param_grid.addWidget(self.txt_fs, 2, 1)
        param_grid.addWidget(QLabel("Scaler Type:"), 2, 2); param_grid.addWidget(self.combo_scaler_type, 2, 3)
        preproc_layout.addLayout(param_grid)

        preproc_section.set_content_layout(preproc_layout)
        left_layout.addWidget(preproc_section)

        ctrl_group = QGroupBox("Execution Controls")
        ctrl_layout = QVBoxLayout(ctrl_group)
        self.btn_start = QPushButton("Start Neurofeedback Loop")
        self.btn_start.setStyleSheet("background-color: #2e7d32; color: white; padding: 10px;")
        self.btn_start.clicked.connect(self.start_engine)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_engine)
        self.lbl_status = QLabel("Engine Status: Offline")
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.lbl_status)
        left_layout.addWidget(ctrl_group)

        vis_group = QGroupBox("Live Prediction Values")
        vis_layout = QVBoxLayout(vis_group)
        self.lbl_pred_class = QLabel("Target Class: --")
        vis_layout.addWidget(self.lbl_pred_class)
        vis_layout.addWidget(QLabel("Target Class Probability:"))
        self.bar_confidence = QProgressBar()
        self.bar_confidence.setRange(0, 100)
        vis_layout.addWidget(self.bar_confidence)
        left_layout.addWidget(vis_group)

        left_layout.addStretch()
        left_scroll.setWidget(left_panel)
        splitter.addWidget(left_scroll)

        right_panel = QGroupBox("Engine Runtime Logs")
        right_layout = QVBoxLayout(right_panel)
        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)
        self.txt_logs.setStyleSheet("background-color: #0c0c0c; color: #00ff66; font-family: monospace;")
        right_layout.addWidget(self.txt_logs)
        splitter.addWidget(right_panel)

        splitter.setSizes([480, 550])
        main_layout.addWidget(splitter)

    def browse_model(self):
        """Opens a file dialog to pick the deployment model (.pkl)."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Compiled Model", "", "Pickle Files (*.pkl)")
        if file_path:
            self.ent_model_path.setText(file_path)

    def browse_coadapt_data(self):
        """Opens a file dialog to pick the co-adaptation base training CSV."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Base Training Data (CSV)", "", "CSV Files (*.csv)")
        if file_path:
            self.ent_coadapt_data_path.setText(file_path)

    def browse_channel_filter_list(self):
        """Loads a channel exclusion list (e.g. exported from the Offline tab) for the real-time 'Manual Channel Filter' step."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Channel Exclusion List", "", "Channel List Files (*.json *.txt *.csv);;All Files (*)"
        )
        if not file_path:
            return
        try:
            self.loaded_exclude_channels = load_channel_list(file_path)
            self.ent_channel_filter_path.setText(file_path)
            self.lbl_channel_filter_count.setText(f"{len(self.loaded_exclude_channels)} channel(s) loaded")
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Could not parse channel list file: {str(e)}")

    def _on_coadapt_toggled(self, checked):
        """Enables or disables the co-adaptation-specific controls."""
        self.combo_coadapt_decoder.setEnabled(checked)
        self.ent_coadapt_data_path.setEnabled(checked)
        self.chk_apply_coadapt.setEnabled(checked)

    def log_msg(self, level, msg):
        """Appends a "[LEVEL] message" line to the runtime log panel."""
        self.txt_logs.append(f"[{level}] {msg}")

    def compile_preproc_params(self):
        """Builds the same-shaped params dict the offline pipeline uses, from UI fields + config defaults."""
        defaults = config.DEFAULT_PREPROC_CONFIG
        return {
            'cv_threshold': float(self.txt_cv_thresh.text()),
            'sci_threshold': float(self.txt_sci_thresh.text()),
            'low_cut': float(self.txt_low_cut.text()),
            'high_cut': float(self.txt_high_cut.text()),
            'fs': float(self.txt_fs.text()),
            'butterworth_order': defaults.get('butterworth_order', 5),
            'detrend_batches': defaults.get('detrend_batches', 10),
            'mbll_wavelengths': defaults.get('mbll_wavelengths', [735, 850]),
            'extinction_coefficients': defaults.get('extinction_coefficients', {}),
            'scaler_type': self.combo_scaler_type.currentText(),
            'manual_exclude_channels': self.loaded_exclude_channels,
        }

    def start_engine(self):
        """Validates inputs, then creates, wires up and starts a NeurofeedbackWorker."""
        model_path = self.ent_model_path.text()
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Model Not Found", "Please select a valid trained model (.pkl) file first.")
            return

        coadapt_data_path = ""
        if self.chk_coadapt.isChecked():
            coadapt_data_path = self.ent_coadapt_data_path.text()
            if not coadapt_data_path or not os.path.exists(coadapt_data_path):
                QMessageBox.warning(
                    self, "Missing Base Training Data",
                    "Co-adaptation is enabled - please select a valid base training data CSV file first."
                )
                return

        try:
            preproc_params = self.compile_preproc_params()
        except ValueError:
            QMessageBox.warning(self, "Invalid Parameters", "Preprocessing parameter fields must contain valid numbers.")
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.txt_logs.clear()

        self.worker = NeurofeedbackWorker(
            model_path=model_path,
            sig_type=self.combo_sig_type.currentText(),
            marker_name="OpenSesame_Markers",
            coadaptation_enabled=self.chk_coadapt.isChecked(),
            pipeline_steps=self.step_table.compiled_steps(),
            preproc_params=preproc_params,
            coadapt_decoder_key=self.combo_coadapt_decoder.currentText(),
            coadapt_base_data_path=coadapt_data_path,
            apply_coadapt_predictions=self.chk_apply_coadapt.isChecked()
        )
        self.worker.log_signal.connect(self.log_msg)
        self.worker.status_signal.connect(lambda s: self.lbl_status.setText(f"Status: {s}"))
        self.worker.prediction_signal.connect(self.update_telemetry)
        # QThread.finished fires however run() ends (error, failed model load,
        # run_offset, or Stop), so the buttons are always reset.
        self.worker.finished.connect(lambda w=self.worker: self._on_worker_finished(w))
        self.worker.start()

    def update_telemetry(self, target_class, prob):
        """Shows the latest target class and its probability (as a percentage)."""
        self.lbl_pred_class.setText(f"Target Class: {target_class}")
        self.bar_confidence.setValue(int(prob * 100))

    def _on_worker_finished(self, worker):
        """Re-enables Start once the given worker's thread has ended, ignoring stale workers."""
        if worker is not self.worker:
            return
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def stop_engine(self):
        """Stops the running worker (if any) and resets the start/stop buttons."""
        if self.worker:
            self.worker.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)