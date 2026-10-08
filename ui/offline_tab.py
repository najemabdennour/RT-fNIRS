"""Offline analysis tab: merges multi-run sessions and runs the reorderable offline preprocessing pipeline."""
import os
from datetime import datetime
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
    QLineEdit, QPushButton, QCheckBox, QGroupBox, QTextEdit, \
    QFileDialog, QMessageBox, QApplication, QComboBox, QProgressDialog,
    QScrollArea, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

import config
from core.offline_processor import OfflineDataPreprocessor, OfflineSessionCombiner
from core.channel_lists import save_channel_list, load_channel_list
from core.io_utils import read_csv
from ui.components import CollapsibleSection, PipelineStepTable


class OfflineAnalysisTab(QWidget):
    """Offline preprocessing workspace.

    Combines per-run recordings/behavior logs into session files, lets the user
    pick an experiment profile, reorder and toggle pipeline steps (with optional
    per-step checkpoint saves), and exports SCI/SQA-rejected channel lists.
    """
    def __init__(self, parent=None):
        """Builds the tab UI and initialises the rejected-channel cache."""
        super().__init__(parent)
        self.last_rejected_channels = []  # SCI/SQA rejections from the most recent pipeline run
        self.init_ui()

    def init_ui(self):
        """Builds the scrollable left control column and the right-hand log console."""
        master_layout = QHBoxLayout(self)
        master_layout.setContentsMargins(12, 12, 12, 12)
        master_layout.setSpacing(15)

        # Scroll area keeps the left column usable on low-resolution monitors.
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet("background-color: transparent;")
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: transparent;")
        left_column = QVBoxLayout(scroll_content)
        left_column.setContentsMargins(0, 0, 8, 0)
        left_column.setSpacing(14)
        
        defaults = config.DEFAULT_PREPROC_CONFIG

        # Panel 1: session run aggregator
        io_section = CollapsibleSection("1. Session Run Aggregator (Optional)")
        io_layout = QGridLayout()
        io_layout.setSpacing(8)

        self.ent_rec_dir = QLineEdit()
        self.ent_rec_dir.setPlaceholderText("Path to directory containing raw multi-run data...")
        btn_browse_rec = QPushButton("Browse")
        btn_browse_rec.clicked.connect(lambda: self.browse_directory(self.ent_rec_dir))

        self.ent_beh_dir = QLineEdit()
        self.ent_beh_dir.setPlaceholderText("Path to directory containing run behavioral logs...")
        btn_browse_beh = QPushButton("Browse")
        btn_browse_beh.clicked.connect(lambda: self.browse_directory(self.ent_beh_dir))

        self.ent_out_rec = QLineEdit()
        self.ent_out_rec.setPlaceholderText("Output folder path for aggregated run data...")
        btn_browse_out_rec = QPushButton("Browse")
        btn_browse_out_rec.clicked.connect(lambda: self.browse_directory(self.ent_out_rec))

        self.ent_out_beh = QLineEdit()
        self.ent_out_beh.setPlaceholderText("Output folder path for master behavioral matrices...")
        btn_browse_out_beh = QPushButton("Browse")
        btn_browse_out_beh.clicked.connect(lambda: self.browse_directory(self.ent_out_beh))

        self.btn_run_merge = QPushButton("Consolidate & Merge Session Runs")
        self.btn_run_merge.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_run_merge.clicked.connect(self.execute_session_merging_pipeline)

        io_layout.addWidget(QLabel("Raw Signals Dir:"), 0, 0)
        io_layout.addWidget(self.ent_rec_dir, 0, 1)
        io_layout.addWidget(btn_browse_rec, 0, 2)
        io_layout.addWidget(QLabel("Behavior Logs Dir:"), 1, 0)
        io_layout.addWidget(self.ent_beh_dir, 1, 1)
        io_layout.addWidget(btn_browse_beh, 1, 2)
        io_layout.addWidget(QLabel("Output Signals Dir:"), 2, 0)
        io_layout.addWidget(self.ent_out_rec, 2, 1)
        io_layout.addWidget(btn_browse_out_rec, 2, 2)
        io_layout.addWidget(QLabel("Output Behavior Dir:"), 3, 0)
        io_layout.addWidget(self.ent_out_beh, 3, 1)
        io_layout.addWidget(btn_browse_out_beh, 3, 2)
        io_layout.addWidget(self.btn_run_merge, 4, 0, 1, 3)
        io_section.set_content_layout(io_layout)
        left_column.addWidget(io_section)

        # Panel 2: experiment profile and input/output files
        target_section = CollapsibleSection("2. Pipeline Inputs / Paradigm Selector")
        target_section.set_expanded(True)
        target_layout = QGridLayout()
        target_layout.setSpacing(8)

        target_layout.addWidget(QLabel("Active Experiment:"), 0, 0)
        self.combo_exp = QComboBox()

        # The selected profile is read from this combo when needed and passed
        # explicitly to the processing code (no global state).
        self.combo_exp.addItems(list(config.EXPERIMENT_PROFILES.keys()))
        self.combo_exp.setCurrentText(config.DEFAULT_EXPERIMENT)
        self.combo_exp.currentTextChanged.connect(self.on_experiment_selection_changed)
        target_layout.addWidget(self.combo_exp, 0, 1, 1, 2)

        target_layout.addWidget(QLabel("New Registry Name:"), 1, 0)
        self.txt_new_exp_name = QLineEdit()
        self.txt_new_exp_name.setPlaceholderText("e.g., AudioNBack")
        target_layout.addWidget(self.txt_new_exp_name, 1, 1)
        
        btn_register_exp = QPushButton("Register Profile")
        btn_register_exp.setStyleSheet("font-weight: bold; background-color: #1976d2; color: white;")
        btn_register_exp.clicked.connect(self.register_custom_experiment_profile)
        target_layout.addWidget(btn_register_exp, 1, 2)

        target_layout.addWidget(QLabel("Define Columns:"), 2, 0)
        self.txt_new_exp_cols = QLineEdit()
        self.txt_new_exp_cols.setPlaceholderText("trial_index, condition, sequence (comma-separated)")
        target_layout.addWidget(self.txt_new_exp_cols, 2, 1, 1, 2)

        self.ent_target_signal_file = QLineEdit()
        self.ent_target_signal_file.setPlaceholderText("Select consolidated session signal CSV file...")
        btn_browse_target_sig = QPushButton("Browse File")
        btn_browse_target_sig.clicked.connect(lambda: self.browse_file(self.ent_target_signal_file, "CSV Files (*.csv)"))

        self.ent_target_behavior_file = QLineEdit()
        self.ent_target_behavior_file.setPlaceholderText("Select consolidated behavior CSV file (Optional for non-WM)...")
        btn_browse_target_beh = QPushButton("Browse File")
        btn_browse_target_beh.clicked.connect(lambda: self.browse_file(self.ent_target_behavior_file, "CSV Files (*.csv)"))

        self.ent_target_output_dir = QLineEdit()
        self.ent_target_output_dir.setPlaceholderText("Select directory to write preprocessing outcome arrays...")
        btn_browse_target_out = QPushButton("Browse Dir")
        btn_browse_target_out.clicked.connect(lambda: self.browse_directory(self.ent_target_output_dir))

        target_layout.addWidget(QLabel("Target Signal File:"), 3, 0)
        target_layout.addWidget(self.ent_target_signal_file, 3, 1)
        target_layout.addWidget(btn_browse_target_sig, 3, 2)
        target_layout.addWidget(QLabel("Target Behavior File:"), 4, 0)
        target_layout.addWidget(self.ent_target_behavior_file, 4, 1)
        target_layout.addWidget(btn_browse_target_beh, 4, 2)
        target_layout.addWidget(QLabel("Preproc Save Dir:"), 5, 0)
        target_layout.addWidget(self.ent_target_output_dir, 5, 1)
        target_layout.addWidget(btn_browse_target_out, 5, 2)
        target_section.set_content_layout(target_layout)
        left_column.addWidget(target_section)

        # Panel 3: reorderable pipeline steps, channel filter and parameters
        prep_section = CollapsibleSection("3. Reorder Preprocessing Pipeline Sequence / Save Checkpoints")
        prep_section.set_expanded(True)
        prep_main_layout = QVBoxLayout()
        prep_main_layout.setSpacing(12)

        default_steps = [
            ("channel_filter", "Manual Channel Filter", "Drops/keeps channels by exact name from a typed or loaded list - e.g. re-apply a previous run's SCI/SQA rejections", False),
            ("sci", "Scalp Coupling Index (SCI)", "Validates optode contact quality via cross-wavelength correlation", defaults["enable_sci"]),
            ("sqa", "Signal Quality Assessment (SQA)", "Filters individual noisy channels via standard Coefficient of Variation (CV)", defaults["enable_sqa"]),
            ("mbll", "Convert to HbO/HbR (MBLL)", "Transforms raw wavelength intensity measurements into concentrations", defaults["enable_mbll"]),
            ("epoch_prep", "Epoch Slicing & Behavioral Setup", "Segments continuous stream indices into aligned time windows", True),
            ("filter", "Apply Butterworth Bandpass Filter", "Attenuates low-frequency baseline drifts and systemic physiological noise", defaults["butterworth_filter"]),
            ("detrend", "Apply Linear Detrending Passes", "Eliminates continuous, linear drift anomalies across blocks", defaults["detrend"]),
            ("scaling", "Signal Amplitude Scaling", "Apply StandardScaler or MinMaxScaler features across clean data channels",  defaults["scaling"]),
            ("averaging", "Compute Trial-Level Averaging", "Groups segments by target values to compute run response averages", defaults["averaging"]),
            ("block_avg", "Compute Block Condition Averaging", "Synthesizes macro block matrices aggregated relative to tasks", defaults["block_avg"])
        ]

        self.step_table = PipelineStepTable(default_steps, show_checkpoint=True, row_height=56)
        prep_main_layout.addWidget(self.step_table)

        # Manual Channel Filter List feeds the "channel_filter" step. After a run
        # with SCI/SQA, the rejected channels can be exported and re-applied here
        # or in the Neurofeedback tab, so the live pipeline excludes the same
        # channels instead of re-deciding rejections on a short live buffer.
        chan_filter_group = QGroupBox("Manual Channel Filter List (used by the 'Manual Channel Filter' step)")
        chan_filter_layout = QGridLayout(chan_filter_group)
        chan_filter_layout.setSpacing(8)

        self.txt_channel_filter_list = QLineEdit()
        self.txt_channel_filter_list.setPlaceholderText("Comma-separated channel names to exclude, e.g. S(1)_D(2), S(3)_D(4) ...")
        chan_filter_layout.addWidget(QLabel("Exclude Channels:"), 0, 0)
        chan_filter_layout.addWidget(self.txt_channel_filter_list, 0, 1, 1, 2)

        btn_load_channel_list = QPushButton("Load List File...")
        btn_load_channel_list.clicked.connect(self.browse_and_load_channel_list)
        chan_filter_layout.addWidget(btn_load_channel_list, 0, 3)

        self.lbl_rejected_summary = QLabel("Last run rejected: 0 channel(s) via SCI/SQA.")
        self.lbl_rejected_summary.setStyleSheet("color: #b0b0b0; font-size: 10px;")
        chan_filter_layout.addWidget(self.lbl_rejected_summary, 1, 0, 1, 3)

        self.btn_save_rejected = QPushButton("Save Rejected Channels...")
        self.btn_save_rejected.setEnabled(False)
        self.btn_save_rejected.setStyleSheet("background-color: #383838; color: white;")
        self.btn_save_rejected.clicked.connect(self.save_rejected_channels)
        chan_filter_layout.addWidget(self.btn_save_rejected, 1, 3)

        prep_main_layout.addWidget(chan_filter_group)

        param_grid = QGridLayout()
        param_grid.setSpacing(8)
        
        self.txt_cv_thresh = QLineEdit(str(defaults["sqa_cv_threshold"]))
        self.txt_sci_thresh = QLineEdit(str(defaults["sci_correlation_min"]))
        self.txt_low_cut = QLineEdit(str(defaults["filter_low_band"]))
        self.txt_high_cut = QLineEdit(str(defaults["filter_high_band"]))
        self.txt_onset_buffer = QLineEdit(str(defaults["peak_onset_time_buffer"]))
        self.txt_offset_buffer = QLineEdit(str(defaults["peak_offset_time_buffer"]))
        self.txt_fs = QLineEdit(str(defaults["sampling_rate"]))

        self.combo_scaler_type = QComboBox()
        self.combo_scaler_type.addItems(["standard", "minmax"])
        self.combo_scaler_type.setCurrentText("standard") 

        param_grid.addWidget(QLabel("SQA CV Limit (%):"), 0, 0); param_grid.addWidget(self.txt_cv_thresh, 0, 1)
        param_grid.addWidget(QLabel("SCI Correlation Min:"), 0, 2); param_grid.addWidget(self.txt_sci_thresh, 0, 3)
        param_grid.addWidget(QLabel("Filter Low Cut (Hz):"), 1, 0); param_grid.addWidget(self.txt_low_cut, 1, 1)
        param_grid.addWidget(QLabel("Filter High Cut (Hz):"), 1, 2); param_grid.addWidget(self.txt_high_cut, 1, 3)
        param_grid.addWidget(QLabel("Peak Onset Buffer (s):"), 2, 0); param_grid.addWidget(self.txt_onset_buffer, 2, 1)
        param_grid.addWidget(QLabel("Peak Offset Buffer (s):"), 2, 2); param_grid.addWidget(self.txt_offset_buffer, 2, 3)
        param_grid.addWidget(QLabel("Sampling Frequency (Fs):"), 3, 0); param_grid.addWidget(self.txt_fs, 3, 1)
        param_grid.addWidget(QLabel("Scaler SelectionProfile:"), 3, 2); param_grid.addWidget(self.combo_scaler_type, 3, 3)
        prep_main_layout.addLayout(param_grid)

        self.btn_run_preproc = QPushButton("Execute Sequential Pipeline Routine")
        self.btn_run_preproc.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 10px; font-size: 12px;")
        self.btn_run_preproc.clicked.connect(self.execute_preprocessing_only)
        prep_main_layout.addWidget(self.btn_run_preproc)
        prep_section.set_content_layout(prep_main_layout)
        left_column.addWidget(prep_section)

        scroll_area.setWidget(scroll_content)
        master_layout.addWidget(scroll_area, stretch=5)

        # Right column: log console
        log_group = QGroupBox("Pipeline Operational Matrix Log Terminal")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(8, 12, 8, 8)
        
        self.txt_console_logs = QTextEdit()
        self.txt_console_logs.setReadOnly(True)
        self.txt_console_logs.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                background-color: #0c0c0c;
                color: #00ff66;
                border: 2px solid #222222;
                border-radius: 6px;
                padding: 10px;
                line-height: 140%;
            }
        """)
        log_layout.addWidget(self.txt_console_logs)
        master_layout.addWidget(log_group, stretch=5)

        self.append_status_log("SYSTEM", f"Pipeline Environment Ready. Active experiment profile: '{self.combo_exp.currentText()}'.")

    def append_status_log(self, status_level, message):
        """Appends a timestamped, colour-coded line to the log console.

        Args:
            status_level: One of SYSTEM, SUCCESS, WARNING, ERROR, INFO (sets the colour).
            message: Text to log.
        """
        timestamp = datetime.now().strftime('%H:%M:%S')
        color_map = {"SYSTEM": "#0099ff", "SUCCESS": "#00ff66", "WARNING": "#ffaa00", "ERROR": "#ff3333", "INFO": "#ffffff"}
        color = color_map.get(status_level, "#ffffff")
        html_msg = f"<span style='color: #666666;'>[{timestamp}]</span> <span style='color: {color}; font-weight: bold;'>[{status_level}]</span> <span style='color: #dcdcdc;'>{message}</span>"
        self.txt_console_logs.append(html_msg)
        QApplication.processEvents()

    def browse_directory(self, target_lineedit):
        """Opens a directory picker and writes the chosen path into target_lineedit."""
        folder = QFileDialog.getExistingDirectory(self, "Select Directory Source")
        if folder: target_lineedit.setText(folder)

    def browse_file(self, target_lineedit, file_filter):
        """Opens a file picker with file_filter and writes the chosen path into target_lineedit."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV Resource File", "", file_filter)
        if file_path: target_lineedit.setText(file_path)

    def on_experiment_selection_changed(self, selected_paradigm):
        """Logs whether the newly selected profile requires a behavior log and its label column."""
        profile = config.get_experiment_profile(selected_paradigm)
        requirement = "MANDATORY" if profile['behavior_required'] else "OPTIONAL"
        target = profile['target_column'] or "none (target defaults to 1)"
        self.append_status_log(
            "SYSTEM",
            f"Active Paradigm changed to '{selected_paradigm}'. Behavioral log is {requirement}; label column: {target}."
        )

    def register_custom_experiment_profile(self):
        """Registers a new experiment profile from the name/columns fields and selects it.

        Spaces in the name become underscores; with no columns given the profile
        defaults to ["trial_index"].
        """
        name = self.txt_new_exp_name.text().strip().replace(" ", "_")
        columns_raw = self.txt_new_exp_cols.text().strip()
        
        if not name:
            QMessageBox.warning(self, "Validation Error", "Please declare a valid unique name profile label for the tracking layout register.")
            return
            
        parsed_cols = [col.strip() for col in columns_raw.split(",") if col.strip()]
        if not parsed_cols: parsed_cols = ["trial_index"]
        
        profile = config.register_experiment_profile(name, parsed_cols)

        self.combo_exp.blockSignals(True)
        self.combo_exp.clear()
        self.combo_exp.addItems(list(config.EXPERIMENT_PROFILES.keys()))
        self.combo_exp.setCurrentText(name)
        self.combo_exp.blockSignals(False)

        self.append_status_log(
            "SUCCESS",
            f"Registered new customized runtime paradigm architecture '{name}' mapped to array features: {parsed_cols} "
            f"(label column: {profile['target_column'] or 'none - target defaults to 1'})"
        )
        self.txt_new_exp_name.clear()
        self.txt_new_exp_cols.clear()

    def execute_session_merging_pipeline(self):
        """Combines per-run signal and behavior files into session files.

        On success, fills the target signal/behavior fields with the outputs and,
        if empty, sets the preprocessing save dir to the signal output folder.
        """
        try:
            rec_dir = self.ent_rec_dir.text()
            beh_dir = self.ent_beh_dir.text()
            if not rec_dir: raise ValueError("Recording source path configuration parameter cannot be null.")

            self.btn_run_merge.setEnabled(False)
            self.append_status_log("SYSTEM", "LAUNCHING SESSION CONSOLIDATION AND ALIGNMENT ENGINE...")

            progress = QProgressDialog("Consolidating independent data sweeps into session arrays...", None, 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            QApplication.processEvents()

            rec_out, beh_out, rec_shape, beh_shape = OfflineSessionCombiner.process_and_combine(
                rec_dir, beh_dir, self.ent_out_rec.text(), self.ent_out_beh.text(),
                experiment=self.combo_exp.currentText()
            )
            progress.close()

            self.ent_target_signal_file.setText(rec_out)
            if beh_out: self.ent_target_behavior_file.setText(beh_out)
            if not self.ent_target_output_dir.text(): self.ent_target_output_dir.setText(os.path.dirname(rec_out))
            
            self.append_status_log("SUCCESS", f"Session Unified Stream successfully parsed. Signal Shape: {rec_shape} | Behavioral Logs: {beh_shape}")
        except Exception as e:
            self.append_status_log("ERROR", f"Consolidation script aborted: {str(e)}")
            QMessageBox.critical(self, "Aggregation Error", str(e))
        finally:
            self.btn_run_merge.setEnabled(True)

    def execute_preprocessing_only(self):
        """Runs the offline preprocessing pipeline on the selected session file.

        Validates inputs (the behavior log is required only if the profile says so),
        runs the steps in table order with the entered parameters, and records the
        SCI/SQA-rejected channels so they can be exported.
        """
        try:
            sig_file = self.ent_target_signal_file.text()
            beh_file = self.ent_target_behavior_file.text()
            output_dir = self.ent_target_output_dir.text()
            defaults = config.DEFAULT_PREPROC_CONFIG

            selected_paradigm = self.combo_exp.currentText()
            behavior_required = config.get_experiment_profile(selected_paradigm)['behavior_required']

            if not sig_file or not os.path.exists(sig_file):
                raise FileNotFoundError("Target Signal matrix file path entry cannot be located or is invalid.")
            
            behav_df = None
            if behavior_required:
                if not beh_file or not os.path.exists(beh_file):
                    raise FileNotFoundError(f"A verified behavior log framework is marked explicitly MANDATORY for the '{selected_paradigm}' profile.")
                behav_df = read_csv(beh_file, log_callback=lambda m: self.append_status_log("WARNING", m))
            else:
                if beh_file and os.path.exists(beh_file):
                    behav_df = read_csv(beh_file, log_callback=lambda m: self.append_status_log("WARNING", m))
                    self.append_status_log("INFO", "Behavior array metrics identified for custom pipeline. Processing sync alignments.")
                else:
                    self.append_status_log("INFO", "Executing pipeline processing pipeline paths in standalone layout configuration modes.")

            compiled_steps = self.step_table.compiled_steps()

            self.btn_run_preproc.setEnabled(False)
            self.append_status_log("SYSTEM", "COMPILING AND LAUNCHING PIPELINE SEQUENCES IN SELECTED ORDER...")

            params = {
                'active_experiment': selected_paradigm,
                'pipeline_steps': compiled_steps,
                'cv_threshold': float(self.txt_cv_thresh.text()),
                'sci_threshold': float(self.txt_sci_thresh.text()),
                'low_cut': float(self.txt_low_cut.text()),
                'high_cut': float(self.txt_high_cut.text()),
                'peak_onset_buffer': float(self.txt_onset_buffer.text()),
                'peak_offset_buffer': float(self.txt_offset_buffer.text()),
                'fs': float(self.txt_fs.text()),
                'butterworth_order': defaults.get('butterworth_order', 5),
                'detrend_batches': defaults.get('detrend_batches', 10),
                'mbll_wavelengths': defaults.get('mbll_wavelengths', [735, 850]),
                'extinction_coefficients': defaults.get('extinction_coefficients', {}),
                'scaler_type': self.combo_scaler_type.currentText(),
                'manual_exclude_channels': [c.strip() for c in self.txt_channel_filter_list.text().split(",") if c.strip()],
            }

            session_df = read_csv(sig_file, log_callback=lambda m: self.append_status_log("WARNING", m))
            today_str = datetime.today().strftime("%Y-%m-%d")

            preprocessor = OfflineDataPreprocessor(
                log_callback=lambda msg: self.append_status_log("INFO", msg),
                warning_callback=lambda msg: self.append_status_log("WARNING", msg)
            )

            saved_paths = preprocessor.run_preprocessing_pipeline(
                session_df, behav_df, params, output_dir if output_dir else os.path.dirname(sig_file), today_str
            )

            if saved_paths:
                self.append_status_log("SUCCESS", f"Pipeline executed cleanly. Checkpoint outputs saved: {list(saved_paths.keys())}")
            else:
                self.append_status_log("SUCCESS", "Pipeline sequence completed. No intermediate checkpoints requested.")

            self.last_rejected_channels = preprocessor.get_rejected_channels()
            self.lbl_rejected_summary.setText(f"Last run rejected: {len(self.last_rejected_channels)} channel(s) via SCI/SQA.")
            self.btn_save_rejected.setEnabled(bool(self.last_rejected_channels))
            if self.last_rejected_channels:
                self.append_status_log("INFO", f"SCI/SQA rejected {len(self.last_rejected_channels)} channel(s) this run - use 'Save Rejected Channels...' to export the list.")

            QMessageBox.information(self, "Pipeline Complete", "Dataset arrays output smoothly matching your sequence layout constraints.")
        except Exception as e:
            self.append_status_log("ERROR", f"Pipeline calculation routine broke down: {str(e)}")
            QMessageBox.critical(self, "Pipeline Blocked", str(e))
        finally:
            self.btn_run_preproc.setEnabled(True)

    def browse_and_load_channel_list(self):
        """Loads a channel name list (JSON or flat text/CSV) into the manual filter field."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Channel List", "", "Channel List Files (*.json *.txt *.csv);;All Files (*)"
        )
        if not file_path:
            return
        try:
            channels = load_channel_list(file_path)
            self.txt_channel_filter_list.setText(", ".join(channels))
            self.append_status_log("SUCCESS", f"Loaded {len(channels)} channel name(s) from '{os.path.basename(file_path)}'.")
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Could not parse channel list file: {str(e)}")

    def save_rejected_channels(self):
        """Exports the SCI/SQA rejections from the most recent pipeline run to a channel-list file."""
        if not self.last_rejected_channels:
            return
        suggested_name = f"rejected_channels_{datetime.today().strftime('%Y-%m-%d')}.json"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Rejected Channel List", suggested_name, "JSON Files (*.json)")
        if not file_path:
            return
        try:
            saved = save_channel_list(
                file_path, self.last_rejected_channels,
                notes=f"SCI/SQA-rejected channels from an offline pipeline run on {datetime.today().strftime('%Y-%m-%d')}."
            )
            self.append_status_log("SUCCESS", f"Saved {len(saved)} rejected channel name(s) to '{os.path.basename(file_path)}'.")
            QMessageBox.information(self, "Saved", f"Rejected channel list saved to:\n{file_path}\n\nLoad this in the Neurofeedback tab's Channel Filter section to exclude the same channels live.")
        except Exception as e:
            self.append_status_log("ERROR", f"Failed to save rejected channel list: {str(e)}")
            QMessageBox.critical(self, "Save Failed", str(e))