"""Decoder tab: cross-validates registered decoders on preprocessed datasets and trains/exports the chosen one."""
import os
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
    QLineEdit, QPushButton, QCheckBox, QGroupBox, QTextEdit, \
    QFileDialog, QMessageBox, QApplication, QComboBox, QProgressDialog,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView,
    QScrollArea, QFrame, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt

import config
from core.decoders import decoder_selection, DECODER_REGISTRY
from core.io_utils import read_csv, write_csv
from ui.components import CollapsibleSection


class DecoderTrainingTab(QWidget):
    """Decoder benchmarking and training workspace.

    Cross-validates every checked decoder on every listed preprocessed CSV,
    ranks the results by macro F1, and fits the chosen decoder on a full dataset,
    saving it with joblib to ./decoders/<name>.pkl.
    """
    def __init__(self, parent=None):
        """Builds the tab UI and initialises the evaluation results store."""
        super().__init__(parent)
        self.evaluation_results = []  # one dict per (file, decoder): file_name, file_path, decoder, f1, auc

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

        # Panel 1: dataset list, CV strategy and decoder checklist
        data_section = CollapsibleSection("1. Evaluation Dataset Browser / Cross-Validation Config")
        data_section.set_expanded(True)
        data_layout = QVBoxLayout()
        data_layout.setSpacing(10)
        
        data_layout.addWidget(QLabel("Target Evaluation Dataset Files Pool:"))
        self.data_files_list = QListWidget()
        self.data_files_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.data_files_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #454545;
                border-radius: 4px;
                color: #ffffff;
                padding: 5px;
            }
            QListWidget::item {
                padding: 6px;
                border-bottom: 1px solid #2d2d2d;
                color: #e0e0e0;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        self.data_files_list.setFixedHeight(120)
        data_layout.addWidget(self.data_files_list)

        btn_grid = QHBoxLayout()
        btn_grid.setSpacing(8)
        
        btn_add_files = QPushButton("➕ Add Preprocessed Data File(s)...")
        btn_add_files.setStyleSheet("background-color: #383838; color: white; padding: 6px; font-size: 11px;")
        btn_add_files.clicked.connect(self.browse_and_append_data_files)
        
        btn_clear_files = QPushButton("🗑️ Clear List")
        btn_clear_files.setStyleSheet("background-color: #5c2424; color: white; padding: 6px; font-size: 11px;")
        btn_clear_files.clicked.connect(self.data_files_list.clear)
        
        btn_grid.addWidget(btn_add_files)
        btn_grid.addWidget(btn_clear_files)
        data_layout.addLayout(btn_grid)

        cv_grid = QGridLayout()
        cv_grid.setSpacing(8)
        cv_grid.addWidget(QLabel("Cross-Validation Split Strategy:"), 0, 0)
        self.combo_cv = QComboBox()
        self.combo_cv.addItems(["leaveonegroupout", "stratifiedshufflesplit"])
        cv_grid.addWidget(self.combo_cv, 0, 1)
        data_layout.addLayout(cv_grid)

        # Every registered decoder, checked by default; unchecked ones are skipped.
        data_layout.addWidget(QLabel("Decoders to Evaluate:"))
        self.decoder_checklist = QListWidget()
        self.decoder_checklist.setStyleSheet(self.data_files_list.styleSheet())
        self.decoder_checklist.setFixedHeight(160)
        for key in DECODER_REGISTRY.keys():
            item = QListWidgetItem(key.upper())
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, key)  # the actual registry key (lowercase), for lookup at run time
            self.decoder_checklist.addItem(item)
        data_layout.addWidget(self.decoder_checklist)

        decoder_btn_grid = QHBoxLayout()
        decoder_btn_grid.setSpacing(8)
        btn_select_all_decoders = QPushButton("Select All")
        btn_select_all_decoders.setStyleSheet("background-color: #383838; color: white; padding: 6px; font-size: 11px;")
        btn_select_all_decoders.clicked.connect(lambda: self._set_all_decoder_checks(Qt.Checked))
        btn_deselect_all_decoders = QPushButton("Deselect All")
        btn_deselect_all_decoders.setStyleSheet("background-color: #383838; color: white; padding: 6px; font-size: 11px;")
        btn_deselect_all_decoders.clicked.connect(lambda: self._set_all_decoder_checks(Qt.Unchecked))
        decoder_btn_grid.addWidget(btn_select_all_decoders)
        decoder_btn_grid.addWidget(btn_deselect_all_decoders)
        data_layout.addLayout(decoder_btn_grid)

        data_section.set_content_layout(data_layout)
        left_column.addWidget(data_section)

        # Panel 2: evaluation results table
        results_section = CollapsibleSection("2. Model Evaluation Rankings Matrix")
        results_section.set_expanded(True)
        results_layout = QVBoxLayout()
        results_layout.setSpacing(12)

        self.btn_run_eval = QPushButton("Run Comparative Multi-Decoder Evaluation Routine")
        self.btn_run_eval.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold; padding: 10px; font-size: 11px;")
        self.btn_run_eval.clicked.connect(self.execute_comparative_evaluations)
        results_layout.addWidget(self.btn_run_eval)

        self.table_results = QTableWidget(0, 5)
        self.table_results.setHorizontalHeaderLabels(["Target File Name", "Decoder Architecture", "F1-Score (Macro)", "ROC AUC", "Full Absolute Path"])
        self.table_results.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_results.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_results.verticalHeader().setVisible(True)
        self.table_results.setAlternatingRowColors(True)
        
        self.table_results.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.table_results.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_results.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_results.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_results.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table_results.setColumnWidth(0, 220)
        
        self.table_results.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                alternate-background-color: #282828;
                gridline-color: #383838;
                border: 1px solid #454545;
                border-radius: 4px;
            }
            QTableWidget::item {
                border-bottom: 1px solid #2d2d2d;
                color: #ffffff;
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #ffffff;
                font-weight: bold;
                padding: 8px;
                border: 1px solid #3e3e3e;
                font-size: 11px;
            }
        """)
        self.table_results.setFixedHeight(260)
        self.table_results.itemSelectionChanged.connect(self.on_table_row_selected)
        results_layout.addWidget(self.table_results)

        self.btn_save_csv = QPushButton("\U0001F4BE Save Evaluation Table (CSV)...")
        self.btn_save_csv.setEnabled(False)
        self.btn_save_csv.setStyleSheet("background-color: #383838; color: white; padding: 6px; font-size: 11px;")
        self.btn_save_csv.clicked.connect(self.save_evaluation_table_csv)
        results_layout.addWidget(self.btn_save_csv)

        results_section.set_content_layout(results_layout)
        left_column.addWidget(results_section)

        # Panel 3: full-data training and export
        train_section = CollapsibleSection("3. Fit and Train a Decoder on the Full Data")
        train_layout = QGridLayout()
        train_layout.setSpacing(10)

        train_layout.addWidget(QLabel("Target Training Dataset:"), 0, 0)
        self.lbl_selected_model_info = QLabel("None Selected (Pick a row from the ranking table above)")
        self.lbl_selected_model_info.setStyleSheet("color: #b0b0b0; font-style: italic; font-weight: bold;")
        train_layout.addWidget(self.lbl_selected_model_info, 0, 1, 1, 2)

        train_layout.addWidget(QLabel("Decoder to Train:"), 1, 0)
        self.combo_train_decoder = QComboBox()
        self.combo_train_decoder.addItems([key.upper() for key in DECODER_REGISTRY.keys()])
        self.combo_train_decoder.setToolTip(
            "Defaults to the decoder from whichever ranking row you select, but you can pick any "
            "other registered decoder to fit on the full dataset instead."
        )
        train_layout.addWidget(self.combo_train_decoder, 1, 1, 1, 2)

        train_layout.addWidget(QLabel("Custom Model Name:"), 2, 0)
        self.txt_model_filename = QLineEdit()
        self.txt_model_filename.setPlaceholderText("e.g., optimized_svm_filtered_run")
        train_layout.addWidget(self.txt_model_filename, 2, 1, 1, 2)

        self.btn_train_save = QPushButton("Fit and Train Selected Decoder on Full Data")
        self.btn_train_save.setEnabled(False)
        self.btn_train_save.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 10px; font-size: 12px;")
        self.btn_train_save.clicked.connect(self.execute_final_training_and_export)
        train_layout.addWidget(self.btn_train_save, 3, 0, 1, 3)

        train_section.set_content_layout(train_layout)
        left_column.addWidget(train_section)

        scroll_area.setWidget(scroll_content)
        master_layout.addWidget(scroll_area, stretch=5)

        # Right column: log console
        log_group = QGroupBox("Model Analytics Terminal Dashboard")
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

        self.append_status_log("SYSTEM", "Decoder Orchestration Workspace initialized safely. Load data files above to begin benchmarking loops.")

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

    def _set_all_decoder_checks(self, state):
        """Sets every item in the decoder checklist to the given Qt check state."""
        for i in range(self.decoder_checklist.count()):
            self.decoder_checklist.item(i).setCheckState(state)

    def browse_and_append_data_files(self):
        """Opens a multi-file picker (starting in ../2_data) and appends new CSV paths to the dataset list."""
        base_data_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "2_data")
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Preprocessed Dataset Files for Comparison", 
            base_data_path, "CSV Data Matrices (*.csv)"
        )
        if file_paths:
            for path in file_paths:
                existing_items = [self.data_files_list.item(i).text() for i in range(self.data_files_list.count())]
                if path not in existing_items:
                    self.data_files_list.addItem(path)
                    self.append_status_log("INFO", f"Registered dataset target array: {os.path.basename(path)}")

    def load_dataset_from_path(self, file_path):
        """Loads a preprocessed CSV and splits it into features, labels and CV groups.

        Returns:
            (data, labels, groups): features with config.NON_SIGNAL_COLUMNS dropped,
            the 'target' column, and the 'run' values used as CV groups.

        Raises:
            FileNotFoundError: if the file does not exist.
            KeyError: if the 'target' or 'run' column is missing.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Selected file targets missing on system: {file_path}")
            
        loaded_data = read_csv(file_path, log_callback=lambda m: self.append_status_log("WARNING", m))
        
        if "target" not in loaded_data.columns or "run" not in loaded_data.columns:
            raise KeyError("Invalid internal file structure. Data must contain explicit columns labeled 'target' and 'run'.")

        excluded_cols = config.NON_SIGNAL_COLUMNS

        groups = loaded_data["run"].values
        labels = loaded_data["target"]
        
        # Only drop columns that exist, so both epoch-level and block-averaged files work.
        data = loaded_data.drop([col for col in excluded_cols if col in loaded_data.columns], axis=1)
        return data, labels, groups

    def execute_comparative_evaluations(self):
        """Cross-validates each checked decoder on each listed dataset file.

        Uses LeaveOneGroupOut (groups = run) or a default StratifiedShuffleSplit,
        storing mean macro F1 and ROC AUC (in %) per file/decoder. ROC AUC uses
        predict_proba when available, else the hard predictions. Files that fail
        to load are logged and skipped; decoders that raise are silently skipped.
        """
        file_count = self.data_files_list.count()
        if file_count == 0:
            QMessageBox.warning(self, "Datasets Needed", "Please load at least one data file using the file browser before running evaluations.")
            return

        classifiers = [
            self.decoder_checklist.item(i).data(Qt.UserRole)
            for i in range(self.decoder_checklist.count())
            if self.decoder_checklist.item(i).checkState() == Qt.Checked
        ]
        if not classifiers:
            QMessageBox.warning(self, "Decoders Needed", "Select at least one decoder from the checklist before running evaluations.")
            return

        cv_method = self.combo_cv.currentText()
        self.evaluation_results.clear()
        self.table_results.setRowCount(0)
        self.btn_save_csv.setEnabled(False)
        
        self.btn_run_eval.setEnabled(False)
        self.append_status_log("SYSTEM", f"Starting cross-validation sequence across {file_count} dataset target(s) and {len(classifiers)} selected decoder(s)...")

        progress = QProgressDialog("Calculating validation scoring matrices...", "Abort Process", 0, file_count * len(classifiers), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        step_counter = 0
        
        for i in range(file_count):
            file_path = self.data_files_list.item(i).text()
            file_name = os.path.basename(file_path)
            
            self.append_status_log("INFO", f"Parsing target arrays inside file: '{file_name}'...")
            try:
                data, labels, groups = self.load_dataset_from_path(file_path)
            except Exception as e:
                self.append_status_log("ERROR", f"Bypassed tracking file '{file_name}': {str(e)}")
                continue

            for decoder in classifiers:
                if progress.wasCanceled():
                    self.append_status_log("WARNING", "Evaluation routine interrupted by operator command.")
                    break
                
                progress.setLabelText(f"Evaluating {decoder.upper()} on file:\n{file_name}")
                progress.setValue(step_counter)
                QApplication.processEvents()

                try:
                    pipeline = decoder_selection(preprocessing=True, selected_method=decoder)
                    cv = LeaveOneGroupOut() if cv_method == "leaveonegroupout" else StratifiedShuffleSplit()
                    f_score_res, roc_auc_res = [], []

                    for train_idx, test_idx in cv.split(data, labels, groups=groups):
                        pipeline.fit(data.iloc[train_idx], labels.iloc[train_idx])
                        y_pred = pipeline.predict(data.iloc[test_idx])

                        try:
                            y_score = pipeline.predict_proba(data.iloc[test_idx])[:, 1]
                            roc_auc = roc_auc_score(labels.iloc[test_idx], y_score)
                        except (AttributeError, IndexError):
                            roc_auc = roc_auc_score(labels.iloc[test_idx], y_pred)

                        f1 = f1_score(labels.iloc[test_idx], y_pred, average='macro')
                        roc_auc_res.append(roc_auc)
                        f_score_res.append(f1)

                    mean_f1 = np.mean(f_score_res) * 100
                    mean_auc = np.mean(roc_auc_res) * 100

                    self.evaluation_results.append({
                        'file_name': file_name,
                        'file_path': file_path,
                        'decoder': decoder,
                        'f1': mean_f1,
                        'auc': mean_auc
                    })

                except Exception as e:
                    pass  # skip decoders that fail on this dataset
                
                step_counter += 1

        progress.close()
        self.populate_and_sort_results_matrix()
        self.btn_run_eval.setEnabled(True)

    def populate_and_sort_results_matrix(self):
        """Sorts results by F1 (descending), fills the results table and selects the top row."""
        self.evaluation_results.sort(key=lambda x: x['f1'], reverse=True)
        
        self.table_results.setRowCount(len(self.evaluation_results))
        for idx, record in enumerate(self.evaluation_results):
            self.table_results.setRowHeight(idx, 40)
            
            item_file = QTableWidgetItem(record['file_name'])
            item_decoder = QTableWidgetItem(record['decoder'].upper())
            item_f1 = QTableWidgetItem(f"{record['f1']:.2f}%")
            item_auc = QTableWidgetItem(f"{record['auc']:.2f}%")
            item_path = QTableWidgetItem(record['file_path'])

            item_file.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            item_path.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            for item in [item_decoder, item_f1, item_auc]:
                item.setTextAlignment(Qt.AlignCenter)

            # Read-only cells
            for item in [item_file, item_decoder, item_f1, item_auc, item_path]:
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

            self.table_results.setItem(idx, 0, item_file)
            self.table_results.setItem(idx, 1, item_decoder)
            self.table_results.setItem(idx, 2, item_f1)
            self.table_results.setItem(idx, 3, item_auc)
            self.table_results.setItem(idx, 4, item_path)

        if self.evaluation_results:
            best_record = self.evaluation_results[0]
            self.append_status_log("SUCCESS", f"Evaluation cycles completed. Top configuration identified: {best_record['decoder'].upper()} on '{best_record['file_name']}' ({best_record['f1']:.2f}% F1).")
            self.table_results.selectRow(0) 
        else:
            self.append_status_log("WARNING", "No model metrics generated. Ensure your loaded CSVs possess valid structural configurations.")

        self.btn_save_csv.setEnabled(bool(self.evaluation_results))

    def save_evaluation_table_csv(self):
        """Exports the current ranked evaluation results to a CSV file."""
        if not self.evaluation_results:
            QMessageBox.warning(self, "No Results", "Run an evaluation first - there's nothing to save yet.")
            return

        suggested_name = f"decoder_evaluation_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.csv"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Evaluation Results", suggested_name, "CSV Files (*.csv)")
        if not file_path:
            return

        try:
            export_df = pd.DataFrame(self.evaluation_results).rename(columns={
                'file_name': 'Target File Name',
                'decoder': 'Decoder Architecture',
                'f1': 'F1-Score (Macro, %)',
                'auc': 'ROC AUC (%)',
                'file_path': 'Full Absolute Path',
            })[['Target File Name', 'Decoder Architecture', 'F1-Score (Macro, %)', 'ROC AUC (%)', 'Full Absolute Path']]
            write_csv(export_df, file_path)
            self.append_status_log("SUCCESS", f"Evaluation table exported to: {file_path}")
            QMessageBox.information(self, "Export Complete", f"Evaluation results saved to:\n{file_path}")
        except Exception as e:
            self.append_status_log("ERROR", f"Failed to export evaluation table: {str(e)}")
            QMessageBox.critical(self, "Export Failed", str(e))

    def on_table_row_selected(self):
        """Uses the selected ranking row to pre-fill the training decoder and default model name."""
        selected_ranges = self.table_results.selectedRanges()
        if not selected_ranges:
            self.btn_train_save.setEnabled(False)
            return

        target_row = selected_ranges[0].topRow()
        file_name = self.table_results.item(target_row, 0).text()
        decoder = self.table_results.item(target_row, 1).text().lower()

        self.lbl_selected_model_info.setText(f"File: {file_name}")
        self.lbl_selected_model_info.setStyleSheet("color: #00ff66; font-weight: bold; font-style: normal;")

        # Default to the ranked decoder; the user can still pick another one.
        combo_idx = self.combo_train_decoder.findText(decoder.upper())
        if combo_idx >= 0:
            self.combo_train_decoder.setCurrentIndex(combo_idx)

        clean_file_root = os.path.splitext(file_name)[0].replace(" ", "_")
        self.txt_model_filename.setText(f"model_{decoder}_{clean_file_root}")
        self.btn_train_save.setEnabled(True)

    def execute_final_training_and_export(self):
        """Fits the chosen decoder on the selected row's full dataset and saves it.

        The fitted pipeline is written with joblib to ./decoders/<model name>.pkl
        (relative to the current working directory).
        """
        selected_ranges = self.table_results.selectedRanges()
        if not selected_ranges: return

        target_row = selected_ranges[0].topRow()
        file_path = self.table_results.item(target_row, 4).text()
        decoder = self.combo_train_decoder.currentText().lower()
        
        raw_name = self.txt_model_filename.text().strip().replace(" ", "_")
        if not raw_name:
            QMessageBox.warning(self, "Invalid Identifier", "Please state a valid label string to name the compiled model file.")
            return

        # Models go to ./decoders under the current working directory.
        decoders_dir = os.path.join(os.path.abspath(os.getcwd()), "decoders")
        os.makedirs(decoders_dir, exist_ok=True)
        export_target_path = os.path.join(decoders_dir, f"{raw_name}.pkl")

        self.btn_train_save.setEnabled(False)
        self.append_status_log("SYSTEM", f"Starting full-set production training for model '{decoder.upper()}' using file: {os.path.basename(file_path)}...")

        try:
            data, labels, _ = self.load_dataset_from_path(file_path)

            production_pipeline = decoder_selection(preprocessing=True, selected_method=decoder)
            production_pipeline.fit(data, labels)

            joblib.dump(production_pipeline, export_target_path)
            
            self.append_status_log("SUCCESS", f"Production model deployed seamlessly to system location: {export_target_path}")
            QMessageBox.information(self, "Export Complete", f"Trained production decoder saved successfully to:\n{export_target_path}")
        except Exception as e:
            self.append_status_log("ERROR", f"Production compilation process encountered a fatal error: {str(e)}")
            QMessageBox.critical(self, "Compilation Error", str(e))
        finally:
            self.btn_train_save.setEnabled(True)