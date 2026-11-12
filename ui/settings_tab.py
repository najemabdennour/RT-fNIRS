# ui/settings_tab.py
"""
Settings tab: edit every setting of config.py from the GUI.

config.py stays the single source of truth and stays hand-editable: the
tab reads the file, shows each setting with the comment that documents it
there, and Save patches only the changed values back into the file in
place (core.config_io), keeping comments and layout, with a backup.
"""
import ast
import copy
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
                             QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QScrollArea,
                             QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
                             QAbstractItemView)

import config
from core import config_io

DESCRIPTION_STYLE = "color: #9a9a9a; font-size: 10px;"

# Labels for settings whose name alone is unclear (others are derived from the name).
LABELS = {
    'DEFAULT_EXPERIMENT': "Default experiment profile",
    'COADAPTATION_ACCEPTANCE_CRITERIA': "Co-adaptation acceptance probability",
    'DEFAULT_RAW_RECORDINGS': "Raw recordings folder",
    'DEFAULT_RAW_BEHAVIORAL': "Raw behavioral logs folder",
    'DEFAULT_OUT_SESSIONS': "Merged sessions output folder",
    'DEFAULT_OUT_BEHAVIORAL': "Merged behavioral logs output folder",
    'SIMULATOR_LAYOUT_FILE': "Dev Mode simulator layout table",
    'HELMET_TEMPLATE_DIR': "Helmet templates folder",
    'NON_SIGNAL_COLUMNS': "Non-signal columns",
    'butterworth_filter': "Run Butterworth filter (default)",
    'detrend': "Run detrending (default)",
    'averaging': "Run trial averaging (default)",
    'block_avg': "Run block averaging (default)",
    'enable_od': "Run optical density conversion (default)",
    'enable_sci': "Run SCI (default)",
    'enable_sqa': "Run SQA (default)",
    'enable_mbll': "Run MBLL (default)",
    'scaling': "Run scaling (default)",
    'sqa_cv_threshold': "SQA CV threshold (%)",
    'sci_correlation_min': "SCI minimum correlation",
    'mbll_wavelengths': "MBLL wavelengths (nm)",
}
SCALER_TYPES = ["standard", "minmax"]


def _label(name):
    """Readable label for a setting name."""
    return LABELS.get(name, name.replace('_', ' ').capitalize())


def _split_list(text):
    """Comma-separated text -> list of stripped, non-empty strings."""
    return [t.strip() for t in text.split(',') if t.strip()]


def _parse_scalar(text):
    """A Python literal if the text is one (True, 2, 1.5, 'x'), else the text itself."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


class SettingsTab(QWidget):
    """
    Form over config.py: general settings, preprocessing defaults (with the
    extinction-coefficient table), folder paths, non-signal columns and the
    experiment-profile table, plus Save / Reload. Changes are written to
    config.py in place and applied to the running app.
    """

    def __init__(self, parent=None, config_path=None):
        """
        Args:
            config_path: config.py to edit (default: the imported config module's file).
        """
        super().__init__(parent)
        self.config_path = config_path or os.path.abspath(config.__file__.replace('.pyc', '.py'))
        self.cfg = config_io.ConfigFile(self.config_path)
        self.preproc_widgets = {}
        self.path_widgets = {}
        self.init_ui()
        self.populate()

    # --- UI construction ---

    def init_ui(self):
        """Builds the header (file path, Save, Reload, status) and the scrollable settings sections."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)

        header = QHBoxLayout()
        header.addWidget(QLabel(f"<b>Settings file:</b> {self.config_path}"))
        header.addStretch()
        self.btn_reload = QPushButton("Reload from config.py")
        self.btn_reload.setToolTip("Discard unsaved changes here and re-read config.py (e.g. after editing it by hand).")
        self.btn_reload.clicked.connect(self.reload)
        header.addWidget(self.btn_reload)
        self.btn_save = QPushButton("Save to config.py")
        self.btn_save.setToolTip("Writes only the changed values into config.py, keeping its comments and "
                                 "layout; the previous version is kept as config.py.bak.")
        self.btn_save.clicked.connect(self.save)
        header.addWidget(self.btn_save)
        outer.addLayout(header)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet(DESCRIPTION_STYLE)
        outer.addWidget(self.lbl_status)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        self.sections = QVBoxLayout(content)
        self.sections.setSpacing(12)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        self._build_general()
        self._build_preprocessing()
        self._build_paths()
        self._build_columns()
        self._build_profiles()
        self.sections.addStretch()

    def _grid_section(self, title):
        """A group box with a label | field | description grid; returns the grid."""
        box = QGroupBox(title)
        grid = QGridLayout(box)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 2)
        self.sections.addWidget(box)
        return grid

    def _add_row(self, grid, label, widget, description):
        """Adds one setting row: label, field and its config.py comment."""
        row = grid.rowCount()
        lbl = QLabel(label)
        lbl.setToolTip(description)
        grid.addWidget(lbl, row, 0)
        grid.addWidget(widget, row, 1)
        desc = QLabel(description)
        desc.setStyleSheet(DESCRIPTION_STYLE)
        desc.setWordWrap(True)
        grid.addWidget(desc, row, 2)

    def _build_general(self):
        """General section: default experiment profile and co-adaptation threshold."""
        grid = self._grid_section("General")
        self.combo_default_experiment = QComboBox()
        self._add_row(grid, _label('DEFAULT_EXPERIMENT'), self.combo_default_experiment,
                      self.cfg.comment_for('DEFAULT_EXPERIMENT'))
        self.spin_coadapt = QDoubleSpinBox()
        self.spin_coadapt.setRange(0.0, 1.0)
        self.spin_coadapt.setDecimals(3)
        self.spin_coadapt.setSingleStep(0.05)
        self._add_row(grid, _label('COADAPTATION_ACCEPTANCE_CRITERIA'), self.spin_coadapt,
                      self.cfg.comment_for('COADAPTATION_ACCEPTANCE_CRITERIA'))

    def _build_preprocessing(self):
        """Preprocessing defaults: one field per DEFAULT_PREPROC_CONFIG entry, typed from its value."""
        grid = self._grid_section("Preprocessing defaults (DEFAULT_PREPROC_CONFIG)")
        for key, value in self.cfg.values.get('DEFAULT_PREPROC_CONFIG', {}).items():
            heading = self.cfg.heading_for('DEFAULT_PREPROC_CONFIG', key)
            if heading:
                self._add_heading(grid, heading)
            if key == 'extinction_coefficients':
                self._build_extinction_table(grid)
                continue
            if isinstance(value, bool):
                widget = QCheckBox()
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(-1_000_000, 1_000_000)
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setRange(-1e9, 1e9)
                # At least 3 decimals, more if the value in config.py needs them.
                decimals = len(repr(value).split('.')[1]) if '.' in repr(value) and 'e' not in repr(value) else 3
                widget.setDecimals(min(max(3, decimals), 8))
                widget.setSingleStep(0.01)
            elif key == 'scaler_type':
                widget = QComboBox()
                widget.addItems(SCALER_TYPES)
            else:
                widget = QLineEdit()
                if isinstance(value, list):
                    widget.setPlaceholderText("comma-separated values")
            self.preproc_widgets[key] = widget
            self._add_row(grid, _label(key), widget, self.cfg.comment_for('DEFAULT_PREPROC_CONFIG', key))

    def _add_heading(self, grid, text):
        """A sub-heading row in a section (from a section comment in config.py)."""
        lbl = QLabel(f"<b>{text}</b>")
        lbl.setWordWrap(True)
        grid.addWidget(lbl, grid.rowCount(), 0, 1, 3)

    def _build_extinction_table(self, grid):
        """The extinction coefficient table (wavelength, e_HbO, e_HbR) with add/remove buttons."""
        self.table_ext = QTableWidget(0, 3)
        self.table_ext.setHorizontalHeaderLabels(["Wavelength (nm)", "e_HbO", "e_HbR"])
        self.table_ext.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_ext.setMinimumHeight(220)
        ext_box = QWidget()
        ext_layout = QVBoxLayout(ext_box)
        ext_layout.setContentsMargins(0, 0, 0, 0)
        ext_layout.addWidget(self.table_ext)
        buttons = QHBoxLayout()
        btn_add = QPushButton("Add Wavelength")
        btn_add.clicked.connect(lambda: self._add_ext_row("", "", ""))
        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(lambda: self._remove_selected(self.table_ext))
        buttons.addWidget(btn_add)
        buttons.addWidget(btn_remove)
        buttons.addStretch()
        ext_layout.addLayout(buttons)
        self._add_row(grid, "Extinction coefficients", ext_box,
                      self.cfg.comment_for('DEFAULT_PREPROC_CONFIG', 'extinction_coefficients'))

    def _build_paths(self):
        """Folder/file paths, each with a Browse button."""
        grid = self._grid_section("Folders and files")
        for name in config_io.PATH_SETTINGS:
            if name not in self.cfg.values:
                continue
            edit = QLineEdit()
            if name == 'SIMULATOR_LAYOUT_FILE':
                edit.setPlaceholderText("empty = 200 generic simulator channels")
            btn = QPushButton("Browse")
            btn.clicked.connect(lambda _c, n=name, e=edit: self._browse_path(n, e))
            field = QWidget()
            row = QHBoxLayout(field)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(edit)
            row.addWidget(btn)
            self.path_widgets[name] = edit
            self._add_row(grid, _label(name), field, self.cfg.comment_for(name))

    def _build_columns(self):
        """Non-signal column list (comma-separated)."""
        grid = self._grid_section("Columns")
        self.edit_non_signal = QLineEdit()
        self._add_row(grid, _label('NON_SIGNAL_COLUMNS'), self.edit_non_signal, self.cfg.comment_for('NON_SIGNAL_COLUMNS'))

    def _build_profiles(self):
        """Experiment profile table with add/remove buttons and the field descriptions from config.py."""
        box = QGroupBox("Experiment profiles (EXPERIMENT_PROFILES)")
        layout = QVBoxLayout(box)
        desc = QLabel(self.cfg.comment_for('EXPERIMENT_PROFILES'))
        desc.setStyleSheet(DESCRIPTION_STYLE)
        desc.setWordWrap(True)
        layout.addWidget(desc)
        self.table_profiles = QTableWidget(0, 6)
        self.table_profiles.setHorizontalHeaderLabels(
            ["Name", "Columns", "Label (target) column", "Carry columns", "Keep trials where", "Behavior required"])
        header = self.table_profiles.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        for col, width in ((0, 160), (2, 140), (3, 220), (4, 160), (5, 120)):
            self.table_profiles.setColumnWidth(col, width)
        self.table_profiles.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_profiles.setMinimumHeight(200)
        self.table_profiles.itemChanged.connect(lambda item: item.column() == 0 and self._refresh_experiment_names())
        self.table_profiles.setToolTip(
            "Columns / carry columns: comma-separated. Keep trials where: e.g. correct=True, block=2 "
            "(empty keeps every trial). Label column: empty = none (target defaults to 1).")
        layout.addWidget(self.table_profiles)
        buttons = QHBoxLayout()
        btn_add = QPushButton("Add Profile")
        btn_add.clicked.connect(lambda: self._add_profile_row("new_profile", {}))
        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(lambda: (self._remove_selected(self.table_profiles), self._refresh_experiment_names()))
        buttons.addWidget(btn_add)
        buttons.addWidget(btn_remove)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.sections.addWidget(box)

    # --- Filling and reading the form ---

    def populate(self):
        """Fills every field from the loaded config.py values."""
        v = self.cfg.values
        self.spin_coadapt.setValue(float(v.get('COADAPTATION_ACCEPTANCE_CRITERIA', 0.0)))

        pre = v.get('DEFAULT_PREPROC_CONFIG', {})
        for key, widget in self.preproc_widgets.items():
            value = pre.get(key)
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QComboBox):
                if widget.findText(str(value)) < 0:
                    widget.addItem(str(value))
                widget.setCurrentText(str(value))
            else:
                widget.setText(", ".join(str(x) for x in value) if isinstance(value, list) else str(value))
        self.table_ext.setRowCount(0)
        for wl, (e_hbo, e_hbr) in pre.get('extinction_coefficients', {}).items():
            self._add_ext_row(wl, e_hbo, e_hbr)

        for name, edit in self.path_widgets.items():
            edit.setText(v.get(name) or "")
        self.edit_non_signal.setText(", ".join(v.get('NON_SIGNAL_COLUMNS', [])))

        self.table_profiles.blockSignals(True)
        self.table_profiles.setRowCount(0)
        for name, profile in v.get('EXPERIMENT_PROFILES', {}).items():
            self._add_profile_row(name, profile)
        self.table_profiles.blockSignals(False)
        self._refresh_experiment_names()
        self.combo_default_experiment.setCurrentText(v.get('DEFAULT_EXPERIMENT', ''))
        self.lbl_status.setText("Loaded from config.py. Changes here are written back to the file only when you press Save.")

    def collect(self):
        """
        Reads every field into {setting name: value} in config.py's types.
        Raises ValueError with a readable message on invalid input.
        """
        values = {'DEFAULT_EXPERIMENT': self.combo_default_experiment.currentText(),
                  'COADAPTATION_ACCEPTANCE_CRITERIA': round(self.spin_coadapt.value(), 6)}

        original = self.cfg.values.get('DEFAULT_PREPROC_CONFIG', {})
        pre = {}
        for key, widget in self.preproc_widgets.items():
            if isinstance(widget, QCheckBox):
                pre[key] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                pre[key] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                pre[key] = round(widget.value(), 6)
            elif isinstance(widget, QComboBox):
                pre[key] = widget.currentText()
            elif isinstance(original.get(key), list):
                items = [_parse_scalar(t) for t in _split_list(widget.text())]
                if not all(isinstance(x, (int, float)) for x in items):
                    raise ValueError(f"'{_label(key)}' must be comma-separated numbers.")
                pre[key] = items
            else:
                pre[key] = widget.text()
        ext = {}
        for row in range(self.table_ext.rowCount()):
            cells = [(self.table_ext.item(row, c).text().strip() if self.table_ext.item(row, c) else "")
                     for c in range(3)]
            if not any(cells):
                continue
            try:
                ext[str(int(float(cells[0])))] = [float(cells[1]), float(cells[2])]
            except ValueError:
                raise ValueError(f"Extinction coefficients row {row + 1} needs a wavelength and two numbers.")
        pre['extinction_coefficients'] = ext
        missing = [w for w in pre.get('mbll_wavelengths', []) if str(w) not in ext]
        if missing:
            raise ValueError(f"MBLL wavelengths {missing} have no extinction coefficients.")
        # Keep config.py's key order (and any key the form doesn't show).
        values['DEFAULT_PREPROC_CONFIG'] = {k: pre.get(k, original[k]) for k in original}
        values['DEFAULT_PREPROC_CONFIG'].update({k: v for k, v in pre.items() if k not in original})

        for name, edit in self.path_widgets.items():
            values[name] = edit.text().strip() or None
        values['NON_SIGNAL_COLUMNS'] = _split_list(self.edit_non_signal.text())

        profiles = {}
        for row in range(self.table_profiles.rowCount()):
            text = lambda c: (self.table_profiles.item(row, c).text().strip() if self.table_profiles.item(row, c) else "")
            name = text(0)
            if not name:
                raise ValueError(f"Experiment profile row {row + 1} has no name.")
            if name in profiles:
                raise ValueError(f"Experiment profile '{name}' appears twice.")
            keep = {}
            for part in _split_list(text(4)):
                if '=' not in part:
                    raise ValueError(f"'{name}': keep-trials condition '{part}' must look like column=value.")
                col, val = part.split('=', 1)
                keep[col.strip()] = _parse_scalar(val.strip())
            profiles[name] = {
                'columns': _split_list(text(1)),
                'target_column': text(2) or None,
                'carry_columns': _split_list(text(3)),
                'keep_trials_where': keep,
                'behavior_required': self.table_profiles.item(row, 5).checkState() == Qt.Checked,
            }
        if not profiles:
            raise ValueError("At least one experiment profile is needed.")
        values['EXPERIMENT_PROFILES'] = profiles
        if values['DEFAULT_EXPERIMENT'] not in profiles:
            raise ValueError("The default experiment must be one of the experiment profiles.")
        return values

    # --- Row helpers ---

    def _add_ext_row(self, wavelength, e_hbo, e_hbr):
        """Appends an extinction-coefficient row."""
        row = self.table_ext.rowCount()
        self.table_ext.insertRow(row)
        for col, value in enumerate((wavelength, e_hbo, e_hbr)):
            self.table_ext.setItem(row, col, QTableWidgetItem(str(value)))

    def _add_profile_row(self, name, profile):
        """Appends an experiment-profile row."""
        row = self.table_profiles.rowCount()
        self.table_profiles.insertRow(row)
        keep = ", ".join(f"{k}={v!r}" for k, v in profile.get('keep_trials_where', {}).items())
        cells = [name, ", ".join(profile.get('columns', [])), profile.get('target_column') or "",
                 ", ".join(profile.get('carry_columns', [])), keep]
        for col, text in enumerate(cells):
            self.table_profiles.setItem(row, col, QTableWidgetItem(text))
        required = QTableWidgetItem()
        required.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        required.setCheckState(Qt.Checked if profile.get('behavior_required') else Qt.Unchecked)
        self.table_profiles.setItem(row, 5, required)
        self._refresh_experiment_names()

    @staticmethod
    def _remove_selected(table):
        """Removes the selected rows of a table."""
        for row in sorted({index.row() for index in table.selectedIndexes()}, reverse=True):
            table.removeRow(row)

    def _refresh_experiment_names(self):
        """Keeps the default-experiment choices in step with the profile names in the table."""
        current = self.combo_default_experiment.currentText()
        names = [self.table_profiles.item(r, 0).text().strip() for r in range(self.table_profiles.rowCount())
                 if self.table_profiles.item(r, 0) and self.table_profiles.item(r, 0).text().strip()]
        self.combo_default_experiment.blockSignals(True)
        self.combo_default_experiment.clear()
        self.combo_default_experiment.addItems(names)
        if current in names:
            self.combo_default_experiment.setCurrentText(current)
        self.combo_default_experiment.blockSignals(False)

    def _browse_path(self, name, edit):
        """Picks a folder (or, for the simulator layout, a CSV file) for a path setting."""
        start = edit.text() or self.cfg.base_dir
        if name == 'SIMULATOR_LAYOUT_FILE':
            path, _ = QFileDialog.getOpenFileName(self, _label(name), start, "CSV Files (*.csv)")
        else:
            path = QFileDialog.getExistingDirectory(self, _label(name), start)
        if path:
            edit.setText(path)

    # --- Save / reload ---

    def has_unsaved_changes(self):
        """True if the form differs from config.py (invalid input counts as a change)."""
        try:
            return bool(self.cfg.diff(self.collect()))
        except ValueError:
            return True

    def save(self):
        """Validates the form, confirms the changes, writes them into config.py and applies them to the app."""
        try:
            values = self.collect()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Setting", str(e))
            return
        changes = self.cfg.diff(values)
        if not changes:
            self.lbl_status.setText("No changes to save - the form matches config.py.")
            return
        listing = "\n".join(f"  • {c}" for c in changes)
        if QMessageBox.question(self, "Save to config.py",
                                f"Write {len(changes)} change(s) into config.py?\n\n{listing}\n\n"
                                "Only these values are changed in the file; comments and layout are kept, "
                                "and the current file is backed up as config.py.bak.",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) != QMessageBox.Yes:
            return
        try:
            self.cfg.save(values)
        except config_io.ConfigChangedError:
            if QMessageBox.question(self, "config.py Was Edited",
                                    "config.py was changed outside the app since it was loaded here.\n\n"
                                    "Reload it (your unsaved changes in this tab are discarded)? "
                                    "Then re-apply your changes and save again.",
                                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                self.cfg.reload()
                self.populate()
            return
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"config.py was not changed:\n\n{e}")
            return
        config_io.apply_to_module(config, copy.deepcopy(self.cfg.values))
        self.populate()
        self.lbl_status.setText(f"Saved {len(changes)} change(s) to config.py (backup: config.py.bak).")
        QMessageBox.information(
            self, "Settings Saved",
            f"{len(changes)} change(s) written to config.py.\n\n"
            "Values read while running (e.g. filter order, wavelengths and extinction coefficients, "
            "co-adaptation threshold, simulator layout, non-signal columns, experiment profiles) apply now. "
            "Fields already filled in other tabs (thresholds, cut-offs, the experiment list) and the "
            "helmet-template folder show the new defaults after restarting the app.")

    def reload(self):
        """Re-reads config.py, asking first if the form has unsaved changes."""
        if self.has_unsaved_changes() and QMessageBox.question(
                self, "Discard Changes?", "Discard the unsaved changes in this tab and reload config.py?",
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        self.cfg.reload()
        config_io.apply_to_module(config, copy.deepcopy(self.cfg.values))
        self.populate()

    def showEvent(self, event):
        """Picks up hand edits of config.py when the tab is shown (if the form has no unsaved changes)."""
        super().showEvent(event)
        if self.cfg.changed_on_disk():
            if self.has_unsaved_changes():
                self.lbl_status.setText("config.py was edited outside the app - press Reload to see those "
                                        "changes (your unsaved changes here would be discarded).")
            else:
                self.cfg.reload()
                config_io.apply_to_module(config, copy.deepcopy(self.cfg.values))
                self.populate()
                self.lbl_status.setText("config.py was edited outside the app - reloaded.")
