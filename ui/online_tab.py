# ui/online_tab.py
"""
Online streaming monitor tab: connects to live LSL data/marker streams,
plots selected channels in real time and exports the recorded session.
"""
import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QGroupBox, QLineEdit, QFileDialog, 
                             QMessageBox, QListWidget, QAbstractItemView, 
                             QComboBox, QSplitter, QCheckBox, QGridLayout)
from PyQt5.QtCore import QTimer, Qt
from pylsl import StreamInlet

from core.data_manager import DataSession
from ui.themes import THEMES
from core.simulators import SyntheticNIRSThread, SyntheticMarkerThread
from core.lsl_client import AdvancedStreamResolver, MarkerDiagnosticWorker
from core.lsl_utils import resolve_channel_names


class OnlineStreamingTab(QWidget):
    """
    Live LSL monitor: resolves data/marker streams (optionally spawning
    synthetic simulators), appends incoming samples to the main window's
    global_session_cache, and plots up to 15 selected channels with markers.
    """
    def __init__(self, parent_window):
        """
        Args:
            parent_window: The main window; provides global_session_cache and combo_theme.
        """
        super().__init__()
        self.parent_win = parent_window
        self.session = DataSession()
        self.inlet = None
        self.marker_inlet = None
        self.stream_info = None
        self.channel_names = []
        self.selected_indices = []
        self.is_running = False
        self.max_display_points = 100
        
        # Created before init_ui() because the canvas ribbon embeds it.
        self.combo_layout = QComboBox()
        self.combo_layout.addItems(["1-Column Stack", "2-Column Grid", "3-Column Grid"])
        self.combo_layout.currentTextChanged.connect(self.refresh_canvas_plots)
        
        # Polls the inlet every 30 ms; update_loop() is a no-op until a stream is attached.
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(30)
        
        self.init_ui()

    def init_ui(self):
        """Builds the settings/control/channel-selection sidebar and the plot canvas."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Sidebar: hardware settings, session controls, channel selection.
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(6, 6, 6, 6)
        sidebar_layout.setSpacing(4)
        
        config_group = QGroupBox("Hardware & Pipeline Settings")
        config_layout = QGridLayout(config_group)
        config_layout.addWidget(QLabel("Target Signal:"), 0, 0)
        self.combo_sig_type = QComboBox()
        self.combo_sig_type.addItems(["NIRS", "EEG", "MEG"])
        config_layout.addWidget(self.combo_sig_type, 0, 1)
        
        self.chk_markers = QCheckBox("Enable Marker Input Tracking")
        self.chk_markers.setChecked(True)
        config_layout.addWidget(self.chk_markers, 1, 0, 1, 2)
        
        config_layout.addWidget(QLabel("Marker Name:"), 2, 0)
        self.ent_marker_name = QLineEdit("OpenSesame_Markers")
        config_layout.addWidget(self.ent_marker_name, 2, 1)

        self.btn_ping_marker = QPushButton("🔍 Ping Marker Stream")
        self.btn_ping_marker.setStyleSheet("background-color: #4a4a4a; color: white;")
        self.btn_ping_marker.clicked.connect(self.run_marker_diagnostics)
        config_layout.addWidget(self.btn_ping_marker, 3, 0)

        self.lbl_diag_result = QLabel("<i>Unverified</i>")
        self.lbl_diag_result.setStyleSheet("color: #b0b0b0; font-size: 10px;")
        config_layout.addWidget(self.lbl_diag_result, 3, 1)

        self.chk_simulate = QCheckBox("Enable LSL Simulators (Dev Mode)")
        config_layout.addWidget(self.chk_simulate, 4, 0, 1, 2)

        sidebar_layout.addWidget(config_group)
        
        ctrl_group = QGroupBox("Session Controls")
        ctrl_layout = QVBoxLayout(ctrl_group)
        self.btn_start = QPushButton("Initialize LSL Stream")
        self.btn_start.clicked.connect(self.start_stream_pipeline)
        self.btn_stop = QPushButton("Stop LSL Stream")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_stream_pipeline)
        self.btn_save = QPushButton("Export Dataset (CSV)")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_recorded_dataset)
        
        self.lbl_status = QLabel("Status: Offline")
        self.lbl_status.setStyleSheet("font-weight: bold; color: gray;")
        self.lbl_pool_count = QLabel("Cache: 0 frames collected")
        
        self.lbl_last_marker = QLabel("Last Marker: [None]")
        self.lbl_last_marker.setStyleSheet("font-weight: bold; color: #ffaa00;")
        self.last_marker_count = 0 

        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.btn_save)
        ctrl_layout.addWidget(self.lbl_status)
        ctrl_layout.addWidget(self.lbl_pool_count)
        sidebar_layout.addWidget(ctrl_group)
        
        chan_group = QGroupBox("Active Channel Monitor Selection")
        chan_layout = QVBoxLayout(chan_group)
        self.lbl_selected_count = QLabel("Active Plots: 0 / 15")
        chan_layout.addWidget(self.lbl_selected_count)
        
        self.search_filter = QLineEdit()
        self.search_filter.setPlaceholderText("Filter channel names...")
        self.search_filter.textChanged.connect(self.filter_channel_list)
        chan_layout.addWidget(self.search_filter)
        
        self.channel_list_widget = QListWidget()
        self.channel_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.channel_list_widget.itemSelectionChanged.connect(self.on_channel_selection_updated)
        chan_layout.addWidget(self.channel_list_widget)
        sidebar_layout.addWidget(chan_group)
        
        splitter.addWidget(sidebar)
        
        # Right panel: layout selector ribbon above the real-time plot canvas.
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(2, 2, 2, 2)
        
        canvas_ribbon = QHBoxLayout()
        canvas_ribbon.addWidget(QLabel("<b>Plot Formatting Matrix Layout:</b>"))
        canvas_ribbon.addWidget(self.combo_layout)
        canvas_ribbon.addStretch()
        canvas_layout.addLayout(canvas_ribbon)
        
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        canvas_layout.addWidget(self.canvas)
        splitter.addWidget(canvas_container)
        
        splitter.setSizes([280, 970])

    def start_stream_pipeline(self):
        """
        Stops any active stream, optionally starts the synthetic LSL
        simulators, then resolves the target streams on a background
        AdvancedStreamResolver thread.
        """
        self.stop_stream_pipeline()
        self.lbl_status.setText("Resolving LSL Targets...")
        self.btn_start.setEnabled(False)
        
        self.session.reset()
        self.channel_list_widget.clear()
        
        if self.chk_simulate.isChecked():
            self.sim_nirs = SyntheticNIRSThread(
                name='LUMO_Synthetic' if self.combo_sig_type.currentText() == "NIRS" else 'Synthetic_Signal',
                n_channels=200 # Large enough to stress-test, small enough to keep UI lists responsive
            )
            self.sim_markers = SyntheticMarkerThread(self.ent_marker_name.text())
            
            self.sim_nirs.start()
            if self.chk_markers.isChecked():
                self.sim_markers.start()
            
            # Give the LSL network 500ms to broadcast the new synthetic outlets
            time.sleep(0.5) 

        self.resolver_worker = AdvancedStreamResolver(
            self.combo_sig_type.currentText(),
            self.chk_markers.isChecked(),
            self.ent_marker_name.text()
        )
        self.resolver_worker.success.connect(self._on_resolution_success)
        self.resolver_worker.failure.connect(self._on_resolution_failure)
        self.resolver_worker.start()

    def _on_resolution_success(self, primary_info, marker_info, status_text):
        """
        Opens inlets for the resolved streams, populates the channel list
        (pre-selecting the first four) and starts live plotting.

        Channel names come from inlet.info(), which carries the full XML
        description that the resolver's discovery record lacks.
        """
        try:
            self.inlet = StreamInlet(primary_info)
            self.stream_info = self.inlet.info()
            if marker_info: 
                self.marker_inlet = StreamInlet(marker_info)
            
            self.channel_names = resolve_channel_names(self.stream_info)
            
            self.channel_list_widget.blockSignals(True)
            for idx, label in enumerate(self.channel_names):
                self.channel_list_widget.addItem(label)
                self.channel_list_widget.item(idx).setData(Qt.UserRole, idx)
            for i in range(min(4, len(self.channel_names))):
                self.channel_list_widget.item(i).setSelected(True)
            self.channel_list_widget.blockSignals(False)
            
            self.on_channel_selection_updated()
            self.is_running = True
            self.lbl_status.setText(status_text)
            self.btn_stop.setEnabled(True)
            self.btn_save.setEnabled(True)
            self.btn_start.setEnabled(True)
        except Exception as e:
            self._on_resolution_failure(str(e))

    def _on_resolution_failure(self, error):
        """Resets the status/start button and shows the resolution error."""
        self.lbl_status.setText("Resolution Error")
        self.btn_start.setEnabled(True)
        QMessageBox.critical(self, "LSL Handshake Failure", f"Pipeline couldn't attach:\n{error}")

    def stop_stream_pipeline(self):
        """Stops plotting, closes the inlets and stops any running simulators."""
        self.is_running = False
        if self.inlet:
            try: self.inlet.close_stream()
            except: pass
            self.inlet = None
        if self.marker_inlet:
            try: self.marker_inlet.close_stream()
            except: pass
            self.marker_inlet = None
        if hasattr(self, 'sim_nirs') and self.sim_nirs:
            self.sim_nirs.stop()
            self.sim_nirs = None
        if hasattr(self, 'sim_markers') and self.sim_markers:
            self.sim_markers.stop()
            self.sim_markers = None

        self.lbl_status.setText("Status: Halted")
        self.btn_stop.setEnabled(False)

    def filter_channel_list(self, pattern):
        """Hides channel list entries whose name doesn't contain pattern (case-insensitive)."""
        for i in range(self.channel_list_widget.count()):
            item = self.channel_list_widget.item(i)
            item.setHidden(pattern.lower() not in item.text().lower())

    def on_channel_selection_updated(self):
        """Caps the selection at 15 channels, records their indices and redraws the plots."""
        chosen = self.channel_list_widget.selectedItems()
        if len(chosen) > 15:
            self.channel_list_widget.blockSignals(True)
            for item in chosen[15:]: item.setSelected(False)
            self.channel_list_widget.blockSignals(False)
            chosen = chosen[:15]
        self.selected_indices = [item.data(Qt.UserRole) for item in chosen]
        self.lbl_selected_count.setText(f"Active Plots: {len(self.selected_indices)} / 15")
        self.refresh_canvas_plots()

    def run_marker_diagnostics(self):
        """Pings the named marker stream on a background MarkerDiagnosticWorker thread."""
        marker_target = self.ent_marker_name.text().strip()
        if not marker_target:
            return
            
        self.btn_ping_marker.setEnabled(False)
        self.btn_ping_marker.setText("Pinging...")
        self.lbl_diag_result.setText("<i>Searching network...</i>")
        self.lbl_diag_result.setStyleSheet("color: #ffaa00; font-size: 10px;")

        self.diag_worker = MarkerDiagnosticWorker(marker_target)
        self.diag_worker.diagnostic_success.connect(self._on_diag_success)
        self.diag_worker.diagnostic_failure.connect(self._on_diag_failure)
        self.diag_worker.start()

    def _on_diag_success(self, data):
        """
        Shows the found marker stream's hostname and truncated UID, so the
        operator can verify it is the expected (fresh) outlet.
        """
        self.lbl_diag_result.setText(f"<b>Host:</b> {data['hostname']}<br><b>UID:</b> {data['uid'][:8]}...")
        self.lbl_diag_result.setStyleSheet("color: #00ff66; font-size: 10px;")
        self.btn_ping_marker.setText("🔍 Ping Marker Stream")
        self.btn_ping_marker.setEnabled(True)

    def _on_diag_failure(self, error_msg):
        """Marks the marker stream as not found and re-enables the ping button."""
        self.lbl_diag_result.setText("<i>Not Found (Check Network/Name)</i>")
        self.lbl_diag_result.setStyleSheet("color: #ff3333; font-size: 10px;")
        self.btn_ping_marker.setText("🔍 Ping Marker Stream")
        self.btn_ping_marker.setEnabled(True)

    def update_loop(self):
        """
        Timer slot: drains all pending data samples into the global session
        cache (which also pulls markers), then updates the counters and plots.
        Stops the stream on any inlet error.
        """
        if self.is_running and self.inlet:
            try:
                new_frames = False
                while True:
                    sample, timestamp = self.inlet.pull_sample(timeout=0.0)
                    if sample is None: 
                        break
                    self.parent_win.global_session_cache.process_and_append(sample, timestamp, self.marker_inlet)
                    new_frames = True
                if new_frames:
                    cache = self.parent_win.global_session_cache
                    self.lbl_pool_count.setText(f"Cache: {len(cache.samples)} frames collected")
                    
                    current_marker_count = len(cache.cached_visual_markers)
                    if current_marker_count > self.last_marker_count:
                        last_time, last_trigger = cache.cached_visual_markers[-1]
                        self.lbl_last_marker.setText(f"Last Marker: '{last_trigger}'")
                        self.last_marker_count = current_marker_count

                    self.refresh_canvas_plots()

            except Exception as e:
                self.stop_stream_pipeline()
                QMessageBox.critical(self, "Data Link Fractured", f"Inlet drop Exception: {e}")

    def refresh_canvas_plots(self):
        """
        Redraws the selected channels over the last max_display_points
        samples in the chosen column layout, overlaying markers that fall
        inside the visible window.
        """
        self.figure.clear()
        cache = self.parent_win.global_session_cache
        if not cache.samples or not self.selected_indices:
            self.canvas.draw_idle()
            return
            
        view_samples = cache.samples[-self.max_display_points:]
        view_times = cache.timestamps[-self.max_display_points:]
        base_t = view_times[0] if view_times else 0
        x_axis = [t - base_t for t in view_times]
        
        total_plots = len(self.selected_indices)
        layout_text = self.combo_layout.currentText()
        
        if "3-Column" in layout_text and total_plots > 3: cols, rows = 3, (total_plots + 2) // 3
        elif "2-Column" in layout_text and total_plots > 2: cols, rows = 2, (total_plots + 1) // 2
        else: cols, rows = 1, total_plots
            
        active_theme = THEMES[self.parent_win.combo_theme.currentText()]
        
        for p_idx, ch_idx in enumerate(self.selected_indices):
            ax = self.figure.add_subplot(rows, cols, p_idx + 1)
            ax.set_facecolor(active_theme["bg"])
            
            y_data = [frame[ch_idx] for frame in view_samples]
            ax.plot(x_axis, y_data, label=self.channel_names[ch_idx], color=f"C{p_idx % 10}", lw=1.2)
            
            for m_time, m_label in cache.cached_visual_markers[-50:]:
                if view_times[0] <= m_time <= view_times[-1]:
                    rel_x = m_time - base_t
                    ax.axvline(x=rel_x, color="#FF3366", linestyle="--", alpha=0.7, lw=1.0)
                    if p_idx == 0: ax.text(rel_x, ax.get_ylim()[1], f" {m_label}", color="#FF3366", fontsize=7, verticalalignment='top')
            
            ax.legend(loc="upper right", fontsize=7)
            ax.tick_params(labelsize=7, colors=active_theme["fg"])
            ax.grid(True, linestyle=":", alpha=0.4)
            if (p_idx >= (rows - 1) * cols) or (cols == 1 and p_idx == total_plots - 1):
                ax.set_xlabel("Time (s)", fontsize=8, color=active_theme["fg"])
                
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def save_recorded_dataset(self):
        """Pauses stream polling while prompting for a path, then exports the session cache to CSV."""
        if not self.parent_win.global_session_cache.samples: 
            return
        checkpoint = self.is_running
        self.is_running = False
        
        suggest_fn = f"lsl_session_{datetime.date.today()}_{time.strftime('%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export Live Stream Data", suggest_fn, "Comma Separated (*.csv)")
        if path:
            if self.parent_win.global_session_cache.export_to_csv(path, self.channel_names):
                QMessageBox.information(self, "Export Saved", f"File written successfully:\n{path}")
        self.is_running = checkpoint