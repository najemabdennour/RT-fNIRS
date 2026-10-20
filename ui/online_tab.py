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

from ui.themes import THEMES
from core.simulators import SyntheticNIRSThread, SyntheticMarkerThread
from core.lsl_client import AdvancedStreamResolver, MarkerDiagnosticWorker, StreamAcquisitionWorker, release_inlets_async
from core.lsl_utils import resolve_channel_names


class OnlineStreamingTab(QWidget):
    """
    Live LSL monitor: resolves data/marker streams (optionally spawning
    synthetic simulators), records incoming samples into the main window's
    global_session_cache, and plots up to 15 selected channels with markers.

    Recording runs on a StreamAcquisitionWorker thread, so it never waits on
    the GUI. The GUI thread only redraws, on a ~10 Hz timer: axes are built
    once per channel selection/layout/theme, and each update blits just the
    data and marker lines over a cached background, re-rendering the whole
    figure only when an axis range has to change.
    """
    def __init__(self, parent_window):
        """
        Args:
            parent_window: The main window; provides global_session_cache and combo_theme.
        """
        super().__init__()
        self.parent_win = parent_window
        self.inlet = None
        self.marker_inlet = None
        self.stream_info = None
        self.channel_names = []
        self.selected_indices = []
        self.is_running = False
        self.max_display_points = 100
        self.acq_worker = None
        self._plot_axes = []        # (axes, line, channel index) per plotted channel
        self._marker_artists = []   # marker lines/labels redrawn on every update
        self._background = None     # cached render of everything except lines/markers (for blitting)
        self._rendered_count = -1   # sample count shown by the last plot update
        
        # Created before init_ui() because the canvas ribbon embeds it.
        self.combo_layout = QComboBox()
        self.combo_layout.addItems(["1-Column Stack", "2-Column Grid", "3-Column Grid"])
        self.combo_layout.currentTextChanged.connect(self.refresh_canvas_plots)
        
        # Refreshes counters/plots ~10x per second; update_loop() is a no-op until a stream is attached.
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(100)
        
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
        self.canvas.mpl_connect('draw_event', self._on_canvas_draw)
        canvas_layout.addWidget(self.canvas)
        splitter.addWidget(canvas_container)
        
        splitter.setSizes([280, 970])

    def start_stream_pipeline(self):
        """
        Stops any active stream, optionally starts the synthetic LSL
        simulators, then resolves the target streams on a background
        AdvancedStreamResolver thread.

        Starting clears the recording buffer, so if it holds samples that were
        never exported, the user is first asked to export, discard or cancel.
        """
        if not self._confirm_discard_unsaved():
            return
        self.stop_stream_pipeline()
        self.lbl_status.setText("Resolving LSL Targets...")
        self.btn_start.setEnabled(False)
        
        # Each Start begins a fresh recording: clear the shared buffer and its display state.
        self.parent_win.global_session_cache.reset()
        self.last_marker_count = 0
        self._rendered_count = -1
        self.lbl_pool_count.setText("Cache: 0 frames collected")
        self.lbl_last_marker.setText("Last Marker: [None]")
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

            self.acq_worker = StreamAcquisitionWorker(self.inlet, self.marker_inlet, self.parent_win.global_session_cache)
            self.acq_worker.error.connect(self._on_acquisition_error)
            self.acq_worker.start()

            self._rendered_count = -1
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

    def _on_acquisition_error(self, error):
        """Stops the stream and reports an inlet failure raised on the acquisition thread."""
        self.stop_stream_pipeline()
        QMessageBox.critical(self, "Data Link Fractured", f"Inlet drop Exception: {error}")

    def stop_stream_pipeline(self):
        """
        Stops plotting and recording, releases the inlets and stops any
        running simulators. Returns quickly: the inlets' slow liblsl teardown
        runs on a background thread.
        """
        self.is_running = False
        self.btn_stop.setEnabled(False)
        # Stop the worker first: it owns the inlets while running.
        if self.acq_worker:
            self.acq_worker.stop()
            self.acq_worker.inlet = self.acq_worker.marker_inlet = None
            self.acq_worker = None
        # Hand the inlets over (no references kept here) so their ~0.5 s liblsl
        # teardown runs on a background thread, not the GUI thread.
        inlets = [self.inlet, self.marker_inlet]
        self.inlet = None
        self.marker_inlet = None
        release_inlets_async(inlets)
        if hasattr(self, 'sim_nirs') and self.sim_nirs:
            self.sim_nirs.stop()
            self.sim_nirs = None
        if hasattr(self, 'sim_markers') and self.sim_markers:
            self.sim_markers.stop()
            self.sim_markers = None

        self.lbl_status.setText("Status: Halted")

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
        Timer slot (~10 Hz): when new samples have been recorded since the
        last update, refreshes the frame counter, the last-marker label and
        the plot lines. Recording itself happens on the acquisition thread.
        """
        if not self.is_running:
            return
        snapshot = self.parent_win.global_session_cache.snapshot(self.max_display_points)
        total = snapshot[3]
        if total == self._rendered_count:
            return
        self._rendered_count = total
        self.lbl_pool_count.setText(f"Cache: {total} frames collected")

        markers = snapshot[2]
        if len(markers) > self.last_marker_count:
            self.lbl_last_marker.setText(f"Last Marker: '{markers[-1][1]}'")
            self.last_marker_count = len(markers)

        self._update_plot_data(snapshot)

    def refresh_canvas_plots(self):
        """
        Rebuilds the plot axes for the current channel selection, layout and
        theme, then draws the latest data. Called on selection/layout/theme
        changes; routine updates go through update_loop() instead.
        """
        self._rebuild_plot_axes()
        self._update_plot_data()

    def _rebuild_plot_axes(self):
        """
        Creates one subplot and line per selected channel in the chosen
        column layout, with legend, grid, theme colors and tight layout.
        This is the expensive part of plotting, so it runs only when the
        figure's structure changes.
        """
        self.figure.clear()
        self._plot_axes = []
        self._marker_artists = []
        self._background = None
        if not self.selected_indices:
            return

        total_plots = len(self.selected_indices)
        layout_text = self.combo_layout.currentText()

        if "3-Column" in layout_text and total_plots > 3: cols, rows = 3, (total_plots + 2) // 3
        elif "2-Column" in layout_text and total_plots > 2: cols, rows = 2, (total_plots + 1) // 2
        else: cols, rows = 1, total_plots

        active_theme = THEMES[self.parent_win.combo_theme.currentText()]

        for p_idx, ch_idx in enumerate(self.selected_indices):
            ax = self.figure.add_subplot(rows, cols, p_idx + 1)
            ax.set_facecolor(active_theme["bg"])
            line, = ax.plot([], [], label=self.channel_names[ch_idx], color=f"C{p_idx % 10}", lw=1.2, animated=True)
            ax.legend(loc="upper right", fontsize=7)
            ax.tick_params(labelsize=7, colors=active_theme["fg"])
            ax.grid(True, linestyle=":", alpha=0.4)
            if (p_idx >= (rows - 1) * cols) or (cols == 1 and p_idx == total_plots - 1):
                ax.set_xlabel("Time (s)", fontsize=8, color=active_theme["fg"])
            self._plot_axes.append((ax, line, ch_idx))

        self.figure.tight_layout()

    def _update_plot_data(self, snapshot=None):
        """
        Puts the last max_display_points samples into the existing plot
        lines and redraws the marker lines (and, on the first plot, their
        labels) that fall inside the visible window.

        Blits only these animated artists over the cached background; a full
        redraw (which refreshes the background) happens only when an axis
        range had to change or no background is cached yet.

        Args:
            snapshot: a DataSession.snapshot() result, or None to take one.
        """
        if not self._plot_axes:
            self.canvas.draw_idle()
            return
        samples, times, markers, _ = snapshot or self.parent_win.global_session_cache.snapshot(self.max_display_points)
        if not samples:
            self.canvas.draw_idle()
            return

        base_t = times[0]
        x_axis = [t - base_t for t in times]
        for artist in self._marker_artists:
            artist.remove()
        self._marker_artists = []
        visible_markers = [(m_time, m_label) for m_time, m_label in markers[-50:] if times[0] <= m_time <= times[-1]]

        limits_changed = False
        for p_idx, (ax, line, ch_idx) in enumerate(self._plot_axes):
            y_data = [frame[ch_idx] for frame in samples]
            line.set_data(x_axis, y_data)
            limits_changed |= self._fit_axis_limits(ax, x_axis[-1], len(x_axis), self.max_display_points, y_data)
            for m_time, m_label in visible_markers:
                rel_x = m_time - base_t
                self._marker_artists.append(ax.axvline(x=rel_x, color="#FF3366", linestyle="--", alpha=0.7,
                                                       lw=1.0, animated=True))
                if p_idx == 0:
                    self._marker_artists.append(ax.text(rel_x, ax.get_ylim()[1], f" {m_label}", color="#FF3366",
                                                        fontsize=7, verticalalignment='top', animated=True))

        if limits_changed or self._background is None:
            self.canvas.draw_idle()  # _on_canvas_draw re-caches the background and draws the lines
        else:
            self.canvas.restore_region(self._background)
            self._draw_animated_artists()
            self.canvas.blit(self.figure.bbox)

    @staticmethod
    def _fit_axis_limits(ax, x_max, n_shown, n_window, y_data):
        """
        Adjusts an axis's limits only when the data no longer fits them well,
        so most updates can be blitted without re-rendering ticks and labels.

        x spans 0 to the full display window's duration, estimated from the
        samples shown so far (x_max over n_shown of n_window samples), with
        5% headroom - so it stays fixed while the window fills up. y gets a
        25% margin and is only reset when the data leaves the current range
        or shrinks below 25% of it, so noise and passing spikes rarely force
        a full re-render. Returns True if either limit changed.
        """
        changed = False
        x_lo, x_hi = ax.get_xlim()
        window = x_max * (n_window - 1) / (n_shown - 1) if n_shown > 1 else 1.0
        if x_lo != 0 or x_max > x_hi or window > x_hi or window < 0.8 * x_hi:
            ax.set_xlim(0, max(window, 1e-6) * 1.05)
            changed = True

        finite = [v for v in y_data if v == v and abs(v) != float('inf')]
        if finite:
            y_min, y_max = min(finite), max(finite)
            span = (y_max - y_min) or max(abs(y_max), 1.0) * 1e-3
            lo, hi = ax.get_ylim()
            if y_min < lo or y_max > hi or span < 0.25 * (hi - lo):
                ax.set_ylim(y_min - 0.25 * span, y_max + 0.25 * span)
                changed = True
        return changed

    def _draw_animated_artists(self):
        """Renders the animated data lines and marker artists onto the canvas buffer."""
        for ax, line, _ in self._plot_axes:
            ax.draw_artist(line)
        for artist in self._marker_artists:
            artist.axes.draw_artist(artist)

    def _on_canvas_draw(self, event):
        """
        After every full draw (also on resize or theme change), caches the
        static background for blitting and draws the animated artists, which
        a full draw skips.
        """
        if not self._plot_axes:
            self._background = None
            return
        self._background = self.canvas.copy_from_bbox(self.figure.bbox)
        self._draw_animated_artists()

    def _confirm_discard_unsaved(self):
        """
        Asks before Start clears samples that were never exported.

        Offers Export... (save first; proceeds only if the export succeeds),
        Discard (proceed without saving) and Cancel (keep everything, including
        a running stream). Returns True if starting may proceed. Returns True
        without asking when there is nothing unsaved.
        """
        unsaved = self.parent_win.global_session_cache.unsaved_count()
        if unsaved <= 0:
            return True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Discard Unsaved Recording?")
        box.setText(f"{unsaved} recorded sample(s) have not been exported.")
        box.setInformativeText("Starting a new recording clears the current one. Export it first?")
        btn_export = box.addButton("Export...", QMessageBox.AcceptRole)
        btn_discard = box.addButton("Discard", QMessageBox.DestructiveRole)
        btn_cancel = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(btn_export)
        box.setEscapeButton(btn_cancel)
        box.exec_()

        clicked = box.clickedButton()
        if clicked is btn_export:
            return self.save_recorded_dataset()
        return clicked is btn_discard

    def save_recorded_dataset(self):
        """
        Prompts for a path and exports the session cache to CSV. Recording
        continues in the background meanwhile; the file holds everything
        recorded up to the moment it is written.

        Returns True if a file was written; False if there was nothing to
        export, the dialog was cancelled, or writing failed (reported).
        """
        if not len(self.parent_win.global_session_cache):
            return False

        suggest_fn = f"lsl_session_{datetime.date.today()}_{time.strftime('%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export Live Stream Data", suggest_fn, "Comma Separated (*.csv)")
        if not path:
            return False
        try:
            saved = self.parent_win.global_session_cache.export_to_csv(path, self.channel_names)
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not write the recording to:\n{path}\n\n{e}")
            return False
        if saved:
            QMessageBox.information(self, "Export Saved", f"File written successfully:\n{path}")
        return saved