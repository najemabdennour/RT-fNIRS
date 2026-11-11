# ui/montage_view.py
"""2D montage layout view (head, tiles, empty helmet slots, optodes, channels) with layout table export."""
import os
import re
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
                             QDoubleSpinBox, QCheckBox, QFileDialog, QMessageBox, QSizePolicy, QToolTip)
from scipy.spatial import ConvexHull

import config
from core import helmet_templates as ht
from core.io_utils import read_csv, write_csv
from core.montage import project_2d, montage_table, montage_from_table, head_outline

SOURCE_COLOR = "#e74c3c"
DETECTOR_COLOR = "#3498db"
AUTO_TEMPLATE = "Auto (from stream)"
NO_TEMPLATE = "None"


class MontageView(QWidget):
    """
    Draws a montage (core.montage.Montage) as a flat 2D layout inside a head
    outline: each tile shaded in its own color and named, sources and
    detectors, a line per source-detector pair up to a chosen length, and -
    with a helmet template (core.helmet_templates) - the helmet's empty slots
    as dashed outlines. Channels currently plotted in the live view are
    highlighted in their plot colors; hovering shows optode/channel/slot details.

    The view frames all tiles and slots by default (optionally the full
    head), with marker and label sizes scaled to the optode spacing, and a
    toolbar for zooming/panning. A saved layout table can be loaded for
    streams whose metadata has no optode positions.

    The layout is drawn once per montage or option change, never per
    sample, so it adds no cost while recording. Changes made while the view
    is hidden (e.g. on another tab) are drawn when it is next shown.
    """

    def __init__(self, theme_getter, parent=None, template_dir=None):
        """
        Args:
            theme_getter: callable returning the active theme dict (ui.themes.THEMES entry).
            template_dir: helmet template folder (default config.HELMET_TEMPLATE_DIR).
        """
        super().__init__(parent)
        self.theme_getter = theme_getter
        self.template_dir = template_dir or config.HELMET_TEMPLATE_DIR
        self.montage = None
        self.positions = {}
        self.method = 'none'
        self.highlight = []      # channel indices of the live plots, in plot order
        self.template = None     # helmet template in use (dict) or None
        self.tile_slots = {}     # tile -> template slot id
        self.empty_points = {}   # 'slot<id>/<optode>' -> 3D position of empty slots
        self.head = None         # (centre_xy, radius, approximate) or None
        self._template_note = ""
        self._hover = None       # hit-test data for tooltips, built by redraw()
        self._message = "Connect to a stream to see its channel and tile layout."
        self._dirty = True
        self._listeners = []     # callables run after the montage, template or highlights change
        self.init_ui()

    def init_ui(self):
        """Builds the two control ribbons, system/summary lines, navigation toolbar and canvas."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        ribbon = QHBoxLayout()
        ribbon.addWidget(QLabel("<b>Max channel length (mm):</b>"))
        self.spin_max_length = QDoubleSpinBox()
        self.spin_max_length.setRange(1, 500)
        self.spin_max_length.setValue(30)
        self.spin_max_length.setSingleStep(5)
        self.spin_max_length.setToolTip("Only source-detector pairs up to this separation are drawn.")
        self.spin_max_length.valueChanged.connect(self.redraw)
        ribbon.addWidget(self.spin_max_length)

        self.chk_tile_labels = QCheckBox("Tile names")
        self.chk_tile_labels.setChecked(True)
        self.chk_tile_labels.toggled.connect(self.redraw)
        ribbon.addWidget(self.chk_tile_labels)

        self.chk_optode_labels = QCheckBox("Optode names")
        self.chk_optode_labels.setToolTip("Short optode ids (e.g. 'A', '1'); the tile name gives the rest.")
        self.chk_optode_labels.toggled.connect(self.redraw)
        ribbon.addWidget(self.chk_optode_labels)

        self.chk_full_head = QCheckBox("Full head")
        self.chk_full_head.setToolTip("Show the whole head instead of framing the tiles and helmet slots.")
        self.chk_full_head.toggled.connect(self.redraw)
        ribbon.addWidget(self.chk_full_head)

        ribbon.addStretch()
        self.btn_load_layout = QPushButton("Load Layout File...")
        self.btn_load_layout.setToolTip("Apply a saved layout table (CSV) to this stream's channels, "
                                        "e.g. when the stream metadata has no optode positions.")
        self.btn_load_layout.clicked.connect(self.load_layout_file)
        ribbon.addWidget(self.btn_load_layout)
        self.btn_save_table = QPushButton("Save Layout Table (CSV)...")
        self.btn_save_table.setEnabled(False)
        self.btn_save_table.clicked.connect(self.save_table)
        ribbon.addWidget(self.btn_save_table)
        layout.addLayout(ribbon)

        template_ribbon = QHBoxLayout()
        template_ribbon.addWidget(QLabel("<b>Helmet template:</b>"))
        self.combo_template = QComboBox()
        self.combo_template.setToolTip(
            "Helmet template drawing the empty slots. 'Auto' picks the template named after the "
            "stream's cap id (e.g. GA00396).")
        self.combo_template.setMinimumWidth(180)
        self.combo_template.activated.connect(lambda _i: self._apply_template())
        template_ribbon.addWidget(self.combo_template)

        self.chk_auto_learn = QCheckBox("Learn slots from each stream")
        self.chk_auto_learn.setChecked(True)
        self.chk_auto_learn.setToolTip(
            "Adds the positions of every connected stream's tiles to its helmet template, so the "
            "template fills up as different slots are used across sessions.")
        template_ribbon.addWidget(self.chk_auto_learn)

        self.btn_save_template = QPushButton("Save as Helmet Template...")
        self.btn_save_template.setEnabled(False)
        self.btn_save_template.setToolTip(
            "Replaces the helmet template with exactly the current tiles. Use it with every slot "
            "of the helmet populated, to record the complete helmet once.")
        self.btn_save_template.clicked.connect(self.save_as_template)
        template_ribbon.addWidget(self.btn_save_template)
        template_ribbon.addStretch()
        layout.addLayout(template_ribbon)

        self.lbl_system = QLabel("")
        self.lbl_system.setStyleSheet("font-weight: bold; font-size: 11px;")
        self.lbl_system.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.lbl_system)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet("color: #b0b0b0; font-size: 10px;")
        self.lbl_summary.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.lbl_summary)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)
        self.canvas.mpl_connect('motion_notify_event', self._on_hover)

        # Marker/label sizes depend on the canvas size: redraw shortly after a resize settles.
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(250)
        self._resize_timer.timeout.connect(self.redraw)
        self.canvas.mpl_connect('resize_event', lambda event: self._resize_timer.start())

        self._refresh_template_list()

    # --- Montage and template state ---

    def add_listener(self, callback):
        """Registers a callable run (without arguments) whenever the montage, template or highlights change."""
        self._listeners.append(callback)

    def _notify(self):
        """Runs the registered change listeners (e.g. the 3D layout view)."""
        for callback in self._listeners:
            callback()

    def reset(self, message):
        """Clears the montage and shows `message` instead of a layout."""
        self.montage = None
        self.positions, self.method = {}, 'none'
        self.highlight = []
        self.template, self.tile_slots, self.empty_points, self.head = None, {}, {}, None
        self._message = message
        self.btn_save_table.setEnabled(False)
        self.btn_save_template.setEnabled(False)
        self.lbl_system.setText("")
        self.lbl_summary.setText("")
        self.redraw()
        self._notify()

    def set_montage(self, montage, source=""):
        """
        Shows a new montage (or a message if it is None / has no channels):
        resolves its helmet template, learns its tiles into it when enabled,
        projects tiles and empty slots to 2D, and enables export.
        """
        if montage is None or not montage.channels:
            self.reset("This stream's metadata describes no channels.")
            return
        self.montage = montage
        self.highlight = []
        self._layout_source = source
        self.btn_save_table.setEnabled(True)
        self.btn_save_template.setEnabled(bool(montage.tiles()) and ht.template_key(montage) is not None)
        if self.chk_auto_learn.isChecked():
            self._learn_into_template()
        self._refresh_template_list()
        self._apply_template()

    def _learn_into_template(self):
        """Option B: adds the montage's tiles to its helmet template on disk (created if needed)."""
        key = ht.template_key(self.montage)
        if key is None or not self.montage.tiles():
            return
        try:
            template = ht.load_template(self.template_dir, key) or ht.new_template(key)
            before = len(template['slots'])
            added = ht.merge_montage(template, self.montage, self.montage.system.get('stream_name', ''))
            if added or before == 0:
                ht.save_template(self.template_dir, template)
        except Exception as e:
            QMessageBox.warning(self, "Helmet Template Not Updated",
                                f"Could not update the helmet template '{key}':\n{e}")

    def _refresh_template_list(self):
        """Fills the template selector with Auto, every saved template and None, keeping the choice."""
        current = self.combo_template.currentText()
        self.combo_template.blockSignals(True)
        self.combo_template.clear()
        self.combo_template.addItems([AUTO_TEMPLATE] + ht.list_templates(self.template_dir) + [NO_TEMPLATE])
        index = self.combo_template.findText(current)
        self.combo_template.setCurrentIndex(index if index >= 0 else 0)
        self.combo_template.blockSignals(False)

    def _apply_template(self):
        """Loads the selected template, matches the tiles to its slots, then reprojects and redraws."""
        if self.montage is None:
            return
        choice = self.combo_template.currentText()
        key = ht.template_key(self.montage) if choice == AUTO_TEMPLATE else (None if choice == NO_TEMPLATE else choice)
        self.template, self.tile_slots, self.empty_points = None, {}, {}
        self._template_note = ""
        if key:
            try:
                self.template = ht.load_template(self.template_dir, key)
            except Exception as e:
                self._template_note = f" - template '{key}' unreadable: {e}"
            if self.template is not None:
                self.tile_slots = ht.match_tiles(self.template, self.montage)
                self.empty_points = ht.empty_slot_points(self.template, self.tile_slots)
                kind = "complete helmet" if self.template.get('reference') else "learned from streams"
                unmatched = len(self.montage.tiles()) - len(self.tile_slots)
                self._template_note = (f" - helmet '{key}' ({kind}): {len(self.tile_slots)}/"
                                       f"{len(self.template['slots'])} slots used"
                                       + (f", {unmatched} tile(s) not in the template" if unmatched else ""))
            elif choice == AUTO_TEMPLATE:
                self._template_note = f" - no helmet template '{key}' yet"
        self.positions, self.method = project_2d(self.montage, self.empty_points)
        self.head = head_outline(self.montage, self.positions, self.method, self.empty_points)
        self._update_summary()
        self.redraw()
        self._notify()

    def _update_summary(self):
        """Writes the system line, and the montage counts, layout origin and template status line."""
        projection = {'azimuthal': "flattened from 3D positions about the vertex (nose up)",
                      'saved': "from the saved 2D layout (nose up)",
                      'sphere': "flattened from 3D positions about the montage centre (no head fiducials)",
                      'none': "no optode positions - table only; use Load Layout File... to add them"}[self.method]
        self.lbl_system.setText(self.montage.system_summary())
        self.lbl_summary.setText(f"{self.montage.summary()} - layout {projection}"
                                 f"{getattr(self, '_layout_source', '')}{self._template_note}.")

    def save_as_template(self):
        """
        Option A: replaces the helmet template with exactly the current tiles,
        after confirming (and warning if slots of the existing template would be lost).
        """
        key = ht.template_key(self.montage) if self.montage is not None else None
        if key is None:
            return
        n_tiles = len(self.montage.tiles())
        existing = ht.load_template(self.template_dir, key)
        text = f"Save the {n_tiles} tiles of this stream as the complete helmet template '{key}'?"
        if existing is not None:
            lost = len(existing['slots']) - len(ht.match_tiles(existing, self.montage))
            text += (f"\n\nThis replaces the existing template ({len(existing['slots'])} slots)."
                     + (f" {lost} of its slots are not occupied now and will be removed." if lost > 0 else ""))
        if QMessageBox.question(self, "Save Helmet Template", text,
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            path = ht.save_template(self.template_dir, ht.from_montage(key, self.montage,
                                                                        self.montage.system.get('stream_name', '')))
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save the helmet template:\n{e}")
            return
        self._refresh_template_list()
        self._apply_template()
        QMessageBox.information(self, "Helmet Template Saved", f"{n_tiles} slots saved to:\n{path}")

    def load_layout_file(self):
        """
        Applies a saved layout table (core.montage.montage_table CSV) to the
        current stream: its channels keep the stream's names and order and take
        sources, detectors, tiles, distances and positions from matching rows.
        Without a connected stream, the table is shown on its own.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Load Layout Table", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            table = read_csv(path)
            names = [ch['name'] for ch in self.montage.channels] if self.montage is not None else None
            montage = montage_from_table(table, names)
            if self.montage is not None and not montage.system:
                montage.system = dict(self.montage.system)
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Could not read the layout table:\n{path}\n\n{e}")
            return
        matched = sum(ch['source'] is not None for ch in montage.channels)
        if not matched:
            QMessageBox.warning(self, "No Matching Channels",
                                "None of this stream's channel names appear in the layout table's 'channel' column.")
            return
        highlight = self.highlight
        self.set_montage(montage, f" - loaded from {os.path.basename(path)} "
                                  f"({matched}/{len(montage.channels)} channels matched)")
        self.set_highlighted_channels(highlight)

    def set_highlighted_channels(self, channel_indices):
        """Highlights the optode pairs of these stream channel indices (the live plot selection)."""
        self.highlight = list(channel_indices)
        if self.montage is not None and self.positions:
            self.redraw()
        self._notify()

    # --- Drawing ---

    def showEvent(self, event):
        """Draws any change made while the view was hidden."""
        super().showEvent(event)
        if self._dirty:
            self.redraw()

    def show_message(self, text):
        """Replaces the drawing with a centred message."""
        theme = self.theme_getter()
        self._hover = None
        self.figure.clear()
        self.figure.patch.set_facecolor(theme["bg"])
        self.figure.text(0.5, 0.5, text, ha='center', va='center', color=theme["fg"], fontsize=10, wrap=True)
        self.canvas.draw_idle()

    def redraw(self):
        """
        Draws the head, tiles, empty slots, channel lines, optodes and
        highlighted channels (or the current message). Deferred until shown if hidden.
        """
        if not self.isVisible():
            self._dirty = True
            return
        self._dirty = False
        if self.montage is None:
            self.show_message(self._message)
            return
        if not self.positions:
            self.show_message("This stream's metadata has no optode positions, so no 2D layout can be drawn.\n"
                              "Use 'Load Layout File...' to apply a saved layout table.\n"
                              "The layout table (channels, tiles, distances) can still be saved.")
            return

        theme = self.theme_getter()
        fg = theme["fg"]
        self.figure.clear()
        self.figure.patch.set_facecolor(theme["bg"])
        # Room on the right for the legend, so it never covers the montage.
        ax = self.figure.add_axes([0.01, 0.02, 0.80, 0.92])
        ax.set_facecolor(theme["bg"])
        ax.set_aspect('equal')
        ax.axis('off')
        self._hover = {'ax': ax, 'points': [], 'segments': [], 'slots': []}

        optodes = [lbl for lbl in self.montage.probes if lbl in self.positions]
        empty = [lbl for lbl in self.empty_points if lbl in self.positions]
        xy_all = np.array([self.positions[lbl] for lbl in optodes + empty])
        spacing = self._typical_spacing(np.array([self.positions[lbl] for lbl in optodes]))

        if self.head is not None:
            self._draw_head(ax, fg)
        self._set_view_limits(ax, xy_all, spacing)
        scale = self._points_per_unit(ax)
        marker_pt = float(np.clip(0.38 * spacing * scale, 2.5, 13))   # optode marker diameter (points)
        font_pt = float(np.clip(0.55 * spacing * scale, 5, 9))         # label font size

        self._draw_empty_slots(ax, fg, font_pt, marker_pt)
        self._draw_tiles(ax, fg, font_pt + 1, spacing)

        max_len = self.spin_max_length.value()
        pos = self.positions
        pairs = {}
        for ch in self.montage.channels:
            key = (ch['source'], ch['detector'])
            if ch['source'] not in pos or ch['detector'] not in pos:
                continue
            if ch['length'] is not None and ch['length'] > max_len:
                continue
            pairs.setdefault(key, []).append(ch)
        if pairs:
            segments = [(pos[s], pos[d]) for s, d in pairs]
            ax.add_collection(LineCollection(segments, colors=fg, linewidths=0.6, alpha=0.3, zorder=1))
            for (s, d), chans in pairs.items():
                self._hover['segments'].append((pos[s], pos[d], self._pair_text(s, d, chans)))

        for function, color, marker in (('Source', SOURCE_COLOR, 'o'), ('Detector', DETECTOR_COLOR, 's')):
            labels = [lbl for lbl in optodes if self.montage.probes[lbl]['function'] == function]
            if not labels:
                continue
            xy = np.array([pos[lbl] for lbl in labels])
            ax.scatter(xy[:, 0], xy[:, 1], s=marker_pt ** 2, c=color, marker=marker, zorder=3,
                       label=f"{function}s ({len(labels)})", edgecolors=theme["bg"], linewidths=0.4)
            for lbl, (x, y) in zip(labels, xy):
                self._hover['points'].append(((x, y), self._optode_text(lbl)))
                if self.chk_optode_labels.isChecked():
                    ax.annotate(lbl.split('/')[-1], (x, y), xytext=(marker_pt * 0.6, marker_pt * 0.6),
                                textcoords='offset points', fontsize=font_pt - 1, color=color, zorder=4)

        self._draw_highlights(ax, marker_pt)

        legend = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=7, framealpha=0.3,
                           borderaxespad=0, markerscale=min(1.0, 8 / max(marker_pt, 1)))
        for text in legend.get_texts():
            text.set_color(fg)
        title = f"Channels up to {max_len:g} mm: {len(pairs)} source-detector pairs"
        if self.template is not None:
            title += f"  |  {len(self.tile_slots)}/{len(self.template['slots'])} helmet slots used"
        ax.set_title(title, color=fg, fontsize=9)
        if self.method in ('azimuthal', 'saved'):
            ax.annotate("▲ anterior (nose)", xy=(0.99, 0.99), xycoords='axes fraction', ha='right', va='top',
                        fontsize=7, color=fg, alpha=0.7)
        self.toolbar.update()  # the toolbar's Home button returns to this view
        self.canvas.draw_idle()

    @staticmethod
    def _typical_spacing(xy):
        """Median distance from each optode to its nearest neighbour (layout units), or 10 if unknown."""
        if len(xy) < 2:
            return 10.0
        d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(d, np.inf)
        nearest = d.min(axis=1)
        nearest = nearest[nearest > 0]
        return float(np.median(nearest)) if len(nearest) else 10.0

    def _set_view_limits(self, ax, xy, spacing):
        """
        Frames the tiles and helmet slots with a margin (keeping the head
        outline when it is only slightly larger), or the whole head when
        'Full head' is checked.
        """
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        margin = max(2 * spacing, 0.06 * float(np.max(hi - lo)))
        lo, hi = lo - margin, hi + margin
        if self.head is not None:
            (cx, cy), r, _ = self.head
            r *= 1.12  # room for the nose and ears
            head_lo, head_hi = np.array([cx - r, cy - r]), np.array([cx + r, cy + r])
            if self.chk_full_head.isChecked() or np.all(hi - lo >= 0.75 * (head_hi - head_lo)):
                lo, hi = np.minimum(lo, head_lo), np.maximum(hi, head_hi)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])

    def _points_per_unit(self, ax):
        """How many typographic points one layout unit (mm) spans on screen with equal aspect."""
        bbox = ax.get_position()
        width_pt = bbox.width * self.figure.get_figwidth() * 72
        height_pt = bbox.height * self.figure.get_figheight() * 72
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        return min(width_pt / max(x1 - x0, 1e-9), height_pt / max(y1 - y0, 1e-9))

    def _draw_head(self, ax, fg):
        """
        Draws the head outline with nose and ears, and the fiducial marks.
        Without head fiducials in the stream the outline is estimated
        (dashed, labelled 'approximate').
        """
        (cx, cy), radius, approximate = self.head
        style = dict(fill=False, edgecolor=fg, linewidth=1.2, alpha=0.5, zorder=0,
                     linestyle='--' if approximate else '-')
        ax.add_patch(Circle((cx, cy), radius, **style))
        nose = [(cx - 0.08 * radius, cy + radius * 0.995), (cx, cy + radius * 1.1), (cx + 0.08 * radius, cy + radius * 0.995)]
        ax.add_patch(Polygon(nose, closed=False, **style))
        for side in (-1, 1):
            ear = [(cx + side * radius * 0.995, cy + 0.12 * radius), (cx + side * radius * 1.05, cy + 0.08 * radius),
                   (cx + side * radius * 1.05, cy - 0.12 * radius), (cx + side * radius * 0.995, cy - 0.16 * radius)]
            ax.add_patch(Polygon(ear, closed=False, **style))
        if approximate:
            ax.text(cx, cy - radius * 1.02, "approximate head outline (no head landmarks in the stream)",
                    fontsize=7, color=fg, alpha=0.6, ha='center', va='top', clip_on=True)
        for name, (x, y) in ((n, self.positions[n]) for n in self.montage.fiducials if n in self.positions):
            ax.plot(x, y, marker='+', color=fg, alpha=0.6, zorder=2)
            ax.text(x, y, f" {name}", fontsize=7, color=fg, alpha=0.7, zorder=2, clip_on=True)

    def _draw_tiles(self, ax, fg, font_pt, spacing):
        """
        Shades each tile (convex hull of its optodes) in its own color and
        writes its name just above the tile centre, beneath the optode markers,
        so it never hides the centre optode.
        """
        tiles = {}
        for label, probe in self.montage.probes.items():
            if label in self.positions and probe['module']:
                tiles.setdefault(probe['module'], []).append(self.positions[label])
        cmap = plt.get_cmap('tab20')
        for k, (tile, points) in enumerate(sorted(tiles.items(), key=lambda t: _natural_key(t[0]))):
            xy = np.array(points)
            color = cmap(k % 20)
            hull = _hull(xy)
            if hull is not None:
                ax.add_patch(Polygon(hull, closed=True, facecolor=color, alpha=0.22, edgecolor=color,
                                     linewidth=1.0, zorder=0))
            if self.chk_tile_labels.isChecked():
                cx, cy = xy.mean(axis=0)
                ax.text(cx, cy + 0.55 * spacing, tile, fontsize=font_pt, color=fg, ha='center', va='center',
                        zorder=2, fontweight='bold', clip_on=True,
                        bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=0.45, edgecolor='none'))

    def _draw_empty_slots(self, ax, fg, font_pt, marker_pt):
        """Draws each empty helmet slot as a dashed outline with faint optode dots and its '#id' label."""
        slots = {}
        for label in self.empty_points:
            if label in self.positions:
                slots.setdefault(label.split('/')[0], []).append(self.positions[label])
        info = {f"slot{s['id']}": s for s in (self.template or {}).get('slots', [])}
        for slot_key, points in slots.items():
            xy = np.array(points)
            hull = _hull(xy)
            if hull is not None:
                ax.add_patch(Polygon(hull, closed=True, fill=False, edgecolor=fg, linestyle='--',
                                     linewidth=0.9, alpha=0.45, zorder=0))
            ax.scatter(xy[:, 0], xy[:, 1], s=(marker_pt * 0.5) ** 2, c=fg, alpha=0.25, zorder=0, edgecolors='none')
            slot = info.get(slot_key, {})
            if self.chk_tile_labels.isChecked():
                cx, cy = xy.mean(axis=0)
                ax.text(cx, cy, f"#{slot.get('id', '?')}", fontsize=font_pt - 1, color=fg, alpha=0.5,
                        ha='center', va='center', zorder=1, clip_on=True)
            seen = ", ".join(slot.get('tiles_seen', [])) or "-"
            serials = ", ".join(slot.get('serials_seen', [])) or "-"
            self._hover['slots'].append((xy, f"Empty helmet slot #{slot.get('id', '?')}\n"
                                             f"tiles seen here before: {seen}\nserials: {serials}"))

    def _draw_highlights(self, ax, marker_pt):
        """Draws the live-plot channels' source-detector lines (with ringed endpoints) in their plot colors."""
        for p_idx, ch_idx in enumerate(self.highlight):
            if ch_idx >= len(self.montage.channels):
                continue
            ch = self.montage.channels[ch_idx]
            if ch['source'] in self.positions and ch['detector'] in self.positions:
                (x1, y1), (x2, y2) = self.positions[ch['source']], self.positions[ch['detector']]
                color = f"C{p_idx % 10}"
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=3, zorder=5, solid_capstyle='round',
                        label=ch['name'] if p_idx < 8 else None)
                ax.scatter([x1, x2], [y1, y2], s=(marker_pt * 1.8) ** 2, facecolors='none', edgecolors=color,
                           linewidths=1.5, zorder=5)

    # --- Hover tooltips ---

    def _optode_text(self, label):
        """Tooltip text for an optode: name, role, tile, serial and helmet slot."""
        probe = self.montage.probes[label]
        tile = probe['module']
        lines = [f"{label} ({probe['function'].lower() or 'optode'})", f"tile {tile}"
                 + (f" (serial {self.montage.modules[tile]})" if self.montage.modules.get(tile) else "")]
        if tile in self.tile_slots:
            lines.append(f"helmet slot #{self.tile_slots[tile]}")
        return "\n".join(lines)

    def _pair_text(self, source, detector, channels):
        """Tooltip text for a source-detector pair: distance, tiles and its channels (one per wavelength)."""
        tiles = [self.montage.probes.get(o, {}).get('module', '?') for o in (source, detector)]
        length = next((ch['length'] for ch in channels if ch['length'] is not None), None)
        head = f"{source} → {detector}" + (f"  ·  {length:.1f} mm" if length is not None else "")
        names = "\n".join(f"  {ch['name']}" + (f"  ({ch['wavelength']:g} nm)" if ch['wavelength'] else "")
                          for ch in channels)
        return f"{head}\ntiles {tiles[0]} → {tiles[1]}\nchannels:\n{names}"

    def _on_hover(self, event):
        """Shows a tooltip for the optode, channel line or empty slot under the cursor."""
        hover = self._hover
        if not hover or event.inaxes is not hover['ax'] or event.x is None:
            QToolTip.hideText()
            return
        to_px = hover['ax'].transData.transform
        cursor = np.array([event.x, event.y])
        text = None
        if hover['points']:
            px = to_px(np.array([p for p, _ in hover['points']]))
            d = np.hypot(*(px - cursor).T)
            i = int(np.argmin(d))
            if d[i] <= 8:
                text = hover['points'][i][1]
        if text is None and hover['segments']:
            a = to_px(np.array([s[0] for s in hover['segments']]))
            b = to_px(np.array([s[1] for s in hover['segments']]))
            ab = b - a
            t = np.clip(((cursor - a) * ab).sum(axis=1) / np.maximum((ab ** 2).sum(axis=1), 1e-9), 0, 1)
            d = np.hypot(*(a + ab * t[:, None] - cursor).T)
            i = int(np.argmin(d))
            if d[i] <= 4:
                text = hover['segments'][i][2]
        if text is None:
            for xy, slot_text in hover['slots']:
                px = to_px(xy)
                if np.hypot(*(px.mean(axis=0) - cursor)) <= 0.6 * np.ptp(px, axis=0).max() + 4:
                    text = slot_text
                    break
        if text:
            QToolTip.showText(QCursor.pos(), text, self.canvas)
        else:
            QToolTip.hideText()

    # --- Export ---

    def save_table(self):
        """Asks for a path and writes the per-channel layout table (core.montage.montage_table) as CSV."""
        if self.montage is None:
            return
        suggested = f"channel_layout_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save Layout Table", suggested, "CSV Files (*.csv)")
        if not path:
            return
        try:
            table = montage_table(self.montage, self.positions,
                                  self.tile_slots if self.template is not None else None)
            write_csv(table, path)
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Could not write the layout table to:\n{path}\n\n{e}")
            return
        QMessageBox.information(self, "Layout Saved",
                                f"{len(table)} channel rows written to:\n{os.path.basename(path)}")


def _hull(xy):
    """Convex hull vertices of 2D points, or None if there are under 3 or they are collinear."""
    if len(xy) < 3:
        return None
    try:
        return xy[ConvexHull(xy).vertices]
    except Exception:
        return None


def _natural_key(name):
    """Sort key putting 'N2' before 'N10'."""
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', str(name))]
