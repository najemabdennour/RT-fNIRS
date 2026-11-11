# ui/montage_view_3d.py
"""3D montage layout view: tiles, empty helmet slots, optodes and channels at their real 3D positions."""
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
                             QDoubleSpinBox, QCheckBox, QSizePolicy, QToolTip)
from scipy.spatial import ConvexHull

from core.montage import _fit_sphere

SOURCE_COLOR = "#e74c3c"
DETECTOR_COLOR = "#3498db"

# Camera presets (elevation, azimuth) for LUMO-style coordinates: +x right ear, +y nose, +z up.
VIEW_PRESETS = {"Top": (90, -90), "Front": (5, 90), "Back": (5, -90), "Left": (5, 180), "Right": (5, 0)}


class MontageView3D(QWidget):
    """
    Draws the montage of a MontageView (core.montage.Montage, helmet template,
    empty slots, live-plot highlights) at its true 3D positions: tiles as
    shaded patches, empty helmet slots as dashed outlines, sources and
    detectors, a line per source-detector pair up to a chosen length, the
    head fiducials, a translucent head surface and a face (nose, eyes,
    mouth, ears). Drag to rotate, use the view buttons for standard
    orientations, hover for details.

    Matplotlib's 3D has no hidden-surface removal, so after every rotation
    the elements on the far side of the head are faded (_apply_depth_fade);
    only the side facing the viewer is drawn solid.

    It follows the 2D view through MontageView.add_listener, so both always
    show the same montage. Drawing happens only on changes and only while
    the tab is visible, never per sample.
    """

    def __init__(self, theme_getter, source_view, parent=None):
        """
        Args:
            theme_getter: callable returning the active theme dict (ui.themes.THEMES entry).
            source_view: the ui.montage_view.MontageView whose montage and template are shown.
        """
        super().__init__(parent)
        self.theme_getter = theme_getter
        self.source = source_view
        self._dirty = True
        self._hover = None
        self._view = VIEW_PRESETS["Top"]
        self._depth = []            # registered artists faded by _apply_depth_fade
        self._depth_center = None   # head centre used for outward directions
        self.init_ui()
        source_view.add_listener(self._on_source_changed)

    def init_ui(self):
        """Builds the control ribbon (max length, labels, head surface, view presets) above the canvas."""
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

        self.chk_head = QCheckBox("Head surface")
        self.chk_head.setChecked(True)
        self.chk_head.setToolTip("Translucent head sphere (from the landmarks, else fitted to the optodes).")
        self.chk_head.toggled.connect(self.redraw)
        ribbon.addWidget(self.chk_head)

        self.chk_face = QCheckBox("Face")
        self.chk_face.setChecked(True)
        self.chk_face.setToolTip("Draws nose, eyes, mouth and ears, so the head's orientation is obvious.")
        self.chk_face.toggled.connect(self.redraw)
        ribbon.addWidget(self.chk_face)

        ribbon.addSpacing(12)
        ribbon.addWidget(QLabel("<b>View:</b>"))
        for name in VIEW_PRESETS:
            btn = QPushButton(name)
            btn.setToolTip(f"Look at the head from the {name.lower()}.")
            btn.clicked.connect(lambda _c, n=name: self.set_view(n))
            ribbon.addWidget(btn)
        ribbon.addStretch()
        layout.addLayout(ribbon)

        # Layout files and helmet templates: same actions and shared state as the
        # 2D view (these controls drive it, so both tabs always agree).
        src = self.source
        file_ribbon = QHBoxLayout()
        file_ribbon.addWidget(QLabel("<b>Helmet template:</b>"))
        self.combo_template = QComboBox()
        self.combo_template.setMinimumWidth(180)
        self.combo_template.setToolTip(src.combo_template.toolTip())
        self.combo_template.activated.connect(self._on_template_chosen)
        file_ribbon.addWidget(self.combo_template)

        self.chk_auto_learn = QCheckBox("Learn slots from each stream")
        self.chk_auto_learn.setToolTip(src.chk_auto_learn.toolTip())
        self.chk_auto_learn.setChecked(src.chk_auto_learn.isChecked())
        self.chk_auto_learn.toggled.connect(src.chk_auto_learn.setChecked)
        src.chk_auto_learn.toggled.connect(self._sync_auto_learn)
        file_ribbon.addWidget(self.chk_auto_learn)

        self.btn_save_template = QPushButton("Save as Helmet Template...")
        self.btn_save_template.setToolTip(src.btn_save_template.toolTip())
        self.btn_save_template.clicked.connect(src.save_as_template)
        file_ribbon.addWidget(self.btn_save_template)
        file_ribbon.addStretch()

        self.btn_load_layout = QPushButton("Load Layout File...")
        self.btn_load_layout.setToolTip(src.btn_load_layout.toolTip())
        self.btn_load_layout.clicked.connect(src.load_layout_file)
        file_ribbon.addWidget(self.btn_load_layout)
        self.btn_save_table = QPushButton("Save Layout Table (CSV)...")
        self.btn_save_table.setToolTip("Writes one row per channel (tiles, distances, 3D and 2D positions, "
                                       "helmet slots, system info) - the same table as the 2D Layout tab.")
        self.btn_save_table.clicked.connect(src.save_table)
        file_ribbon.addWidget(self.btn_save_table)
        layout.addLayout(file_ribbon)
        self._sync_controls()

        # System and helmet/montage lines, matching the 2D view (updated on every change).
        self.lbl_system = QLabel("")
        self.lbl_system.setStyleSheet("font-weight: bold; font-size: 11px;")
        self.lbl_system.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.lbl_system)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet("color: #b0b0b0; font-size: 10px;")
        self.lbl_summary.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.lbl_summary)

        self.lbl_hint = QLabel("Drag to rotate, scroll to zoom; hover an optode or slot for details.")
        self.lbl_hint.setStyleSheet("color: #b0b0b0; font-size: 10px;")
        self.lbl_hint.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.lbl_hint)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)
        self.canvas.mpl_connect('motion_notify_event', self._on_hover)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_release_event', lambda event: self._refresh_depth_fade())

    def _sync_controls(self):
        """Mirrors the 2D view's template list, selection and button states into this tab's controls."""
        src = self.source
        self.combo_template.blockSignals(True)
        self.combo_template.clear()
        self.combo_template.addItems([src.combo_template.itemText(i) for i in range(src.combo_template.count())])
        self.combo_template.setCurrentIndex(src.combo_template.currentIndex())
        self.combo_template.blockSignals(False)
        self.btn_save_template.setEnabled(src.btn_save_template.isEnabled())
        self.btn_save_table.setEnabled(src.btn_save_table.isEnabled())

    def _on_template_chosen(self, index):
        """Applies a template chosen here through the 2D view (which then notifies both tabs)."""
        self.source.combo_template.setCurrentIndex(index)
        self.source._apply_template()

    def _sync_auto_learn(self, checked):
        """Follows the 2D view's 'Learn slots' checkbox without echoing the change back."""
        self.chk_auto_learn.blockSignals(True)
        self.chk_auto_learn.setChecked(checked)
        self.chk_auto_learn.blockSignals(False)

    def _on_source_changed(self):
        """
        Shows the 2D view's system line and the montage counts, layout file
        and helmet template status (without the 2D projection note), even
        while hidden, then redraws.
        """
        src = self.source
        self._sync_controls()
        self.lbl_system.setText(src.lbl_system.text())
        if src.montage is None:
            self.lbl_summary.setText("")
        else:
            positions = ("3D optode positions from the stream" if src.montage.has_positions
                         else "no 3D optode positions in the stream")
            self.lbl_summary.setText(f"{src.montage.summary()} - {positions}"
                                     f"{getattr(src, '_layout_source', '')}{src._template_note}.")
        self.redraw()

    def set_view(self, name):
        """Turns the camera to a preset orientation (see VIEW_PRESETS)."""
        self._view = VIEW_PRESETS[name]
        if self.figure.axes and hasattr(self.figure.axes[0], 'view_init'):
            self.figure.axes[0].view_init(*self._view)
            self._refresh_depth_fade()

    def showEvent(self, event):
        """Draws any change made while the view was hidden."""
        super().showEvent(event)
        if self._dirty:
            self.redraw()

    def _show_message(self, text):
        """Replaces the drawing with a centred message."""
        theme = self.theme_getter()
        self._hover = None
        self.figure.clear()
        self.figure.patch.set_facecolor(theme["bg"])
        self.figure.text(0.5, 0.5, text, ha='center', va='center', color=theme["fg"], fontsize=10, wrap=True)
        self.canvas.draw_idle()

    def redraw(self):
        """Draws the current montage of the source view in 3D (deferred until shown if hidden)."""
        if not self.isVisible():
            self._dirty = True
            return
        self._dirty = False
        src = self.source
        montage = src.montage
        if montage is None:
            self._show_message(src._message)
            return
        probes = {lbl: p for lbl, p in montage.probes.items() if p['position'] is not None}
        if not probes:
            self._show_message("This stream's metadata has no 3D optode positions, so no 3D layout can be drawn.")
            return

        # Keep the user's rotation across redraws.
        if self.figure.axes and hasattr(self.figure.axes[0], 'elev'):
            self._view = (self.figure.axes[0].elev, self.figure.axes[0].azim)

        theme = self.theme_getter()
        fg, bg = theme["fg"], theme["bg"]
        self.figure.clear()
        self.figure.patch.set_facecolor(bg)
        ax = self.figure.add_axes([0.0, 0.0, 0.82, 0.95], projection='3d')
        ax.set_facecolor(bg)
        ax.set_axis_off()
        self._hover = {'ax': ax, 'points': [], 'slots': []}
        self._depth = []

        all_xyz = [p['position'] for p in probes.values()] + list(src.empty_points.values())
        all_xyz += list(montage.fiducials.values())
        all_xyz = np.array(all_xyz)

        frame = self._head_frame(montage, all_xyz)
        self._depth_center = frame['center'] if frame is not None else all_xyz.mean(axis=0)
        if frame is not None and self.chk_face.isChecked():
            all_xyz = np.vstack([all_xyz, self._draw_face(ax, fg, montage, frame)])
        if frame is not None and self.chk_head.isChecked():
            self._draw_head(ax, fg, frame, all_xyz)
        self._draw_empty_slots(ax, fg)
        self._draw_tiles(ax, fg, montage, probes)

        max_len = self.spin_max_length.value()
        segments, seen = [], set()
        for ch in montage.channels:
            key = (ch['source'], ch['detector'])
            if key in seen or ch['source'] not in probes or ch['detector'] not in probes:
                continue
            if ch['length'] is not None and ch['length'] > max_len:
                continue
            seen.add(key)
            segments.append((probes[ch['source']]['position'], probes[ch['detector']]['position']))
        if segments:
            lines = Line3DCollection(segments, colors=fg, linewidths=0.5)
            ax.add_collection3d(lines)
            mids = np.array([(np.asarray(a) + np.asarray(b)) / 2 for a, b in segments])
            self._depth.append(('array', lines, mids, np.tile(to_rgba(fg, 0.3), (len(mids), 1)), 'color'))

        for function, color, marker in (('Source', SOURCE_COLOR, 'o'), ('Detector', DETECTOR_COLOR, 's')):
            labels = [lbl for lbl, p in probes.items() if p['function'] == function]
            if labels:
                xyz = np.array([probes[lbl]['position'] for lbl in labels])
                dots = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=14, c=color, marker=marker, depthshade=False,
                                  label=f"{function}s ({len(labels)})", edgecolors=bg, linewidths=0.3)
                self._depth.append(('array', dots, xyz, np.tile(to_rgba(color), (len(xyz), 1)), 'face'))
                for lbl, p in zip(labels, xyz):
                    self._hover['points'].append((p, src._optode_text(lbl)))

        for name, pos in montage.fiducials.items():
            self._fade(ax.scatter(*pos, marker='+', s=60, c=fg, depthshade=False), pos, 0.8)
            self._fade(ax.text(*pos, f" {name}", fontsize=7, color=fg), pos, 0.8)

        self._draw_highlights(ax, montage, probes)
        self._set_equal_limits(ax, all_xyz)
        ax.view_init(*self._view)
        self._apply_depth_fade()

        legend = ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=7, framealpha=0.3)
        for text in legend.get_texts():
            text.set_color(fg)
        title = f"Channels up to {max_len:g} mm: {len(segments)} source-detector pairs"
        if src.template is not None:
            title += f"  |  {len(src.tile_slots)}/{len(src.template['slots'])} helmet slots used"
        ax.set_title(title, color=fg, fontsize=9)
        self.canvas.draw_idle()

    @staticmethod
    def _set_equal_limits(ax, xyz):
        """Gives the three axes the same range around the points, so the head isn't stretched."""
        center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
        half = float(np.max(xyz.max(axis=0) - xyz.min(axis=0))) / 2 * 1.05
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1, 1, 1), zoom=1.45)  # matplotlib leaves wide 3D margins by default

    @staticmethod
    def _outline(xyz):
        """Ordered outline of a tile's optodes: convex hull in their best-fit plane, or None."""
        if len(xyz) < 3:
            return None
        centered = xyz - xyz.mean(axis=0)
        basis = np.linalg.svd(centered, full_matrices=False)[2][:2]
        try:
            return xyz[ConvexHull(centered @ basis.T).vertices]
        except Exception:
            return None

    @staticmethod
    def _head_frame(montage, xyz):
        """
        Head centre, radius and orientation for drawing the head and face.

        With Nasion, Inion, Al and Ar: centre = their mean, radius = their mean
        distance from it, 'right' towards Ar, 'front' towards the Nasion and
        'up' towards Cz (or perpendicular to both). Otherwise a sphere is
        fitted to the points and LUMO's axes are assumed (+x right ear,
        +y nose, +z up).

        Returns:
            dict with 'center', 'radius', 'right', 'front', 'up' (orthonormal),
            'exact' (True when built from landmarks) and the landmarks used, or
            None with fewer than 4 points.
        """
        fid = montage.fiducials
        if all(name in fid for name in ('Nasion', 'Inion', 'Al', 'Ar')):
            center = np.mean([fid[n] for n in ('Nasion', 'Inion', 'Al', 'Ar')], axis=0)
            radius = float(np.mean([np.linalg.norm(fid[n] - center) for n in ('Nasion', 'Inion', 'Al', 'Ar')]))
            right, front = fid['Ar'] - fid['Al'], fid['Nasion'] - fid['Inion']
            up = fid['Cz'] - center if 'Cz' in fid else np.cross(right, front)
            exact = True
        elif len(xyz) >= 4:
            center, radius = _fit_sphere(xyz)
            right, front, up = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
            exact = False
        else:
            return None
        up = up / np.linalg.norm(up)
        right = right - right.dot(up) * up
        right /= np.linalg.norm(right)
        front = np.cross(up, right)
        return {'center': np.asarray(center, dtype=float), 'radius': radius, 'right': right,
                'front': front, 'up': up, 'exact': exact, 'fiducials': fid if exact else {}}

    @staticmethod
    def _surface(frame, lat_deg, az_deg, scale=1.0):
        """
        Point on the head sphere at a latitude (0 = ear/nasion level, +90 = top)
        and azimuth (0 = nose, +90 = right ear), optionally pushed out by `scale`.
        """
        lat, az = np.radians(lat_deg), np.radians(az_deg)
        direction = (np.cos(lat) * (np.sin(az) * frame['right'] + np.cos(az) * frame['front'])
                     + np.sin(lat) * frame['up'])
        return frame['center'] + frame['radius'] * scale * direction

    def _draw_face(self, ax, fg, montage, frame):
        """
        Draws nose, eyes, mouth and ears on the head sphere, and 'Front' /
        'Left' / 'Right' labels. Ears sit at the Al/Ar landmarks and the nose
        bridge at the Nasion when the stream has them. Returns the drawn
        points (for the view limits).
        """
        R, surf = frame['radius'], lambda lat, az, k=1.0: self._surface(frame, lat, az, k)
        fid = frame['fiducials']
        style = dict(color=fg, linewidth=1.3, alpha=0.75)
        drawn = []

        # Nose: three faces from the bridge and the two nostril corners to the tip.
        bridge = fid['Nasion'] if 'Nasion' in fid else surf(-2, 0)
        left_base, right_base, tip = surf(-30, -9), surf(-30, 9), surf(-26, 0, 1.22)
        nose = [[bridge, left_base, tip], [bridge, right_base, tip], [left_base, right_base, tip]]
        self._fade(ax.add_collection3d(Poly3DCollection(nose, facecolors=fg, edgecolors=fg, linewidths=1.0)), tip, 0.28)
        drawn += [bridge, left_base, right_base, tip]

        # Eyes: outlined almond shapes with a pupil.
        t = np.linspace(0, 2 * np.pi, 30)
        for side in (-1, 1):
            eye = np.array([surf(-12 + 4.5 * np.sin(a), side * 27 + 9 * np.cos(a), 1.005) for a in t])
            pupil = surf(-12, side * 27, 1.01)
            self._fade(ax.plot(eye[:, 0], eye[:, 1], eye[:, 2], **style)[0], pupil, style['alpha'])
            self._fade(ax.scatter(*pupil, s=16, c=fg, depthshade=False), pupil, 0.9)
            drawn += list(eye)

        # Mouth: a slight smile.
        mouth = np.array([surf(-46 - 4 * (1 - (az / 20) ** 2), az, 1.005) for az in np.linspace(-20, 20, 25)])
        self._fade(ax.plot(mouth[:, 0], mouth[:, 1], mouth[:, 2], **style)[0], mouth[12], style['alpha'])
        drawn += list(mouth)

        # Ears: flat ovals standing just outside the head at the ear landmarks.
        for side, name in ((1, 'Ar'), (-1, 'Al')):
            base = fid[name] if name in fid else surf(-10, side * 90)
            center = base + side * frame['right'] * 0.06 * R - frame['front'] * 0.04 * R
            ear = np.array([center + 0.30 * R * np.sin(a) * frame['up'] + 0.15 * R * np.cos(a) * frame['front']
                            for a in t])
            self._fade(ax.add_collection3d(Poly3DCollection([ear], facecolors=fg, edgecolors=fg, linewidths=1.2)),
                       center + side * frame['right'] * R, 0.2)
            label = surf(-10, side * 90, 1.35)
            ax.text(*label, "Right" if side > 0 else "Left", fontsize=8, color=fg, alpha=0.85,
                    ha='center', fontweight='bold')
            drawn += list(ear) + [label]

        front_label = surf(-26, 0, 1.45)
        ax.text(*front_label, "Front", fontsize=8, color=fg, alpha=0.85, ha='center', fontweight='bold')
        drawn.append(front_label)
        if not frame['exact']:
            ax.text2D(0.01, 0.01, "Orientation assumed (+x right, +y nose, +z up): no head landmarks in the stream",
                      transform=ax.transAxes, fontsize=7, color=fg, alpha=0.6)
        return np.array(drawn)

    def _draw_head(self, ax, fg, frame, xyz):
        """
        Draws a translucent wireframe of the head sphere, from the top down
        to just below the lowest drawn point (the chin when the face is on).
        """
        center, radius, up = frame['center'], frame['radius'], frame['up']
        lowest = float(np.min((xyz - center) @ up)) / radius - 0.1
        polar = np.linspace(0, np.arccos(np.clip(lowest, -1, 1)), 16)
        azim = np.linspace(0, 2 * np.pi, 40)
        P, A = np.meshgrid(polar, azim)
        e1, e2 = frame['right'], frame['front']
        pts = (center[:, None, None]
               + radius * (np.cos(P) * up[:, None, None]
                           + np.sin(P) * (np.cos(A) * e1[:, None, None] + np.sin(A) * e2[:, None, None])))
        ax.plot_wireframe(pts[0], pts[1], pts[2], color=fg, linewidth=0.3, alpha=0.12)

    def _draw_tiles(self, ax, fg, montage, probes):
        """Shades each used tile in its own color and writes its name just above it."""
        tiles = {}
        for label, probe in probes.items():
            if probe['module']:
                tiles.setdefault(probe['module'], []).append(probe['position'])
        cmap = plt.get_cmap('tab20')
        for k, (tile, points) in enumerate(sorted(tiles.items(), key=lambda t: _natural_key(t[0]))):
            xyz = np.array(points)
            color = cmap(k % 20)
            outline = self._outline(xyz)
            centroid = xyz.mean(axis=0)
            if outline is not None:
                self._fade(ax.add_collection3d(Poly3DCollection([outline], facecolors=color, edgecolors=color,
                                                                linewidths=0.8)), centroid, 0.35)
            if self.chk_tile_labels.isChecked():
                self._fade(ax.text(*centroid, tile, fontsize=7, color=fg, fontweight='bold', ha='center',
                                   bbox=dict(boxstyle='round,pad=0.12', facecolor=color, alpha=0.5, edgecolor='none')),
                           centroid, 1.0)

    def _draw_empty_slots(self, ax, fg):
        """Draws each empty helmet slot as a dashed outline with faint optode dots and its '#id' label."""
        src = self.source
        slots = {}
        for label, pos in src.empty_points.items():
            slots.setdefault(label.split('/')[0], []).append(pos)
        info = {f"slot{s['id']}": s for s in (src.template or {}).get('slots', [])}
        for slot_key, points in slots.items():
            xyz = np.array(points)
            centroid = xyz.mean(axis=0)
            outline = self._outline(xyz)
            if outline is not None:
                closed = np.vstack([outline, outline[:1]])
                self._fade(ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=fg, linestyle='--',
                                   linewidth=0.8)[0], centroid, 0.45)
            self._fade(ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=4, c=fg, depthshade=False), centroid, 0.25)
            slot = info.get(slot_key, {})
            if self.chk_tile_labels.isChecked():
                self._fade(ax.text(*centroid, f"#{slot.get('id', '?')}", fontsize=6, color=fg, ha='center'),
                           centroid, 0.5)
            seen = ", ".join(slot.get('tiles_seen', [])) or "-"
            self._hover['slots'].append((xyz.mean(axis=0), f"Empty helmet slot #{slot.get('id', '?')}\n"
                                                           f"tiles seen here before: {seen}"))

    def _draw_highlights(self, ax, montage, probes):
        """Draws the live-plot channels' source-detector lines in their plot colors."""
        for p_idx, ch_idx in enumerate(self.source.highlight):
            if ch_idx >= len(montage.channels):
                continue
            ch = montage.channels[ch_idx]
            if ch['source'] in probes and ch['detector'] in probes:
                a, b = probes[ch['source']]['position'], probes[ch['detector']]['position']
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=f"C{p_idx % 10}", linewidth=3,
                        label=ch['name'] if p_idx < 8 else None)

    # --- Depth fading (no hidden-surface removal in matplotlib 3D) ---

    def _fade(self, artist, xyz, alpha):
        """
        Registers an artist placed at xyz with its full alpha for depth fading,
        and returns it. 3D polygons keep their face/edge colors (their edges
        ignore set_alpha), and a text's background box is faded with it.
        """
        xyz = np.asarray(xyz, dtype=float)
        if isinstance(artist, Poly3DCollection):
            face = to_rgba(artist.get_facecolor()[0])
            edge = to_rgba(artist.get_edgecolor()[0])
            self._depth.append(('poly', artist, xyz, alpha, (face, edge)))
        else:
            box = artist.get_bbox_patch() if hasattr(artist, 'get_bbox_patch') else None
            self._depth.append(('single', artist, xyz, alpha, box.get_alpha() if box is not None else None))
        return artist

    def _facing_weights(self, xyz):
        """
        How much each point faces the camera, from 0 (far side of the head)
        to 1 (facing the viewer), using its direction from the head centre.
        """
        ax = self.figure.axes[0]
        elev, azim = np.radians(ax.elev), np.radians(ax.azim)
        camera = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
        out = np.atleast_2d(xyz) - self._depth_center
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return np.clip((out @ camera + 0.15) / 0.35, 0.0, 1.0)

    def _apply_depth_fade(self):
        """Sets every registered artist's alpha: full when facing the viewer, faint (12%) on the far side."""
        if not self._depth or self._depth_center is None or not self.figure.axes:
            return
        for kind, artist, xyz, base, prop in self._depth:
            weight = 0.12 + 0.88 * self._facing_weights(xyz)
            if kind == 'single':
                artist.set_alpha(float(base * weight[0]))
                if prop is not None:  # text background box
                    artist.get_bbox_patch().set_alpha(float(prop * weight[0]))
            elif kind == 'poly':
                face, edge = prop
                artist.set_facecolor((*face[:3], float(base * weight[0])))
                artist.set_edgecolor((*edge[:3], float(min(1.0, base * 2.5) * weight[0])))
            else:
                rgba = base.copy()
                rgba[:, 3] *= weight
                (artist.set_color if prop == 'color' else artist.set_facecolor)(rgba)

    def _refresh_depth_fade(self):
        """Re-fades after the camera moved (rotation released or preset view) and redraws."""
        if self._depth and self.figure.axes:
            self._apply_depth_fade()
            self.canvas.draw_idle()

    def _on_scroll(self, event):
        """Zooms in or out around the centre of the view."""
        if not self.figure.axes or event.inaxes is None:
            return
        ax = self.figure.axes[0]
        factor = 0.85 if event.button == 'up' else 1 / 0.85
        for get, set_ in ((ax.get_xlim, ax.set_xlim), (ax.get_ylim, ax.set_ylim), (ax.get_zlim, ax.set_zlim)):
            lo, hi = get()
            mid, half = (lo + hi) / 2, (hi - lo) / 2 * factor
            set_(mid - half, mid + half)
        self.canvas.draw_idle()

    def _on_hover(self, event):
        """Shows a tooltip for the optode or empty slot under the cursor (not while rotating)."""
        hover = self._hover
        if not hover or event.inaxes is not hover['ax'] or event.button is not None or event.x is None:
            QToolTip.hideText()
            return
        ax = hover['ax']
        proj = ax.get_proj()
        candidates = [(p, t, 8) for p, t in hover['points']] + [(p, t, 14) for p, t in hover['slots']]
        if not candidates:
            return
        xyz = np.array([c[0] for c in candidates])
        x2, y2, _ = proj3d.proj_transform(xyz[:, 0], xyz[:, 1], xyz[:, 2], proj)
        px = ax.transData.transform(np.column_stack([x2, y2]))
        d = np.hypot(px[:, 0] - event.x, px[:, 1] - event.y)
        i = int(np.argmin(d))
        if d[i] <= candidates[i][2]:
            QToolTip.showText(QCursor.pos(), candidates[i][1], self.canvas)
        else:
            QToolTip.hideText()


def _natural_key(name):
    """Sort key putting 'N2' before 'N10'."""
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', str(name))]
