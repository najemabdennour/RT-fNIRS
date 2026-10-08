# ui/experiment_tab.py
"""
Experiment Design tab: build a visual trial sequence that runs in a separate
Pygame process, or launch a custom Python paradigm script and show its output.
Also hosts a live LSL stream catcher for watching feedback probabilities.
"""
import os
import sys
import json
import subprocess
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QCheckBox, QGroupBox, QFileDialog, QMessageBox, QSplitter,
    QTextEdit, QSpinBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QComboBox, QTabWidget, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from pylsl import StreamInlet, resolve_byprop

from ui.components import CollapsibleSection

# --- Step table layout ---
STEP_TYPES = ["Fixation Cross", "Text Prompt", "Shape", "Image", "Feedback Window"]
SHAPE_TYPES = ["Circle", "Square", "Triangle", "Rectangle", "Diamond"]
SHAPE_COLORS = ["Green", "Red", "Blue", "Yellow", "White", "Orange", "Purple"]

# The Content column's widget depends on Step Type (text field, shape picker,
# image browser or feedback options), so those are not separate columns.
(COL_TYPE, COL_CONTENT, COL_COLOR, COL_SIZE,
 COL_DURATION, COL_MARKER) = range(6)

COLUMN_LABELS = [
    "Step Type", "Content / Stimulus", "Color", "Size (px)",
    "Duration (s)", "LSL Marker"
]

# Default names for the lifecycle markers the runner sends on its own
# (baseline, run, trial index, target class). Editable in the "System LSL
# Markers" panel; per-step markers come from the table's LSL Marker column.
DEFAULT_SYSTEM_MARKERS = {
    "baseline_start": "baseline_start",
    "baseline_end": "baseline_end",
    "run_onset": "run_onset",
    "run_offset": "run_offset",
    "trial_index": "trial_index",
    "target_class": "target_class",
}

# Source of the Pygame runner. Written to temp_runner.py and executed in a
# separate process with the JSON config path as its only argument.
RUNNER_CODE = """
import sys, time, json, os, pygame
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop

COLOR_MAP = {
    "Green": (0, 230, 120),
    "Red": (230, 60, 60),
    "Blue": (70, 130, 230),
    "Yellow": (230, 210, 60),
    "White": (240, 240, 240),
    "Orange": (240, 150, 40),
    "Purple": (170, 90, 220),
}

def run():
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    pygame.init()
    pygame.mixer.init()

    w, h = cfg.get("screen_width", 1024), cfg.get("screen_height", 768)
    flags = pygame.FULLSCREEN if cfg.get("fullscreen", False) else 0
    screen = pygame.display.set_mode((w, h), flags)
    pygame.display.set_caption("fNIRS Stimulus Renderer")
    font = pygame.font.SysFont("Arial", cfg.get("font_size", 36))
    clock = pygame.time.Clock()

    # Customizable system markers (session/trial lifecycle events). Per-step
    # markers - trial_onset, trial_offset, request_feedback, etc. - are
    # driven entirely by each step's own 'marker' field, set in the table.
    markers_cfg = cfg.get("markers", {})
    M_BASELINE_START = markers_cfg.get("baseline_start", "baseline_start")
    M_BASELINE_END = markers_cfg.get("baseline_end", "baseline_end")
    M_RUN_ONSET = markers_cfg.get("run_onset", "run_onset")
    M_RUN_OFFSET = markers_cfg.get("run_offset", "run_offset")
    M_TRIAL_INDEX = markers_cfg.get("trial_index", "trial_index")
    M_TARGET_CLASS = markers_cfg.get("target_class", "target_class")
    trial_target_sequence = cfg.get("trial_target_sequence", [])

    # Feedback Window display constants. Style and label visibility are set
    # per-step (see each step's "feedback_style"/"show_feedback_label"
    # fields, chosen in the step table's Content column) - only the chance
    # level and the two Disk-style reference-circle radii are fixed here.
    CHANCE_LEVEL = 0.5
    CHANCE_LEVEL_RADIUS = 50
    MAX_RADIUS = 100
    HBAR_W, HBAR_H = 400, 40   # Progress Bar (Horizontal) dimensions
    VBAR_W, VBAR_H = 60, 300   # Progress Bar (Vertical) dimensions

    def circle_radius(decoding_prob, chance_level, chance_level_radius):
        # Transform decoding probability to circle size - matches chance_level_radius
        # exactly when decoding_prob == chance_level, and grows/shrinks from there.
        if chance_level <= 0:
            return 0
        return (decoding_prob * chance_level_radius) / chance_level

    marker_outlet = StreamOutlet(StreamInfo('OpenSesame_Markers', 'Markers', 2, 0, 'string', 'py_runner_01'))
    feedback_inlet = None

    font_cache = {}
    def get_font(size):
        if size not in font_cache:
            font_cache[size] = pygame.font.SysFont("Arial", max(int(size), 6))
        return font_cache[size]

    def check_exit():
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit(); sys.exit()

    def draw_shape(shape, color, size):
        cx, cy = w // 2, h // 2
        if shape == "Circle":
            pygame.draw.circle(screen, color, (cx, cy), size)
        elif shape == "Square":
            pygame.draw.rect(screen, color, pygame.Rect(cx - size, cy - size, size * 2, size * 2))
        elif shape == "Rectangle":
            pygame.draw.rect(screen, color, pygame.Rect(cx - size, cy - size // 2, size * 2, size))
        elif shape == "Triangle":
            pygame.draw.polygon(screen, color, [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)])
        elif shape == "Diamond":
            pygame.draw.polygon(screen, color, [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)])
        else:
            pygame.draw.circle(screen, color, (cx, cy), size)

    def render_screen(step, prob_value=None):
        stype = step.get("type", "")
        screen.fill((15, 15, 15))
        cx, cy = w // 2, h // 2
        color = COLOR_MAP.get(step.get("color", "White"), (240, 240, 240))

        if stype == "Fixation Cross":
            size = step.get("size", 30)
            pygame.draw.line(screen, color, (cx - size, cy), (cx + size, cy), 5)
            pygame.draw.line(screen, color, (cx, cy - size), (cx, cy + size), 5)
        elif stype == "Text Prompt":
            step_font = get_font(step.get("size", cfg.get("font_size", 36)))
            txt = step_font.render(step.get("content", ""), True, color)
            screen.blit(txt, txt.get_rect(center=(cx, cy)))
        elif stype == "Shape":
            shape_color = COLOR_MAP.get(step.get("color", "Green"), (0, 230, 120))
            draw_shape(step.get("shape", "Circle"), shape_color, step.get("size", 80))
        elif stype == "Image":
            img_path = step.get("content", "")
            if img_path and os.path.exists(img_path):
                img = pygame.image.load(img_path)
                img = pygame.transform.scale(img, (w // 2, h // 2))
                screen.blit(img, img.get_rect(center=(cx, cy)))
            else:
                txt = font.render("[Image not found]", True, (230, 60, 60))
                screen.blit(txt, txt.get_rect(center=(cx, cy)))
        elif stype == "Feedback Window":
            feedback_style = step.get("feedback_style", "Text")
            show_label = step.get("show_feedback_label", True)

            # Vertical half-extent of whatever this style actually draws, so
            # the "Feedback" label sits clear above it regardless of style -
            # previously fixed to MAX_RADIUS (the Disk style's own radius),
            # which the taller Vertical Progress Bar grew past and covered.
            if feedback_style == "Progress Bar (Horizontal)":
                content_half_height = HBAR_H // 2
            elif feedback_style == "Progress Bar (Vertical)":
                content_half_height = VBAR_H // 2
            elif feedback_style == "Disk":
                content_half_height = MAX_RADIUS
            else:  # "Text"
                content_half_height = 30

            if show_label:
                step_font = get_font(step.get("size", cfg.get("font_size", 36)))
                label = step_font.render("Feedback", True, color)
                screen.blit(label, label.get_rect(center=(cx, cy - content_half_height - 40)))

            if prob_value is None:
                status_font = get_font(28)
                status = status_font.render("Waiting for decoder...", True, (200, 200, 200))
                screen.blit(status, status.get_rect(center=(cx, cy)))
            elif feedback_style == "Progress Bar (Horizontal)":
                show_pct = step.get("show_percentage", True)
                bar_w, bar_h = HBAR_W, HBAR_H
                bar_x, bar_y = cx - bar_w // 2, cy - bar_h // 2
                pygame.draw.rect(screen, (60, 60, 60), pygame.Rect(bar_x, bar_y, bar_w, bar_h))
                fill_w = int(bar_w * max(0.0, min(1.0, prob_value)))
                pygame.draw.rect(screen, (0, 230, 120), pygame.Rect(bar_x, bar_y, fill_w, bar_h))
                pygame.draw.rect(screen, (240, 240, 240), pygame.Rect(bar_x, bar_y, bar_w, bar_h), 3)
                chance_x = bar_x + int(bar_w * CHANCE_LEVEL)
                pygame.draw.line(screen, (230, 210, 60), (chance_x, bar_y - 8), (chance_x, bar_y + bar_h + 8), 3)
                if show_pct:
                    pct_font = get_font(24)
                    pct_txt = pct_font.render(f"{prob_value * 100:.1f}%", True, (240, 240, 240))
                    screen.blit(pct_txt, pct_txt.get_rect(center=(cx, bar_y + bar_h + 30)))
            elif feedback_style == "Progress Bar (Vertical)":
                show_pct = step.get("show_percentage", True)
                bar_w, bar_h = VBAR_W, VBAR_H
                bar_x, bar_y = cx - bar_w // 2, cy - bar_h // 2
                pygame.draw.rect(screen, (60, 60, 60), pygame.Rect(bar_x, bar_y, bar_w, bar_h))
                fill_h = int(bar_h * max(0.0, min(1.0, prob_value)))
                fill_y = bar_y + bar_h - fill_h  # fill grows upward from the bottom
                pygame.draw.rect(screen, (0, 230, 120), pygame.Rect(bar_x, fill_y, bar_w, fill_h))
                pygame.draw.rect(screen, (240, 240, 240), pygame.Rect(bar_x, bar_y, bar_w, bar_h), 3)
                chance_y = bar_y + bar_h - int(bar_h * CHANCE_LEVEL)
                pygame.draw.line(screen, (230, 210, 60), (bar_x - 8, chance_y), (bar_x + bar_w + 8, chance_y), 3)
                if show_pct:
                    pct_font = get_font(24)
                    pct_txt = pct_font.render(f"{prob_value * 100:.1f}%", True, (240, 240, 240))
                    screen.blit(pct_txt, pct_txt.get_rect(center=(cx, bar_y + bar_h + 30)))
            elif feedback_style == "Disk":
                show_pct = step.get("show_percentage", True)
                # Clamped so a very high-confidence trial can't draw a circle far off-screen.
                perf_radius = max(0, min(circle_radius(prob_value, CHANCE_LEVEL, CHANCE_LEVEL_RADIUS), MAX_RADIUS * 2))
                pygame.draw.circle(screen, (230, 60, 60), (cx, cy), int(perf_radius))            # Participant's performance
                pygame.draw.circle(screen, (230, 210, 60), (cx, cy), int(CHANCE_LEVEL_RADIUS), 2)  # Chance-level reference
                pygame.draw.circle(screen, (240, 240, 240), (cx, cy), int(MAX_RADIUS), 4)          # Maximum performance reference
                if show_pct:
                    pct_font = get_font(24)
                    pct_txt = pct_font.render(f"{prob_value * 100:.1f}%", True, (240, 240, 240))
                    screen.blit(pct_txt, pct_txt.get_rect(center=(cx, cy + MAX_RADIUS + 30)))
            else:  # "Text" - always shows the percentage; that's its whole purpose
                pct_font = get_font(48)
                pct_txt = pct_font.render(f"{prob_value * 100:.1f}%", True, (255, 215, 0))
                screen.blit(pct_txt, pct_txt.get_rect(center=(cx, cy)))

        pygame.display.flip()

    # --- Pre-Baseline Grace Countdown ---
    # Gives downstream LSL consumers (like the Neurofeedback Engine) time to
    # finish resolving and connecting to these streams before 'baseline_start'
    # fires. Without this, the very first marker sample can be pushed and
    # gone before a late-connecting inlet is listening, which silently
    # produces a 0-frame baseline.
    pre_grace = cfg.get("pre_baseline_delay", 10)
    t_grace = time.time()
    while time.time() - t_grace < pre_grace:
        check_exit()
        rem = int(pre_grace - (time.time() - t_grace)) + 1
        render_screen({"type": "Text Prompt", "content": f"Get Ready... Starting in {rem}s"})
        clock.tick(60)

    # Baseline Phase
    b_dur = cfg.get("baseline_duration", 40)
    marker_outlet.push_sample([M_BASELINE_START, "0"])
    t0 = time.time()
    while time.time() - t0 < b_dur:
        check_exit()
        rem = int(b_dur - (time.time() - t0))
        render_screen({"type": "Text Prompt", "content": f"Baseline Phase ({rem}s)"})
        clock.tick(60)

    marker_outlet.push_sample([M_BASELINE_END, "0"])
    marker_outlet.push_sample([M_RUN_ONSET, "0"])

    # Trial Loops
    for trial in range(1, cfg.get("total_trials", 5) + 1):
        marker_outlet.push_sample([M_TRIAL_INDEX, str(trial)])

        # One target_class marker per trial, sent before any of its steps
        # run, so the Neurofeedback Engine's current_target_class is set for
        # the whole trial (e.g. alternating/counterbalanced classes across a
        # run) before that trial's Feedback Window requests a prediction.
        if trial_target_sequence:
            trial_target = trial_target_sequence[(trial - 1) % len(trial_target_sequence)]
            marker_outlet.push_sample([M_TARGET_CLASS, str(int(trial_target))])

        for step in cfg.get("steps", []):
            marker = step.get("marker", "")
            if marker:
                marker_outlet.push_sample([marker, "0"])

            if step.get("type") == "Feedback Window":
                if feedback_inlet is None:
                    streams = resolve_byprop("name", "Neurofeedback_Out", timeout=0.2)
                    if streams: feedback_inlet = StreamInlet(streams[0])

                prob_value = None
                t_end = time.time() + step.get("duration", 2.0)
                while time.time() < t_end:
                    check_exit()
                    if feedback_inlet:
                        s, _ = feedback_inlet.pull_sample(timeout=0.0)
                        if s and s[0] == "probability":
                            prob_value = float(s[1])
                    render_screen(step, prob_value=prob_value)
                    clock.tick(60)
            else:
                t_end = time.time() + step.get("duration", 2.0)
                while time.time() < t_end:
                    check_exit()
                    render_screen(step)
                    clock.tick(60)

    marker_outlet.push_sample([M_RUN_OFFSET, "0"])
    pygame.quit()

if __name__ == "__main__":
    run()
"""


class ScriptRunnerThread(QThread):
    """Runs an external Python paradigm script and streams its merged stdout/stderr line by line."""
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, script_path):
        """Store the script path; the process is started in run()."""
        super().__init__()
        self.script_path = script_path
        self.process = None

    def run(self):
        """Start the script, emit each output line, then emit its return code when it exits."""
        self.process = subprocess.Popen(
            [sys.executable, self.script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in iter(self.process.stdout.readline, ''):
            if line:
                self.output_signal.emit(line.strip())
        self.process.wait()
        self.finished_signal.emit(self.process.returncode)

    def stop(self):
        """Send SIGTERM to the script process if it was started (does not wait for it)."""
        if self.process:
            self.process.terminate()


class LSLStreamCatcher(QThread):
    """
    Resolves an LSL stream by name and emits each sample as (label, value) strings.

    Intended for the Neurofeedback Engine's 'Neurofeedback_Out' stream, but works
    with any marker-style stream: channel 0 is the label, channel 1 (if present)
    the value.
    """
    sample_signal = pyqtSignal(str, str)  # label, value
    status_signal = pyqtSignal(str)

    def __init__(self, stream_name):
        """Store the name of the stream to resolve."""
        super().__init__()
        self.stream_name = stream_name
        self.is_running = True

    def run(self):
        """Retry resolving the stream every second until found or stopped, then pull samples until stopped."""
        self.status_signal.emit(f"Searching for '{self.stream_name}'...")
        inlet = None
        while self.is_running and inlet is None:
            try:
                streams = resolve_byprop("name", self.stream_name, timeout=1.0)
                if streams:
                    inlet = StreamInlet(streams[0])
                    self.status_signal.emit(f"Connected to '{self.stream_name}'.")
            except Exception:
                pass

        while self.is_running and inlet is not None:
            sample, _ = inlet.pull_sample(timeout=0.5)
            if sample:
                label = str(sample[0])
                value = str(sample[1]) if len(sample) > 1 else ""
                self.sample_signal.emit(label, value)

    def stop(self):
        """Signal the loop to exit and block until the thread finishes."""
        self.is_running = False
        self.wait()


class ImagePathCell(QWidget):
    """Compact path field + browse button used when a row's Content column is an Image."""
    def __init__(self, initial_path="", parent=None):
        """Build the path field (pre-filled with initial_path) and browse button."""
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        self.edit_path = QLineEdit(initial_path)
        self.edit_path.setPlaceholderText("No image selected...")

        btn_browse = QPushButton("\U0001F4C1")
        btn_browse.setFixedWidth(30)
        btn_browse.setToolTip("Browse for a stimulus image")
        btn_browse.clicked.connect(self.browse)

        layout.addWidget(self.edit_path)
        layout.addWidget(btn_browse)

    def browse(self):
        """Open a file dialog and put the chosen image path in the field."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Stimulus Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if path:
            self.edit_path.setText(path)

    def text(self):
        """Return the current image path."""
        return self.edit_path.text()

    def setText(self, value):
        """Set the image path."""
        self.edit_path.setText(value)

    def setEnabled(self, enabled):
        """Enable or disable the path field and the whole cell."""
        self.edit_path.setEnabled(enabled)
        super().setEnabled(enabled)


class FeedbackConfigCell(QWidget):
    """
    Content column widget for a "Feedback Window" step.

    Selects the probability display style (Text, horizontal/vertical progress
    bar, Disk), whether to show the "Feedback" label, and whether to show the
    percentage (bar and Disk styles only; Text always shows it). The chance
    level and Disk radii are fixed constants in RUNNER_CODE.
    """
    FEEDBACK_STYLES = ["Text", "Progress Bar (Horizontal)", "Progress Bar (Vertical)", "Disk"]

    def __init__(self, style_val="Text", show_label_val=True, show_percentage_val=True, parent=None):
        """Build the style combo and the two checkboxes; unknown styles fall back to "Text"."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.combo_style = QComboBox()
        self.combo_style.addItems(self.FEEDBACK_STYLES)
        self.combo_style.setCurrentText(style_val if style_val in self.FEEDBACK_STYLES else "Text")
        layout.addWidget(self.combo_style)

        self.chk_show_label = QCheckBox("Show 'Feedback' Label")
        self.chk_show_label.setChecked(bool(show_label_val))
        layout.addWidget(self.chk_show_label)

        self.chk_show_percentage = QCheckBox("Show Percentage Text")
        self.chk_show_percentage.setChecked(bool(show_percentage_val))
        self.chk_show_percentage.setToolTip(
            "Applies to the Progress Bar and Disk styles only - Text style always shows the percentage."
        )
        layout.addWidget(self.chk_show_percentage)

    def style(self):
        """Return the selected feedback style name (shadows QWidget.style())."""
        return self.combo_style.currentText()

    def show_label(self):
        """Return whether the "Feedback" label should be drawn."""
        return self.chk_show_label.isChecked()

    def show_percentage(self):
        """Return whether the percentage text should be drawn."""
        return self.chk_show_percentage.isChecked()


class ExperimentDesignTab(QWidget):
    """Command center for designing visual paradigms or launching python script experiments."""
    def __init__(self, parent=None):
        """Initialize thread handles and build the UI."""
        super().__init__(parent)
        self.script_thread = None
        self.stream_catcher = None
        self.init_ui()

    def init_ui(self):
        """Create the two sub-tabs: sequence builder and script executor."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)

        self.tab_widget = QTabWidget()

        # TAB 1: Visual Sequence Designer
        self.tab_designer = QWidget()
        self.setup_designer_ui(self.tab_designer)
        self.tab_widget.addTab(self.tab_designer, "Visual Sequence Builder (GUI)")

        # TAB 2: Custom Python Script Launcher
        self.tab_script = QWidget()
        self.setup_script_launcher_ui(self.tab_script)
        self.tab_widget.addTab(self.tab_script, "Custom Script Executor (Code)")

        main_layout.addWidget(self.tab_widget)

    # --- DESIGNER UI ---
    def setup_designer_ui(self, parent):
        """Build the sequence builder: global parameters, target sequence, system markers,
        step table and launch button on the left; the live stream catcher on the right."""
        layout = QHBoxLayout(parent)
        splitter = QSplitter(Qt.Horizontal)

        # Left Column: Configuration & Step Table
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 8, 0)

        # Global Config
        cfg_group = QGroupBox("Paradigm Global Parameters")
        cfg_grid = QGridLayout(cfg_group)

        cfg_grid.addWidget(QLabel("Baseline Duration (s):"), 0, 0)
        self.spin_baseline = QSpinBox()
        self.spin_baseline.setRange(0, 300)
        self.spin_baseline.setValue(40)
        cfg_grid.addWidget(self.spin_baseline, 0, 1)

        cfg_grid.addWidget(QLabel("Total Trial Loops:"), 0, 2)
        self.spin_trials = QSpinBox()
        self.spin_trials.setRange(1, 100)
        self.spin_trials.setValue(5)
        cfg_grid.addWidget(self.spin_trials, 0, 3)

        cfg_grid.addWidget(QLabel("Pre-Baseline Grace (s):"), 1, 0)
        self.spin_pregrace = QSpinBox()
        self.spin_pregrace.setRange(0, 120)
        self.spin_pregrace.setValue(10)
        self.spin_pregrace.setToolTip(
            "Countdown shown before 'baseline_start' fires. Gives the Neurofeedback Engine\n"
            "time to finish connecting to the LSL streams first, so it doesn't miss the\n"
            "baseline_start marker (which is what causes a 0-frame baseline)."
        )
        cfg_grid.addWidget(self.spin_pregrace, 1, 1)

        self.chk_fullscreen = QCheckBox("Launch Fullscreen")
        self.chk_fullscreen.setChecked(False)
        cfg_grid.addWidget(self.chk_fullscreen, 1, 2, 1, 2)

        cfg_grid.addWidget(QLabel("Screen Width (px):"), 2, 0)
        self.spin_screen_w = QSpinBox()
        self.spin_screen_w.setRange(320, 7680)
        self.spin_screen_w.setValue(1024)
        cfg_grid.addWidget(self.spin_screen_w, 2, 1)

        cfg_grid.addWidget(QLabel("Screen Height (px):"), 2, 2)
        self.spin_screen_h = QSpinBox()
        self.spin_screen_h.setRange(240, 4320)
        self.spin_screen_h.setValue(768)
        cfg_grid.addWidget(self.spin_screen_h, 2, 3)

        cfg_grid.addWidget(QLabel("Font Size (px):"), 3, 0)
        self.spin_font_size = QSpinBox()
        self.spin_font_size.setRange(10, 200)
        self.spin_font_size.setValue(36)
        self.spin_font_size.setToolTip("Default font size. Text Prompt / Feedback Window steps can override this with their own Size column value.")
        cfg_grid.addWidget(self.spin_font_size, 3, 1)

        left_layout.addWidget(cfg_group)

        # Trial Target Class Sequence: a per-trial target class, sent as a
        # 'target_class' marker at the start of each trial so the neurofeedback
        # tab's current_target_class can vary across a run (e.g. counterbalanced).
        trial_target_group = QGroupBox("Trial Target Class Sequence")
        trial_target_layout = QGridLayout(trial_target_group)

        trial_target_layout.addWidget(QLabel("Sequence:"), 0, 0)
        self.txt_trial_targets = QLineEdit()
        self.txt_trial_targets.setPlaceholderText(
            "e.g. 0,1,0,1 - cycles to fill Total Trial Loops. Leave blank to skip sending a "
            "target_class marker entirely."
        )
        trial_target_layout.addWidget(self.txt_trial_targets, 0, 1, 1, 2)

        btn_shuffle_targets = QPushButton("Shuffle")
        btn_shuffle_targets.setToolTip(
            "Expands the sequence above to exactly match Total Trial Loops (cycling/repeating "
            "the values you've entered) and then randomizes their order - a balanced but "
            "unpredictable trial order instead of a blocked one."
        )
        btn_shuffle_targets.clicked.connect(self.shuffle_trial_targets)
        trial_target_layout.addWidget(btn_shuffle_targets, 0, 3)

        note = QLabel(
            "One target_class marker is sent per trial, before that trial's steps run, so "
            "ui/neurofeedback_tab.py's current_target_class is set for the whole trial."
        )
        note.setStyleSheet("color: #999999; font-size: 10px;")
        note.setWordWrap(True)
        trial_target_layout.addWidget(note, 1, 0, 1, 4)

        left_layout.addWidget(trial_target_group)

        # System LSL Markers (Advanced): names of the lifecycle markers the runner sends itself
        marker_section = CollapsibleSection("System LSL Markers (Advanced)")
        marker_layout = QGridLayout()
        marker_layout.setSpacing(8)

        self.ent_marker_baseline_start = QLineEdit(DEFAULT_SYSTEM_MARKERS["baseline_start"])
        self.ent_marker_baseline_end = QLineEdit(DEFAULT_SYSTEM_MARKERS["baseline_end"])
        self.ent_marker_run_onset = QLineEdit(DEFAULT_SYSTEM_MARKERS["run_onset"])
        self.ent_marker_run_offset = QLineEdit(DEFAULT_SYSTEM_MARKERS["run_offset"])
        self.ent_marker_trial_index = QLineEdit(DEFAULT_SYSTEM_MARKERS["trial_index"])
        self.ent_marker_target_class = QLineEdit(DEFAULT_SYSTEM_MARKERS["target_class"])

        marker_layout.addWidget(QLabel("Baseline Start:"), 0, 0)
        marker_layout.addWidget(self.ent_marker_baseline_start, 0, 1)
        marker_layout.addWidget(QLabel("Baseline End:"), 0, 2)
        marker_layout.addWidget(self.ent_marker_baseline_end, 0, 3)
        marker_layout.addWidget(QLabel("Run Onset:"), 1, 0)
        marker_layout.addWidget(self.ent_marker_run_onset, 1, 1)
        marker_layout.addWidget(QLabel("Run Offset:"), 1, 2)
        marker_layout.addWidget(self.ent_marker_run_offset, 1, 3)
        marker_layout.addWidget(QLabel("Trial Index:"), 2, 0)
        marker_layout.addWidget(self.ent_marker_trial_index, 2, 1)
        marker_layout.addWidget(QLabel("Target Class:"), 2, 2)
        marker_layout.addWidget(self.ent_marker_target_class, 2, 3)

        note = QLabel(
            "Only change these if a downstream consumer (e.g. a customized Neurofeedback\n"
            "Engine) expects different marker names. Per-step markers (trial_onset,\n"
            "trial_offset, request_feedback, etc.) are set directly in the LSL Marker column below."
        )
        note.setStyleSheet("color: #999999; font-size: 10px;")
        note.setWordWrap(True)
        marker_layout.addWidget(note, 3, 0, 1, 4)

        marker_section.set_content_layout(marker_layout)
        left_layout.addWidget(marker_section)

        # Steps Table
        table_group = QGroupBox("Trial Step Sequence")
        table_layout = QVBoxLayout(table_group)

        self.table_steps = QTableWidget(0, len(COLUMN_LABELS))
        self.table_steps.setHorizontalHeaderLabels(COLUMN_LABELS)
        self.table_steps.horizontalHeader().setSectionResizeMode(COL_CONTENT, QHeaderView.Stretch)
        table_layout.addWidget(self.table_steps)

        # Control Buttons
        btn_layout = QHBoxLayout()
        btn_add = QPushButton("Add Step")
        btn_add.clicked.connect(self.add_table_step)
        btn_rem = QPushButton("Remove Selected")
        btn_rem.clicked.connect(self.remove_table_step)
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_rem)
        table_layout.addLayout(btn_layout)

        left_layout.addWidget(table_group)

        # Launch Controls
        self.btn_run_gui = QPushButton("Run Visual Paradigm (Pygame Engine)")
        self.btn_run_gui.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 12px;")
        self.btn_run_gui.clicked.connect(self.launch_visual_experiment)
        left_layout.addWidget(self.btn_run_gui)

        splitter.addWidget(left_widget)

        # Right Column: live LSL stream catcher for monitoring feedback in the GUI.
        # The Pygame runner reads the same stream independently for its display.
        right_widget = QGroupBox("Live LSL Feedback Stream")
        right_layout = QVBoxLayout(right_widget)

        catch_ctrl_layout = QGridLayout()
        catch_ctrl_layout.addWidget(QLabel("Stream Name:"), 0, 0)
        self.ent_catch_stream_name = QLineEdit("Neurofeedback_Out")
        catch_ctrl_layout.addWidget(self.ent_catch_stream_name, 0, 1)

        self.btn_catch_connect = QPushButton("Connect")
        self.btn_catch_connect.clicked.connect(self.start_stream_catcher)
        catch_ctrl_layout.addWidget(self.btn_catch_connect, 1, 0)

        self.btn_catch_disconnect = QPushButton("Disconnect")
        self.btn_catch_disconnect.setEnabled(False)
        self.btn_catch_disconnect.clicked.connect(self.stop_stream_catcher)
        catch_ctrl_layout.addWidget(self.btn_catch_disconnect, 1, 1)

        right_layout.addLayout(catch_ctrl_layout)

        self.lbl_catch_status = QLabel("Status: Idle")
        self.lbl_catch_status.setStyleSheet("color: #b0b0b0; font-size: 10px;")
        right_layout.addWidget(self.lbl_catch_status)

        feedback_group = QGroupBox("Latest Probability")
        feedback_layout = QVBoxLayout(feedback_group)
        self.lbl_feedback_value = QLabel("--")
        self.lbl_feedback_value.setAlignment(Qt.AlignCenter)
        self.lbl_feedback_value.setStyleSheet("font-size: 28px; font-weight: bold;")
        feedback_layout.addWidget(self.lbl_feedback_value)
        self.bar_feedback = QProgressBar()
        self.bar_feedback.setRange(0, 100)
        feedback_layout.addWidget(self.bar_feedback)
        right_layout.addWidget(feedback_group)

        right_layout.addWidget(QLabel("All Caught Samples:"))
        self.txt_catch_log = QTextEdit()
        self.txt_catch_log.setReadOnly(True)
        self.txt_catch_log.setStyleSheet("background-color: #0c0c0c; color: #00ff66; font-family: monospace;")
        right_layout.addWidget(self.txt_catch_log)

        splitter.addWidget(right_widget)
        splitter.setSizes([700, 400])
        layout.addWidget(splitter)

        self.seed_default_steps()

    # --- Step row construction ---
    def _make_content_widget(self, stype, content_val="", shape_val="Circle", image_val="",
                              feedback_style_val="Text", show_feedback_label_val=True, show_percentage_val=True):
        """Build the Content column widget for a step type (shape combo, image picker,
        feedback options, text field, or a disabled field for Fixation Cross)."""
        if stype == "Shape":
            combo = QComboBox()
            combo.addItems(SHAPE_TYPES)
            combo.setCurrentText(shape_val if shape_val in SHAPE_TYPES else "Circle")
            return combo
        elif stype == "Image":
            return ImagePathCell(image_val)
        elif stype == "Feedback Window":
            return FeedbackConfigCell(feedback_style_val, show_feedback_label_val, show_percentage_val)
        elif stype == "Text Prompt":
            edit = QLineEdit(content_val)
            edit.setPlaceholderText("Enter stimulus text...")
            return edit
        else:  # Fixation Cross - no content needed, only Color/Size apply
            edit = QLineEdit("")
            edit.setPlaceholderText("(not used - see Color / Size)")
            edit.setEnabled(False)
            return edit

    def _add_step_row(self, stype="Text Prompt", content="", shape="Circle", color="Green",
                       size=80, image_path="", duration=2.0, marker="",
                       feedback_style="Text", show_feedback_label=True, show_percentage=True):
        """Append a step row populated with the given values and return its index.

        The Step Type combo's change handler is bound to the row index at creation time.
        """
        row = self.table_steps.rowCount()
        self.table_steps.insertRow(row)

        combo_type = QComboBox()
        combo_type.addItems(STEP_TYPES)
        combo_type.setCurrentText(stype)
        combo_type.currentTextChanged.connect(lambda _t, r=row: self._on_step_type_changed(r))
        self.table_steps.setCellWidget(row, COL_TYPE, combo_type)

        self.table_steps.setCellWidget(
            row, COL_CONTENT,
            self._make_content_widget(stype, content_val=content, shape_val=shape, image_val=image_path,
                                       feedback_style_val=feedback_style, show_feedback_label_val=show_feedback_label,
                                       show_percentage_val=show_percentage)
        )

        combo_color = QComboBox()
        combo_color.addItems(SHAPE_COLORS)
        combo_color.setCurrentText(color)
        self.table_steps.setCellWidget(row, COL_COLOR, combo_color)

        spin_size = QSpinBox()
        spin_size.setRange(5, 500)
        spin_size.setValue(int(size))
        self.table_steps.setCellWidget(row, COL_SIZE, spin_size)

        spin_dur = QDoubleSpinBox()
        spin_dur.setRange(0.0, 600.0)
        spin_dur.setSingleStep(0.5)
        spin_dur.setValue(float(duration))
        self.table_steps.setCellWidget(row, COL_DURATION, spin_dur)

        self.table_steps.setItem(row, COL_MARKER, QTableWidgetItem(marker))

        self._apply_type_relevant_enabling(row, stype)
        return row

    def _apply_type_relevant_enabling(self, row, stype):
        """Enable Color/Size for every step type except Image and set the row height for the type."""
        color_size_relevant = stype != "Image"
        self.table_steps.cellWidget(row, COL_COLOR).setEnabled(color_size_relevant)
        self.table_steps.cellWidget(row, COL_SIZE).setEnabled(color_size_relevant)

        # FeedbackConfigCell stacks three controls and needs a taller row;
        # all other content widgets are single-line.
        self.table_steps.setRowHeight(row, 86 if stype == "Feedback Window" else 34)

    def _on_step_type_changed(self, row):
        """Replace the row's Content widget to match the new step type, carrying over prior values."""
        combo_type = self.table_steps.cellWidget(row, COL_TYPE)
        if combo_type is None:
            return
        stype = combo_type.currentText()

        # Carry over a sensible previous value when swapping widget kinds
        old_widget = self.table_steps.cellWidget(row, COL_CONTENT)
        prior_text, prior_shape, prior_image = "", "Circle", ""
        prior_feedback_style, prior_show_label, prior_show_percentage = "Text", True, True
        if isinstance(old_widget, ImagePathCell):
            prior_image = old_widget.text()
        elif isinstance(old_widget, FeedbackConfigCell):
            prior_feedback_style = old_widget.style()
            prior_show_label = old_widget.show_label()
            prior_show_percentage = old_widget.show_percentage()
        elif isinstance(old_widget, QComboBox):
            prior_shape = old_widget.currentText()
        elif isinstance(old_widget, QLineEdit):
            prior_text = old_widget.text()

        new_widget = self._make_content_widget(
            stype, content_val=prior_text, shape_val=prior_shape, image_val=prior_image,
            feedback_style_val=prior_feedback_style, show_feedback_label_val=prior_show_label,
            show_percentage_val=prior_show_percentage
        )
        self.table_steps.setCellWidget(row, COL_CONTENT, new_widget)

        self._apply_type_relevant_enabling(row, stype)

    def seed_default_steps(self):
        """Populate the table with a default fixation / shape / feedback / rest trial."""
        self._add_step_row(stype="Fixation Cross", content="", shape="Circle", color="White",
                            size=30, duration=1.5, marker="fixation_onset")
        self._add_step_row(stype="Shape", content="", shape="Circle", color="Green",
                            size=80, duration=6.0, marker="trial_onset")
        self._add_step_row(stype="Feedback Window", shape="Circle", color="White",
                            size=48, duration=2.0, marker="request_feedback",
                            feedback_style="Text", show_feedback_label=True)
        self._add_step_row(stype="Text Prompt", content="Rest", shape="Circle", color="White",
                            size=48, duration=3.0, marker="trial_offset")

    def add_table_step(self):
        """Append a default Text Prompt step."""
        self._add_step_row()

    def remove_table_step(self):
        """Remove the currently selected step row, if any."""
        curr_row = self.table_steps.currentRow()
        if curr_row >= 0:
            self.table_steps.removeRow(curr_row)

    def shuffle_trial_targets(self):
        """
        Expand the Trial Target Class Sequence to exactly Total Trial Loops entries
        by cycling the given values, then shuffle them in place in the field.
        Uses the values 0 and 1 if the field is empty.
        """
        import random
        raw = [t.strip() for t in self.txt_trial_targets.text().split(",") if t.strip() != ""]
        if not raw:
            raw = ["0", "1"]
        total = self.spin_trials.value()
        expanded = [raw[i % len(raw)] for i in range(total)]
        random.shuffle(expanded)
        self.txt_trial_targets.setText(", ".join(expanded))

    def compile_json_spec(self):
        """Collect the global parameters, markers, target sequence and step table into the runner config dict."""
        steps = []
        for r in range(self.table_steps.rowCount()):
            combo_type = self.table_steps.cellWidget(r, COL_TYPE)
            stype = combo_type.currentText() if combo_type else "Text Prompt"

            content_widget = self.table_steps.cellWidget(r, COL_CONTENT)
            shape_val, content_val = "Circle", ""
            feedback_style_val, show_feedback_label_val, show_percentage_val = "Text", True, True
            if isinstance(content_widget, FeedbackConfigCell):
                feedback_style_val = content_widget.style()
                show_feedback_label_val = content_widget.show_label()
                show_percentage_val = content_widget.show_percentage()
            elif isinstance(content_widget, QComboBox):
                shape_val = content_widget.currentText()
            elif isinstance(content_widget, (ImagePathCell, QLineEdit)):
                content_val = content_widget.text()

            combo_color = self.table_steps.cellWidget(r, COL_COLOR)
            color_val = combo_color.currentText() if combo_color else "Green"

            spin_size = self.table_steps.cellWidget(r, COL_SIZE)
            size_val = spin_size.value() if spin_size else 80

            spin_dur = self.table_steps.cellWidget(r, COL_DURATION)
            duration_val = spin_dur.value() if spin_dur else 2.0

            marker_item = self.table_steps.item(r, COL_MARKER)
            marker_val = marker_item.text() if marker_item else ""

            steps.append({
                "type": stype,
                "content": content_val,
                "shape": shape_val,
                "color": color_val,
                "size": size_val,
                "duration": duration_val,
                "marker": marker_val,
                "feedback_style": feedback_style_val,
                "show_feedback_label": show_feedback_label_val,
                "show_percentage": show_percentage_val
            })

        # Parses to a list of ints; invalid/non-numeric tokens are dropped
        # rather than aborting the whole sequence. Cycling to fit
        # Total Trial Loops happens at runtime (see RUNNER_CODE), so this
        # stays correct even if Total Trial Loops changes after typing it.
        trial_target_sequence = []
        for tok in self.txt_trial_targets.text().split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                trial_target_sequence.append(int(float(tok)))
            except ValueError:
                pass

        return {
            "baseline_duration": self.spin_baseline.value(),
            "pre_baseline_delay": self.spin_pregrace.value(),
            "total_trials": self.spin_trials.value(),
            "fullscreen": self.chk_fullscreen.isChecked(),
            "screen_width": self.spin_screen_w.value(),
            "screen_height": self.spin_screen_h.value(),
            "font_size": self.spin_font_size.value(),
            "trial_target_sequence": trial_target_sequence,
            "markers": {
                "baseline_start": self.ent_marker_baseline_start.text().strip() or DEFAULT_SYSTEM_MARKERS["baseline_start"],
                "baseline_end": self.ent_marker_baseline_end.text().strip() or DEFAULT_SYSTEM_MARKERS["baseline_end"],
                "run_onset": self.ent_marker_run_onset.text().strip() or DEFAULT_SYSTEM_MARKERS["run_onset"],
                "run_offset": self.ent_marker_run_offset.text().strip() or DEFAULT_SYSTEM_MARKERS["run_offset"],
                "trial_index": self.ent_marker_trial_index.text().strip() or DEFAULT_SYSTEM_MARKERS["trial_index"],
                "target_class": self.ent_marker_target_class.text().strip() or DEFAULT_SYSTEM_MARKERS["target_class"],
            },
            "steps": steps
        }

    def launch_visual_experiment(self):
        """Validate image steps, write the config and runner script to the working directory,
        and start the runner in a detached process (not tracked or stoppable from the GUI)."""
        spec = self.compile_json_spec()

        # Validate image steps up front so a typo'd path fails fast with a
        # clear message instead of showing "[Image not found]" mid-session.
        missing_images = [
            s["content"] for s in spec["steps"]
            if s["type"] == "Image" and (not s["content"] or not os.path.exists(s["content"]))
        ]
        if missing_images:
            QMessageBox.warning(
                self, "Missing Image File(s)",
                "One or more 'Image' steps don't have a valid file selected:\n\n"
                + "\n".join(missing_images or ["(no path set)"])
                + "\n\nUse the button in the Content column to select a file."
            )
            return

        temp_cfg = "temp_experiment_cfg.json"
        temp_runner = "temp_runner.py"

        with open(temp_cfg, "w") as f:
            json.dump(spec, f, indent=2)

        with open(temp_runner, "w") as f:
            f.write(RUNNER_CODE)

        subprocess.Popen([sys.executable, temp_runner, temp_cfg])

    # --- Live LSL Feedback Stream Catcher ---
    def start_stream_catcher(self):
        """Reset the feedback display and start an LSLStreamCatcher for the entered stream name."""
        stream_name = self.ent_catch_stream_name.text().strip()
        if not stream_name:
            return

        self.txt_catch_log.clear()
        self.bar_feedback.setValue(0)
        self.lbl_feedback_value.setText("--")
        self.btn_catch_connect.setEnabled(False)
        self.btn_catch_disconnect.setEnabled(True)

        self.stream_catcher = LSLStreamCatcher(stream_name)
        self.stream_catcher.status_signal.connect(lambda s: self.lbl_catch_status.setText(f"Status: {s}"))
        self.stream_catcher.sample_signal.connect(self._on_caught_sample)
        self.stream_catcher.start()

    def _on_caught_sample(self, label, value):
        """Log a caught sample and, for "probability" samples, update the value label and bar."""
        ts = datetime.now().strftime('%H:%M:%S')
        self.txt_catch_log.append(f"[{ts}] {label}: {value}")

        if label == "probability":
            try:
                pct = float(value) * 100
                pct_clamped = max(0, min(100, pct))
                self.bar_feedback.setValue(int(pct_clamped))
                self.lbl_feedback_value.setText(f"{pct:.1f}%")
            except ValueError:
                pass

    def stop_stream_catcher(self):
        """Stop the stream catcher thread and reset the connect/disconnect buttons."""
        if self.stream_catcher:
            self.stream_catcher.stop()
            self.stream_catcher = None
        self.lbl_catch_status.setText("Status: Disconnected")
        self.btn_catch_connect.setEnabled(True)
        self.btn_catch_disconnect.setEnabled(False)

    # --- SCRIPT LAUNCHER UI ---
    def setup_script_launcher_ui(self, parent):
        """Build the script path picker, launch/halt buttons and the output log."""
        layout = QVBoxLayout(parent)

        cfg_group = QGroupBox("Custom Python Experiment Execution")
        cfg_layout = QGridLayout(cfg_group)

        cfg_layout.addWidget(QLabel("Target Experiment Script:"), 0, 0)
        self.ent_script_path = QLineEdit()

        # Default to experiment_skeleton.py in the current working directory, if present
        default_skel = os.path.join(os.getcwd(), "experiment_skeleton.py")
        if os.path.exists(default_skel):
            self.ent_script_path.setText(default_skel)

        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_script)
        cfg_layout.addWidget(self.ent_script_path, 0, 1)
        cfg_layout.addWidget(btn_browse, 0, 2)

        layout.addWidget(cfg_group)

        ctrl_layout = QHBoxLayout()
        self.btn_run_script = QPushButton("▶ Launch Script Paradigm")
        self.btn_run_script.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 10px;")
        self.btn_run_script.clicked.connect(self.start_script_execution)

        self.btn_stop_script = QPushButton("⏹ Halt Script")
        self.btn_stop_script.setEnabled(False)
        self.btn_stop_script.setStyleSheet("background-color: #b71c1c; color: white; font-weight: bold; padding: 10px;")
        self.btn_stop_script.clicked.connect(self.stop_script_execution)

        ctrl_layout.addWidget(self.btn_run_script)
        ctrl_layout.addWidget(self.btn_stop_script)
        layout.addLayout(ctrl_layout)

        # Terminal Output Log
        log_group = QGroupBox("Script Standard Output / Standard Error")
        log_layout = QVBoxLayout(log_group)
        self.txt_script_log = QTextEdit()
        self.txt_script_log.setReadOnly(True)
        self.txt_script_log.setStyleSheet("background-color: #0c0c0c; color: #00ff66; font-family: monospace;")
        log_layout.addWidget(self.txt_script_log)
        layout.addWidget(log_group)

    def browse_script(self):
        """Open a file dialog to choose the Python script to run."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Python Paradigm Script", "", "Python Files (*.py)")
        if path:
            self.ent_script_path.setText(path)

    def start_script_execution(self):
        """Run the selected script in a ScriptRunnerThread, piping its output into the log."""
        script_path = self.ent_script_path.text()
        if not os.path.exists(script_path):
            QMessageBox.critical(self, "File Not Found", f"Could not find script at path: {script_path}")
            return

        self.txt_script_log.clear()
        self.txt_script_log.append(f"Executing process: {script_path}\n" + "-"*50)
        self.btn_run_script.setEnabled(False)
        self.btn_stop_script.setEnabled(True)

        self.script_thread = ScriptRunnerThread(script_path)
        self.script_thread.output_signal.connect(lambda line: self.txt_script_log.append(line))
        self.script_thread.finished_signal.connect(self.on_script_finished)
        self.script_thread.start()

    def stop_script_execution(self):
        """Terminate the running script and reset the controls."""
        if self.script_thread:
            self.script_thread.stop()
            self.txt_script_log.append("\n Process halted by user.")
            self.on_script_finished(-1)

    def on_script_finished(self, return_code):
        """Re-enable the launch button and log the process return code."""
        self.btn_run_script.setEnabled(True)
        self.btn_stop_script.setEnabled(False)
        self.txt_script_log.append(f"\nProcess terminated with return code: {return_code}")