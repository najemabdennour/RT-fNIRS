# ui/main_window.py
"""Main application window hosting the toolbox tabs and the shared session cache."""
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QComboBox, QLabel, QPushButton, QSpinBox

import config
from core import config_io
from core.data_manager import DataSession
from ui.online_tab import OnlineStreamingTab
from ui.offline_tab import OfflineAnalysisTab
from ui.decoders_tab import DecoderTrainingTab
from ui.themes import THEMES, apply_theme, DEFAULT_FONT_PT, FONT_RANGE_PT
from ui.neurofeedback_tab import NeurofeedbackTab
from ui.experiment_tab import ExperimentDesignTab
from ui.settings_tab import SettingsTab


class LSLVisualizerApp(QMainWindow):
    """Top-level window: theme/fullscreen header plus the tab container for every toolbox module."""
    def __init__(self):
        """Creates the global session cache, builds the UI and applies the default theme."""
        super().__init__()
        self.setWindowTitle("Advanced Real Time fNIRS toolbox")
        self.setGeometry(60, 60, 1300, 900)
        
        # Shared live-data cache, filled by the Online Streaming tab.
        self.global_session_cache = DataSession()
        
        self.init_global_ui()
        self.apply_theme_config("Dark Studio")

    def init_global_ui(self):
        """Builds the header ribbon (theme selector, font size, fullscreen toggle) and instantiates all tabs."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        global_layout = QVBoxLayout(central_widget)
        global_layout.setContentsMargins(4, 4, 4, 4)
        
        header_ribbon = QHBoxLayout()
        header_ribbon.addWidget(QLabel("<b>Workspace Environment Styling:</b>"))
        self.combo_theme = QComboBox()
        self.combo_theme.addItems(list(THEMES.keys()))
        self.combo_theme.currentTextChanged.connect(self.apply_theme_config)
        header_ribbon.addWidget(self.combo_theme)

        header_ribbon.addSpacing(15)
        header_ribbon.addWidget(QLabel("<b>Font size:</b>"))
        self.spin_font_size = QSpinBox()
        self.spin_font_size.setRange(*FONT_RANGE_PT)
        self.spin_font_size.setSuffix(" pt")
        self.spin_font_size.setValue(int(getattr(config, 'UI_FONT_SIZE', DEFAULT_FONT_PT)))
        self.spin_font_size.setToolTip("Font size of the whole application. The chosen size is saved to "
                                       "config.py (UI_FONT_SIZE) and used at the next start.")
        self.spin_font_size.valueChanged.connect(self.apply_font_size)
        header_ribbon.addWidget(self.spin_font_size)
        # Saves the size to config.py once the user stops changing it.
        self._font_save_timer = QTimer(self)
        self._font_save_timer.setSingleShot(True)
        self._font_save_timer.setInterval(1000)
        self._font_save_timer.timeout.connect(self._save_font_size)

        header_ribbon.addSpacing(15)
        self.btn_fullscreen = QPushButton("Go Fullscreen")
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen_mode)
        header_ribbon.addWidget(self.btn_fullscreen)
        
        header_ribbon.addStretch()
        global_layout.addLayout(header_ribbon)
        
        self.tabs_container = QTabWidget()
        
        self.tab_online = OnlineStreamingTab(self)
        self.tab_offline = OfflineAnalysisTab(self)
        self.tab_decoder = DecoderTrainingTab(self)
        self.tab_neurofeedback = NeurofeedbackTab(self)
        self.tab_experiment = ExperimentDesignTab(self)
        self.tab_settings = SettingsTab(self)
        
        self.tabs_container.addTab(self.tab_online, "Online Streaming Monitor")
        self.tabs_container.addTab(self.tab_neurofeedback, "Real-Time Neurofeedback Engine")
        self.tabs_container.addTab(self.tab_experiment, "Experiment Design & Presentation")  
        self.tabs_container.addTab(self.tab_offline, "Offline Analytical Processing")
        self.tabs_container.addTab(self.tab_decoder, "Decoder Training / Evaluation")
        self.tabs_container.addTab(self.tab_settings, "Settings")

        global_layout.addWidget(self.tabs_container)

    def toggle_fullscreen_mode(self):
        """Toggles the window between full screen and its normal size, updating the button label."""
        if self.isFullScreen():
            self.showNormal()
            self.btn_fullscreen.setText("Go Fullscreen")
        else:
            self.showFullScreen()
            self.btn_fullscreen.setText("Exit Fullscreen")

    def apply_font_size(self, size):
        """Re-applies the current theme at the new font size and schedules saving it to config.py."""
        self.apply_theme_config(self.combo_theme.currentText())
        if int(size) != int(getattr(config, 'UI_FONT_SIZE', DEFAULT_FONT_PT)):
            self._font_save_timer.start()

    def _save_font_size(self):
        """Writes the chosen font size into config.py (UI_FONT_SIZE), in place, keeping the file's comments."""
        size = self.spin_font_size.value()
        try:
            cfg = config_io.ConfigFile(config.__file__.replace('.pyc', '.py'))
            cfg.save({'UI_FONT_SIZE': size}, backup=False)
            config.UI_FONT_SIZE = size
        except Exception as e:
            print(f"[Settings] Could not save the font size to config.py: {e}")

    def apply_theme_config(self, theme_name):
        """Applies the named theme's stylesheet and redraws the online plots and 2D/3D layouts in its colors."""
        theme = THEMES[theme_name]
        apply_theme(theme_name, self.spin_font_size.value() if hasattr(self, 'spin_font_size') else None)
        self.tab_online.figure.patch.set_facecolor(theme["bg"])
        self.tab_online.refresh_canvas_plots()
        self.tab_online.montage_view.redraw()
        self.tab_online.montage_view_3d.redraw()

    @property
    def combo_layout(self):
        """Backward-compatible alias for the Online tab's plot layout combo box (tab_online.combo_layout)."""
        return self.tab_online.combo_layout
