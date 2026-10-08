# ui/main_window.py
"""Main application window hosting the toolbox tabs and the shared session cache."""
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QComboBox, QLabel, QPushButton
from core.data_manager import DataSession
from ui.online_tab import OnlineStreamingTab
from ui.offline_tab import OfflineAnalysisTab
from ui.decoders_tab import DecoderTrainingTab
from ui.themes import THEMES
from ui.neurofeedback_tab import NeurofeedbackTab
from ui.experiment_tab import ExperimentDesignTab


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
        """Builds the header ribbon (theme selector, fullscreen toggle) and instantiates all tabs."""
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
        
        self.tabs_container.addTab(self.tab_online, "Online Streaming Monitor")
        self.tabs_container.addTab(self.tab_neurofeedback, "Real-Time Neurofeedback Engine")
        self.tabs_container.addTab(self.tab_experiment, "Experiment Design & Presentation")  
        self.tabs_container.addTab(self.tab_offline, "Offline Analytical Processing")
        self.tabs_container.addTab(self.tab_decoder, "Decoder Training / Evaluation")

        global_layout.addWidget(self.tabs_container)

    def toggle_fullscreen_mode(self):
        """Toggles the window between full screen and its normal size, updating the button label."""
        if self.isFullScreen():
            self.showNormal()
            self.btn_fullscreen.setText("Go Fullscreen")
        else:
            self.showFullScreen()
            self.btn_fullscreen.setText("Exit Fullscreen")

    def apply_theme_config(self, theme_name):
        """Applies the named theme's stylesheet and redraws the online plot with its background color."""
        theme = THEMES[theme_name]
        self.setStyleSheet(theme["qss"])
        self.tab_online.figure.patch.set_facecolor(theme["bg"])
        self.tab_online.refresh_canvas_plots()

    @property
    def combo_layout(self):
        """Backward-compatible alias for the Online tab's plot layout combo box (tab_online.combo_layout)."""
        return self.tab_online.combo_layout
