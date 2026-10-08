# ui/themes.py
"""
Application color themes.

Each THEMES entry holds a Qt stylesheet ("qss"), a matplotlib style name
("mpl") and the plot background/foreground colors ("bg", "fg").
"""
THEMES = {
    "Dark Studio": {
        "qss": """
            QMainWindow { background-color: #1c1c1e; color: #f5f5f7; }
            QGroupBox { font-weight: bold; border: 1px solid #3a3a3c; border-radius: 6px; margin-top: 8px; padding: 6px; background-color: #2c2c2e; color: #f5f5f7; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }
            QPushButton { background-color: #0a84ff; color: white; border-radius: 4px; padding: 4px 8px; font-weight: bold; border: none; }
            QPushButton:disabled { background-color: #48484a; color: #8e8e93; }
            QLineEdit, QListWidget, QComboBox { background-color: #3a3a3c; color: #f5f5f7; border: 1px solid #48484a; border-radius: 4px; padding: 2px; font-size: 11px; }
            QListWidget::item:selected { background-color: #0a84ff; color: white; }
            QCheckBox { color: #f5f5f7; font-size: 11px; }
        """,
        "mpl": "dark_background", "bg": "#2c2c2e", "fg": "#f5f5f7"
    },
    "Classic Light": {
        "qss": """
            QMainWindow { background-color: #f5f5f7; color: #1d1d1f; }
            QGroupBox { font-weight: bold; border: 1px solid #d2d2d7; border-radius: 6px; margin-top: 8px; padding: 6px; background-color: #ffffff; color: #1d1d1f; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }
            QPushButton { background-color: #0071e3; color: white; border-radius: 4px; padding: 4px 8px; font-weight: bold; border: none; }
            QPushButton:disabled { background-color: #aeaeaf; color: #e2e2e2; }
            QLineEdit, QListWidget, QComboBox { background-color: #ffffff; color: #1d1d1f; border: 1px solid #d2d2d7; border-radius: 4px; padding: 2px; font-size: 11px; }
            QListWidget::item:selected { background-color: #0071e3; color: white; }
            QCheckBox { color: #1d1d1f; font-size: 11px; }
        """,
        "mpl": "seaborn-v0_8-whitegrid", "bg": "#ffffff", "fg": "#1d1d1f"
    },
    "Cyberpunk Neon": {
        "qss": """
            QMainWindow { background-color: #0b0c10; color: #c5c6c7; }
            QGroupBox { font-weight: bold; border: 1px solid #45f3ff; border-radius: 6px; margin-top: 8px; padding: 6px; background-color: #1f2833; color: #45f3ff; }
            QPushButton { background-color: #1f2833; color: #45f3ff; border: 1px solid #45f3ff; border-radius: 4px; padding: 4px 8px; font-weight: bold; }
            QPushButton:hover { background-color: #45f3ff; color: #0b0c10; }
            QLineEdit, QListWidget, QComboBox { background-color: #0b0c10; color: #66fcf1; border: 1px solid #45f3ff; border-radius: 4px; font-size: 11px; }
            QListWidget::item:selected { background-color: #66fcf1; color: #0b0c10; font-weight: bold; }
            QCheckBox { color: #66fcf1; font-size: 11px; }
        """,
        "mpl": "dark_background", "bg": "#1f2833", "fg": "#66fcf1"
    },
    "Matrix Hacker": {
        "qss": """
            QMainWindow { background-color: black; color: #00FF00; }
            QGroupBox { font-weight: bold; border: 1px solid #00FF00; border-radius: 4px; margin-top: 8px; padding: 6px; background-color: black; color: #00FF00; }
            QPushButton { background-color: black; color: #00FF00; border: 1px solid #00FF00; padding: 4px; }
            QPushButton:hover { background-color: #00FF00; color: black; }
            QLineEdit, QListWidget, QComboBox { background-color: black; color: #00FF00; border: 1px solid #00FF00; font-size: 11px; }
            QListWidget::item:selected { background-color: #00FF00; color: black; }
            QCheckBox { color: #00FF00; font-size: 11px; }
        """,
        "mpl": "dark_background", "bg": "black", "fg": "#00FF00"
    }
}