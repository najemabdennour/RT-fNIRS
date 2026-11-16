# ui/themes.py
"""
Application color themes.

Each theme is a set of color tokens ("colors"). From them, build_qss()
generates one application-wide stylesheet and build_palette() a matching
QPalette, so every widget - including dialogs, drop-down lists, tooltips and
widgets no stylesheet rule names - follows the theme instead of the desktop's
own colors. apply_theme() installs both on the QApplication (with Qt's
platform-independent Fusion style) at a chosen base font size; all
font sizes are relative to it.

Widgets never hard-code colors: they declare a semantic role with
set_role(widget, role) and the theme colors it. Roles:
  buttons:  'primary' (default look), 'secondary', 'success', 'danger'
  labels:   'muted' (small grey descriptions), 'title', 'display' (large
            value readout), 'status',
            'ok' / 'warn' / 'error' (status text)
  text:     'console' (log terminals, dark in every theme)

Each THEMES entry also keeps "bg"/"fg" (plot background/foreground, read by
the matplotlib views) and "mpl" (a matplotlib style name).
"""
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

THEMES = {
    "Dark Studio": {
        "colors": {
            "window": "#1c1c1e", "panel": "#2c2c2e", "input": "#3a3a3c", "alt_row": "#323234",
            "text": "#f5f5f7", "muted": "#a1a1a6", "border": "#48484a", "header": "#38383a",
            "accent": "#0a84ff", "accent_hover": "#409cff", "accent_text": "#ffffff",
            "secondary": "#48484a", "secondary_hover": "#5a5a5e", "secondary_text": "#f5f5f7",
            "success": "#2e8b3e", "success_hover": "#35a047", "danger": "#c0392b", "danger_hover": "#d64535",
            "disabled": "#3a3a3c", "disabled_text": "#7c7c80",
            "ok": "#32d74b", "warn": "#ff9f0a", "error": "#ff453a",
            "console_bg": "#0c0c0c", "console_text": "#00ff66", "console_border": "#2a2a2a",
        },
        "mpl": "dark_background", "bg": "#2c2c2e", "fg": "#f5f5f7",
    },
    "Classic Light": {
        "colors": {
            "window": "#f2f2f7", "panel": "#ffffff", "input": "#ffffff", "alt_row": "#f6f6f9",
            "text": "#1d1d1f", "muted": "#6e6e73", "border": "#c7c7cc", "header": "#e9e9ee",
            "accent": "#0071e3", "accent_hover": "#0a7cf0", "accent_text": "#ffffff",
            "secondary": "#e5e5ea", "secondary_hover": "#d8d8de", "secondary_text": "#1d1d1f",
            "success": "#2e7d32", "success_hover": "#388e3c", "danger": "#c62828", "danger_hover": "#d32f2f",
            "disabled": "#e5e5ea", "disabled_text": "#a1a1a6",
            "ok": "#1e8e3e", "warn": "#b26a00", "error": "#d93025",
            "console_bg": "#1e1f24", "console_text": "#5fd47c", "console_border": "#c7c7cc",
        },
        "mpl": "seaborn-v0_8-whitegrid", "bg": "#ffffff", "fg": "#1d1d1f",
    },
    "Cyberpunk Neon": {
        "colors": {
            "window": "#0b0c10", "panel": "#1f2833", "input": "#0b0c10", "alt_row": "#141a22",
            "text": "#c5c6c7", "muted": "#7f8a96", "border": "#45f3ff", "header": "#1f2833",
            "accent": "#1f2833", "accent_hover": "#45f3ff", "accent_text": "#45f3ff",
            "secondary": "#0b0c10", "secondary_hover": "#1f2833", "secondary_text": "#66fcf1",
            "success": "#0f3d2e", "success_hover": "#145c43", "danger": "#4a0f1f", "danger_hover": "#6b1530",
            "disabled": "#141a22", "disabled_text": "#4f5a66",
            "ok": "#66fcf1", "warn": "#ffcc33", "error": "#ff4f79",
            "console_bg": "#05060a", "console_text": "#66fcf1", "console_border": "#45f3ff",
        },
        "mpl": "dark_background", "bg": "#1f2833", "fg": "#66fcf1",
    },
    "Matrix Hacker": {
        "colors": {
            "window": "#000000", "panel": "#050a05", "input": "#000000", "alt_row": "#071207",
            "text": "#00ff00", "muted": "#00a000", "border": "#00ff00", "header": "#062006",
            "accent": "#000000", "accent_hover": "#00ff00", "accent_text": "#00ff00",
            "secondary": "#000000", "secondary_hover": "#062006", "secondary_text": "#00ff00",
            "success": "#003300", "success_hover": "#005500", "danger": "#330000", "danger_hover": "#550000",
            "disabled": "#020802", "disabled_text": "#006000",
            "ok": "#00ff00", "warn": "#ccff00", "error": "#ff3333",
            "console_bg": "#000000", "console_text": "#00ff00", "console_border": "#00ff00",
        },
        "mpl": "dark_background", "bg": "black", "fg": "#00FF00",
    },
}

# Accent-colored buttons keep the accent text color on hover only for themes whose hover is light.
_INVERT_ON_HOVER = {"Cyberpunk Neon", "Matrix Hacker"}


DEFAULT_FONT_PT = 10
FONT_RANGE_PT = (7, 24)
_current_font_pt = DEFAULT_FONT_PT


def build_qss(c, invert_hover=False, font_pt=DEFAULT_FONT_PT):
    """
    Application stylesheet from a theme's color tokens. Every font size is
    relative to font_pt (the base size, in points), so one setting scales
    the whole interface.
    """
    hover_text = c['window'] if invert_hover else c['accent_text']
    f = font_pt
    small, large, display = max(6, f - 2), f + 1, round(f * 2.6)
    return f"""
        QMainWindow, QDialog {{ background-color: {c['window']}; }}
        QWidget {{ color: {c['text']}; font-size: {f}pt; }}
        QToolTip {{ background-color: {c['panel']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 4px; }}

        QGroupBox {{
            font-weight: bold; border: 1px solid {c['border']}; border-radius: 6px;
            margin-top: 10px; padding: 8px 6px 6px 6px; background-color: {c['panel']};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 4px;
            color: {c['text']}; background-color: {c['panel']};
        }}

        QPushButton {{
            background-color: {c['accent']}; color: {c['accent_text']}; border: 1px solid {c['accent']};
            border-radius: 4px; padding: 5px 10px; font-weight: bold;
        }}
        QPushButton:hover {{ background-color: {c['accent_hover']}; border-color: {c['accent_hover']}; color: {hover_text}; }}
        QPushButton[role="secondary"] {{ background-color: {c['secondary']}; color: {c['secondary_text']}; border: 1px solid {c['border']}; }}
        QPushButton[role="secondary"]:hover {{ background-color: {c['secondary_hover']}; color: {c['secondary_text']}; }}
        QPushButton[role="success"] {{ background-color: {c['success']}; color: #ffffff; border-color: {c['success']}; padding: 8px 12px; }}
        QPushButton[role="success"]:hover {{ background-color: {c['success_hover']}; border-color: {c['success_hover']}; color: #ffffff; }}
        QPushButton[role="danger"] {{ background-color: {c['danger']}; color: #ffffff; border-color: {c['danger']}; }}
        QPushButton[role="danger"]:hover {{ background-color: {c['danger_hover']}; border-color: {c['danger_hover']}; color: #ffffff; }}
        QPushButton:disabled, QPushButton[role]:disabled {{
            background-color: {c['disabled']}; color: {c['disabled_text']}; border-color: {c['disabled']};
        }}
        QPushButton#SectionHeader {{
            text-align: left; padding: 9px 14px; font-size: {large}pt; background-color: {c['header']};
            color: {c['text']}; border: 1px solid {c['border']}; border-radius: 5px;
        }}
        QPushButton#SectionHeader:hover {{ background-color: {c['secondary_hover']}; color: {c['text']}; }}
        QPushButton#SectionHeader:checked {{ border-bottom-left-radius: 0; border-bottom-right-radius: 0; }}
        QWidget#ContentContainer {{
            background-color: {c['panel']}; border: 1px solid {c['border']}; border-top: none;
            border-bottom-left-radius: 5px; border-bottom-right-radius: 5px;
        }}

        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit,
        QListWidget, QListView, QTableWidget, QTableView, QTreeView {{
            background-color: {c['input']}; color: {c['text']}; border: 1px solid {c['border']};
            border-radius: 4px; selection-background-color: {c['accent_hover'] if invert_hover else c['accent']};
            selection-color: {c['window'] if invert_hover else c['accent_text']};
        }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ padding: 2px 4px; min-height: 18px; }}
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
            background-color: {c['disabled']}; color: {c['disabled_text']};
        }}
        QComboBox QAbstractItemView {{ background-color: {c['input']}; color: {c['text']}; border: 1px solid {c['border']}; }}
        QTableWidget, QTableView {{ gridline-color: {c['border']}; alternate-background-color: {c['alt_row']}; }}
        QListWidget::item {{ padding: 3px; }}
        QHeaderView::section {{
            background-color: {c['header']}; color: {c['text']}; font-weight: bold; font-size: {f}pt;
            padding: 5px 6px; border: none; border-right: 1px solid {c['border']}; border-bottom: 1px solid {c['border']};
        }}
        QTableCornerButton::section {{ background-color: {c['header']}; border: none; }}

        QTabWidget::pane {{ border: 1px solid {c['border']}; background-color: {c['window']}; top: -1px; }}
        QTabBar::tab {{
            background-color: {c['header']}; color: {c['muted']}; border: 1px solid {c['border']};
            border-bottom: none; padding: 6px 12px; margin-right: 2px;
            border-top-left-radius: 4px; border-top-right-radius: 4px;
        }}
        QTabBar::tab:selected {{ background-color: {c['window']}; color: {c['text']}; border-top: 2px solid {c['accent_hover'] if invert_hover else c['accent']}; }}
        QTabBar::tab:hover:!selected {{ color: {c['text']}; }}

        QProgressBar {{ background-color: {c['input']}; color: {c['text']}; border: 1px solid {c['border']}; border-radius: 4px; text-align: center; }}
        QProgressBar::chunk {{ background-color: {c['accent_hover'] if invert_hover else c['accent']}; border-radius: 3px; }}
        QSplitter::handle {{ background-color: {c['border']}; }}
        QScrollArea {{ background: transparent; border: none; }}

        QLabel[role="muted"] {{ color: {c['muted']}; font-size: {small}pt; }}
        QLabel[role="title"] {{ color: {c['text']}; font-weight: bold; font-size: {large}pt; }}
        QLabel[role="display"] {{ color: {c['text']}; font-weight: bold; font-size: {display}pt; }}
        QLabel[role="status"] {{ color: {c['muted']}; font-weight: bold; }}
        QLabel[role="ok"] {{ color: {c['ok']}; font-weight: bold; }}
        QLabel[role="warn"] {{ color: {c['warn']}; font-weight: bold; }}
        QLabel[role="error"] {{ color: {c['error']}; font-weight: bold; }}

        QTextEdit[role="console"], QPlainTextEdit[role="console"] {{
            background-color: {c['console_bg']}; color: {c['console_text']}; border: 1px solid {c['console_border']};
            border-radius: 6px; padding: 8px; font-family: 'Consolas', 'DejaVu Sans Mono', 'Courier New', monospace;
            font-size: {f}pt; selection-background-color: #3a6ea5; selection-color: #ffffff;
        }}
    """


def build_palette(c):
    """QPalette from a theme's color tokens (covers what no stylesheet rule styles)."""
    pal = QPalette()
    roles = {
        QPalette.Window: c['window'], QPalette.WindowText: c['text'], QPalette.Base: c['input'],
        QPalette.AlternateBase: c['alt_row'], QPalette.Text: c['text'], QPalette.Button: c['secondary'],
        QPalette.ButtonText: c['secondary_text'], QPalette.BrightText: c['error'], QPalette.ToolTipBase: c['panel'],
        QPalette.ToolTipText: c['text'], QPalette.Highlight: c['accent'], QPalette.HighlightedText: c['accent_text'],
        QPalette.Link: c['accent'], QPalette.PlaceholderText: c['muted'], QPalette.Light: c['panel'],
        QPalette.Midlight: c['header'], QPalette.Mid: c['border'], QPalette.Dark: c['border'], QPalette.Shadow: '#000000',
    }
    for role, color in roles.items():
        pal.setColor(role, QColor(color))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(c['disabled_text']))
    return pal


def apply_theme(name, font_pt=None):
    """
    Installs the named theme (Fusion style, palette and stylesheet) on the
    running QApplication, at font_pt points (default: the size last applied).
    """
    global _current_font_pt
    if font_pt is not None:
        _current_font_pt = int(min(max(font_pt, FONT_RANGE_PT[0]), FONT_RANGE_PT[1]))
    app = QApplication.instance()
    colors = THEMES[name]["colors"]
    if app.style().objectName().lower() != "fusion":
        app.setStyle("Fusion")
    app.setPalette(build_palette(colors))
    font = app.font()
    font.setPointSize(_current_font_pt)
    app.setFont(font)
    app.setStyleSheet(build_qss(colors, invert_hover=name in _INVERT_ON_HOVER, font_pt=_current_font_pt))


def set_role(widget, role):
    """Gives a widget a theme role (see module docstring) and re-applies the stylesheet to it."""
    widget.setProperty("role", role)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()
