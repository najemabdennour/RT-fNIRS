# ui/components.py
"""Reusable widgets shared across tabs: a collapsible section and a reorderable pipeline step table."""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QCheckBox
)
from PyQt5.QtCore import Qt


class CollapsibleSection(QWidget):
    """Custom accordion-style drop-down widget that hides its content layout by default."""
    def __init__(self, title, parent=None):
        """Builds the checkable header button and the initially hidden content container."""
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(4)

        self.toggle_button = QPushButton(f"► {title}")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                font-weight: bold;
                padding: 10px 14px;
                background-color: #262626;
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                color: #ffffff;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #333333;
                border-color: #4c4c4c;
            }
            QPushButton:checked {
                background-color: #2d2d2d;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }
        """)
        self.toggle_button.clicked.connect(self.on_toggle_triggered)

        self.content_container = QWidget()
        self.content_container.setObjectName("ContentContainer")
        self.content_container.setStyleSheet("""
            QWidget#ContentContainer {
                background-color: #1c1c1c;
                border: 1px solid #3d3d3d;
                border-top: none;
                border-bottom-left-radius: 5px;
                border-bottom-right-radius: 5px;
            }
        """)
        self.content_container.setVisible(False)

        self.main_layout.addWidget(self.toggle_button)
        self.main_layout.addWidget(self.content_container)

    def set_content_layout(self, layout):
        """Installs layout (with 12 px margins) as the section's collapsible content."""
        layout.setContentsMargins(12, 12, 12, 12)
        self.content_container.setLayout(layout)

    def on_toggle_triggered(self, checked):
        """Shows or hides the content and flips the header arrow to match."""
        self.content_container.setVisible(checked)
        if checked:
            self.toggle_button.setText(self.toggle_button.text().replace("►", "▼"))
        else:
            self.toggle_button.setText(self.toggle_button.text().replace("▼", "►"))

    def set_expanded(self, expanded):
        """Programmatically forces the menu section to open or close on initialization."""
        self.toggle_button.setChecked(expanded)
        self.content_container.setVisible(expanded)
        if expanded:
            self.toggle_button.setText(self.toggle_button.text().replace("►", "▼"))
        else:
            self.toggle_button.setText(self.toggle_button.text().replace("▼", "►"))


class StepDescriptionWidget(QWidget):
    """Compact title + subtitle label pair used inside a pipeline step row."""
    def __init__(self, title, subtitle, parent=None):
        """Creates the bold title label and the word-wrapped subtitle label."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(3)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("font-weight: bold; color: #ffffff; font-size: 12px;")

        lbl_sub = QLabel(subtitle)
        lbl_sub.setStyleSheet("color: #b0b0b0; font-size: 10px; font-style: normal;")
        lbl_sub.setWordWrap(True)

        layout.addWidget(lbl_title)
        layout.addWidget(lbl_sub)


class PipelineStepTable(QWidget):
    """
    Reusable reorderable pipeline step table: each row has an 'Execute' toggle
    and an optional 'Checkpoint' (save) toggle, with Move Up/Down controls.

    Shared by the offline preprocessing pipeline (ui/offline_tab.py) and the
    real-time neurofeedback pipeline (ui/neurofeedback_tab.py) so the two
    don't maintain separate copies of the same table/reorder-button plumbing.
    """
    def __init__(self, steps, show_checkpoint=True, row_height=56, parent=None):
        """
        Args:
            steps: list of (step_id, title, subtitle, default_run) or
                (step_id, title, subtitle, default_run, default_save) tuples.
                The 5th element is only used when show_checkpoint=True.
            show_checkpoint: Whether to add the 'Checkpoint' (save) column.
            row_height: Height of each step row in pixels.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.show_checkpoint = show_checkpoint

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        row_layout = QHBoxLayout()

        n_cols = 3 if show_checkpoint else 2
        self.table = QTableWidget(len(steps), n_cols)
        headers = ["Preprocessing Step Profile", "Execute"] + (["Checkpoint"] if show_checkpoint else [])
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(True)
        self.table.setAlternatingRowColors(True)
        self.table.setColumnWidth(0, 310)
        self.table.setColumnWidth(1, 80)
        if show_checkpoint:
            self.table.setColumnWidth(2, 90)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                alternate-background-color: #282828;
                gridline-color: #383838;
                border: 1px solid #454545;
                border-radius: 4px;
            }
            QTableWidget::item {
                border-bottom: 1px solid #2d2d2d;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #ffffff;
                font-weight: bold;
                padding: 8px;
                border: 1px solid #3e3e3e;
                font-size: 11px;
            }
        """)
        # Fit the table to the step count, capped at 6 visible rows (the rest scroll).
        visible_rows = min(len(steps), 6)
        self.table.setFixedHeight(row_height * visible_rows + 40)

        for idx, step in enumerate(steps):
            s_id, name, desc, run_val = step[0], step[1], step[2], step[3]
            save_val = step[4] if (show_checkpoint and len(step) > 4) else False
            self.table.setRowHeight(idx, row_height)

            id_item = QTableWidgetItem()
            id_item.setData(Qt.UserRole, s_id)
            self.table.setItem(idx, 0, id_item)
            self.table.setCellWidget(idx, 0, StepDescriptionWidget(name, desc))

            self.table.setCellWidget(idx, 1, self._make_checkbox_cell(run_val))
            if show_checkpoint:
                self.table.setCellWidget(idx, 2, self._make_checkbox_cell(save_val))

        row_layout.addWidget(self.table, stretch=5)

        vbox_btns = QVBoxLayout()
        btn_up = QPushButton("▲ Move Up")
        btn_down = QPushButton("▼ Move Down")
        btn_up.setStyleSheet("padding: 6px; font-weight: bold; min-width: 90px;")
        btn_down.setStyleSheet("padding: 6px; font-weight: bold; min-width: 90px;")
        btn_up.clicked.connect(self._move_up)
        btn_down.clicked.connect(self._move_down)
        vbox_btns.addWidget(btn_up)
        vbox_btns.addWidget(btn_down)
        vbox_btns.addStretch()
        row_layout.addLayout(vbox_btns, stretch=1)

        outer.addLayout(row_layout)

    @staticmethod
    def _make_checkbox_cell(checked):
        """Returns a widget holding a single centered checkbox set to checked."""
        chk = QCheckBox()
        chk.setChecked(checked)
        cell = QWidget()
        l = QHBoxLayout(cell)
        l.addWidget(chk)
        l.setAlignment(Qt.AlignCenter)
        l.setContentsMargins(0, 0, 0, 0)
        return cell

    def _move_up(self):
        """Moves the selected step one row up and keeps it selected."""
        row = self.table.currentRow()
        if row <= 0: return
        self._swap_rows(row, row - 1)
        self.table.setCurrentCell(row - 1, 0)

    def _move_down(self):
        """Moves the selected step one row down and keeps it selected."""
        row = self.table.currentRow()
        if row < 0 or row >= self.table.rowCount() - 1: return
        self._swap_rows(row, row + 1)
        self.table.setCurrentCell(row + 1, 0)

    def _swap_rows(self, r1, r2):
        """
        Swaps two rows' step ids, description widgets and checkbox states.
        Cell widgets can't be moved between cells, so descriptions are rebuilt.
        """
        id_item1 = self.table.item(r1, 0)
        id_item2 = self.table.item(r2, 0)
        id1, id2 = id_item1.data(Qt.UserRole), id_item2.data(Qt.UserRole)
        id_item1.setData(Qt.UserRole, id2)
        id_item2.setData(Qt.UserRole, id1)

        widget1 = self.table.cellWidget(r1, 0)
        widget2 = self.table.cellWidget(r2, 0)
        title1, sub1 = widget1.layout().itemAt(0).widget().text(), widget1.layout().itemAt(1).widget().text()
        title2, sub2 = widget2.layout().itemAt(0).widget().text(), widget2.layout().itemAt(1).widget().text()
        self.table.setCellWidget(r1, 0, StepDescriptionWidget(title2, sub2))
        self.table.setCellWidget(r2, 0, StepDescriptionWidget(title1, sub1))

        for col in range(1, self.table.columnCount()):
            chk1 = self.table.cellWidget(r1, col).findChild(QCheckBox)
            chk2 = self.table.cellWidget(r2, col).findChild(QCheckBox)
            v1, v2 = chk1.isChecked(), chk2.isChecked()
            chk1.setChecked(v2)
            chk2.setChecked(v1)

    def compiled_steps(self):
        """Returns the current row order as a list of {'id','run'[,'save']} dicts."""
        result = []
        for row in range(self.table.rowCount()):
            s_id = self.table.item(row, 0).data(Qt.UserRole)
            run_flag = self.table.cellWidget(row, 1).findChild(QCheckBox).isChecked()
            entry = {'id': s_id, 'run': run_flag}
            if self.show_checkpoint:
                entry['save'] = self.table.cellWidget(row, 2).findChild(QCheckBox).isChecked()
            result.append(entry)
        return result