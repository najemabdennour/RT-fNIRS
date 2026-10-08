# ui/exception_handler.py
"""
Global handler for uncaught exceptions: reports them verbosely and keeps the
application running instead of exiting.

Every uncaught exception - in a Qt slot on the GUI thread, in a QThread
worker's run(), or in a plain Python thread - is printed to stderr with a
timestamp, its origin thread and the full traceback, and shown in an error
dialog whose "Show Details..." section holds the same traceback.
"""
import sys
import threading
import traceback
from datetime import datetime

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMessageBox

_reporter = None


class _ExceptionReporter(QObject):
    """
    Shows exception dialogs on the GUI thread.

    Reports arrive through the `report` signal over a queued connection, even
    from the GUI thread: the excepthook returns at once and the dialog opens
    from the main event loop afterwards, rather than running a modal loop
    inside the excepthook (and Qt widgets may only be created on the GUI
    thread anyway). Only
    one dialog is open at a time: exceptions raised while it is open (e.g. a
    timer slot failing repeatedly) are counted and left to the terminal output
    rather than stacking up dialogs.
    """
    report = pyqtSignal(str, str, str)  # title, message, full traceback

    def __init__(self):
        """Connects the report signal to the dialog slot (always queued)."""
        super().__init__()
        self._dialog_open = False
        self._suppressed = 0
        self.report.connect(self._show_dialog, Qt.QueuedConnection)

    def _show_dialog(self, title, message, details):
        """Opens a modal error dialog, or counts the report if one is already open."""
        if self._dialog_open:
            self._suppressed += 1
            return

        self._dialog_open = True
        try:
            box = QMessageBox(QApplication.activeWindow())
            box.setIcon(QMessageBox.Critical)
            box.setWindowTitle("Unhandled Exception")
            box.setText(title)
            box.setInformativeText(
                f"{message}\n\nThe application is still running, but the action that "
                f"failed may be incomplete. The full traceback is printed in the terminal "
                f"and available under 'Show Details...'."
            )
            box.setDetailedText(details)
            box.exec_()
        finally:
            self._dialog_open = False

        if self._suppressed:
            print(f"[Exception Handler] {self._suppressed} further exception(s) occurred while the "
                  f"error dialog was open - see their tracebacks above.", file=sys.stderr, flush=True)
            self._suppressed = 0


def _current_thread_label():
    """Describes the thread an exception was raised in, e.g. 'GUI thread' or 'QThread NeurofeedbackWorker'."""
    app = QApplication.instance()
    current = QThread.currentThread()
    if app is not None and current is app.thread():
        return "GUI thread"
    if type(current) is not QThread:
        return f"QThread {type(current).__name__}"
    return f"thread '{threading.current_thread().name}'"


def report_exception(exc_type, exc_value, exc_tb, origin=None):
    """
    Prints a full, timestamped traceback to stderr and, once the handler is
    installed, shows it in an error dialog. Safe to call from any thread.
    """
    origin = origin or _current_thread_label()
    details = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rule = "=" * 78
    print(f"\n{rule}\n[{stamp}] UNHANDLED EXCEPTION in {origin}\n{details}{rule}", file=sys.stderr, flush=True)

    if _reporter is not None:
        _reporter.report.emit(f"{exc_type.__name__} in {origin}", str(exc_value) or "(no message)", details)


def _sys_excepthook(exc_type, exc_value, exc_tb):
    """sys.excepthook replacement: Ctrl+C keeps its default behavior, everything else is reported."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    report_exception(exc_type, exc_value, exc_tb)


def _threading_excepthook(args):
    """threading.excepthook replacement for exceptions escaping plain Python threads."""
    report_exception(args.exc_type, args.exc_value, args.exc_traceback,
                     origin=f"thread '{args.thread.name if args.thread else '?'}'")


def install_exception_hook():
    """
    Installs the handler for sys.excepthook (GUI-thread slots and QThread.run)
    and threading.excepthook (plain Python threads). Call once, after the
    QApplication has been created and from the GUI thread.
    """
    global _reporter
    _reporter = _ExceptionReporter()
    sys.excepthook = _sys_excepthook
    threading.excepthook = _threading_excepthook
