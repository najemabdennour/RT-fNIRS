"""Background QThread workers for resolving and probing LSL streams without blocking the UI."""
from PyQt5.QtCore import QThread, pyqtSignal
from pylsl import StreamInlet, resolve_byprop


class AdvancedStreamResolver(QThread):
    """Resolves a primary data stream by type and, optionally, a marker stream by name."""
    success = pyqtSignal(object, object, str)  # primary_info, marker_info (or None), status_message
    failure = pyqtSignal(str)

    def __init__(self, sig_type, use_markers, marker_name):
        """
        Args:
            sig_type: LSL stream type to resolve (e.g. 'NIRS').
            use_markers: whether to also look for a marker stream.
            marker_name: LSL name of the marker stream.
        """
        super().__init__()
        self.sig_type = sig_type
        self.use_markers = use_markers
        self.marker_name = marker_name

    def run(self):
        """
        Resolve the streams and emit success(primary_info, marker_info, status).

        Takes the first primary stream found within 3 s; emits failure if none.
        A missing marker stream (2 s timeout) is not fatal: marker_info is None
        and the status message carries a warning.
        """
        try:
            # 1. Primary data stream (NIRS, EEG, MEG)
            primary_streams = resolve_byprop("type", self.sig_type, timeout=3.0)
            if not primary_streams:
                self.failure.emit(f"No {self.sig_type} telemetry detected on network.")
                return

            primary_info = primary_streams[0]
            marker_info = None
            status_msg = f"Connected to {self.sig_type}"

            # 2. Optional marker stream
            if self.use_markers and self.marker_name:
                marker_streams = resolve_byprop("name", self.marker_name, timeout=2.0)
                if marker_streams:
                    marker_info = marker_streams[0]
                    status_msg += f" + Trigger Stream ({self.marker_name})"
                else:
                    status_msg += f" (Warning: '{self.marker_name}' Not Found)"

            self.success.emit(primary_info, marker_info, status_msg)
        except Exception as e:
            self.failure.emit(str(e))


class MarkerDiagnosticWorker(QThread):
    """Looks up a marker stream by name and reports its metadata for diagnostics."""
    diagnostic_success = pyqtSignal(dict)
    diagnostic_failure = pyqtSignal(str)

    def __init__(self, marker_name):
        """Store the LSL name of the marker stream to probe."""
        super().__init__()
        self.marker_name = marker_name

    def run(self):
        """
        Resolve the marker stream (2 s timeout) and emit diagnostic_success with
        its name, type, hostname, uid and channel count, or diagnostic_failure.
        """
        try:
            streams = resolve_byprop("name", self.marker_name, timeout=2.0)
            if not streams:
                self.diagnostic_failure.emit(f"Timeout: No stream named '{self.marker_name}' detected.")
                return

            info = streams[0]
            # hostname/uid let the user spot stale "zombie" outlets sharing the same name
            diag_data = {
                "name": info.name(),
                "type": info.type(),
                "hostname": info.hostname(),
                "uid": info.uid(),
                "channels": info.channel_count()
            }
            self.diagnostic_success.emit(diag_data)
        except Exception as e:
            self.diagnostic_failure.emit(str(e))
