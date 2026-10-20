"""Background QThread workers for resolving, probing and recording LSL streams without blocking the UI."""
import threading
import time

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


class StreamAcquisitionWorker(QThread):
    """
    Records a live LSL stream into a DataSession on a background thread.

    Pulls data in chunks and appends each sample (polling the marker inlet
    once per sample, see DataSession.process_and_append), so recording keeps
    up with the stream no matter how busy the GUI thread is. The worker owns
    both inlets while running: stop() it before closing them.
    """
    error = pyqtSignal(str)

    def __init__(self, inlet, marker_inlet, session, idle_sleep=0.01):
        """
        Args:
            inlet: StreamInlet of the data stream.
            marker_inlet: StreamInlet of the marker stream, or None.
            session: DataSession receiving the samples.
            idle_sleep: seconds to sleep when no data is waiting.
        """
        super().__init__()
        self.inlet = inlet
        self.marker_inlet = marker_inlet
        self.session = session
        self.idle_sleep = idle_sleep
        self.is_running = True

    def run(self):
        """Pulls and records chunks until stopped; emits `error` and exits if the inlet fails."""
        try:
            while self.is_running:
                samples, timestamps = self.inlet.pull_chunk(timeout=0.0)
                if not timestamps:
                    time.sleep(self.idle_sleep)
                    continue
                for sample, timestamp in zip(samples, timestamps):
                    self.session.process_and_append(sample, timestamp, self.marker_inlet)
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        """Asks the loop to finish and waits for the thread to exit."""
        self.is_running = False
        self.wait()


def release_inlets_async(inlets):
    """
    Closes and releases LSL inlets on a background thread.

    liblsl's inlet teardown blocks for about 0.5 s per inlet while its
    internal threads shut down; doing it here keeps the GUI responsive.

    Takes ownership: `inlets` is a list the caller fills and keeps no other
    references to; it is emptied here, so the background thread holds the
    last reference and the teardown always runs there (if the caller still
    held one, the teardown could end up back on the caller's thread). None
    entries are ignored. Returns the thread, or None if there was nothing to
    release.
    """
    pending = [inlet for inlet in inlets if inlet is not None]
    inlets.clear()
    if not pending:
        return None

    def _release(pending):
        """Closes each inlet, then drops the last reference so liblsl destroys it here."""
        while pending:
            inlet = pending.pop()
            try:
                inlet.close_stream()
            except Exception:
                pass
            del inlet

    thread = threading.Thread(target=_release, args=(pending,), name="lsl-inlet-release", daemon=True)
    thread.start()
    return thread
