"""In-memory recording buffer for a live LSL session and its CSV export."""
import pandas as pd

from core.io_utils import write_csv


class DataSession:
    """Buffers streamed samples with their aligned event markers and exports them to CSV."""
    def __init__(self):
        """Create empty sample, timestamp and per-marker-type buffers."""
        self.samples = []
        self.timestamps = []
        self.trial_onset_marker = []
        self.trial_offset_marker = []
        self.run_onset_marker = []
        self.run_offset_marker = []
        self.cached_visual_markers = [] # Pairs of (timestamp, label) for drawing vertical lines

    def reset(self):
        """Clears buffers to prepare for a fresh recording session."""
        self.samples.clear()
        self.timestamps.clear()
        self.trial_onset_marker.clear()
        self.trial_offset_marker.clear()
        self.run_onset_marker.clear()
        self.run_offset_marker.clear()
        self.cached_visual_markers.clear()

    def process_and_append(self, sample, timestamp, marker_inlet):
        """
        Appends one data sample and polls the marker inlet for at most one marker.

        A pending marker (non-blocking pull) is attached to this sample's row:
        its LSL timestamp goes into the column matching its label
        ('trial_onset', 'trial_offset', 'run_onset', 'run_offset'; other labels
        leave all four None), and (timestamp, label) is cached for plotting.
        Without a marker, all four marker columns get None for this row.
        """
        self.samples.append(sample)
        self.timestamps.append(timestamp)

        if marker_inlet:
            m_sample, m_timestamp = marker_inlet.pull_sample(timeout=0.0)
            if m_sample is not None:
                trigger = str(m_sample[0])

                self.trial_onset_marker.append(m_timestamp if trigger == "trial_onset" else None)
                self.trial_offset_marker.append(m_timestamp if trigger == "trial_offset" else None)
                self.run_onset_marker.append(m_timestamp if trigger == "run_onset" else None)
                self.run_offset_marker.append(m_timestamp if trigger == "run_offset" else None)

                self.cached_visual_markers.append((m_timestamp, trigger))
            else:
                self._append_null_markers()
        else:
            self._append_null_markers()

    def _append_null_markers(self):
        """Append None to every marker column for a sample with no marker."""
        self.trial_onset_marker.append(None)
        self.trial_offset_marker.append(None)
        self.run_onset_marker.append(None)
        self.run_offset_marker.append(None)

    def export_to_csv(self, file_path, channel_names):
        """
        Writes the buffered samples to CSV: one column per channel plus
        'timestamps' and the four marker columns.

        Returns False (writing nothing) if the buffer is empty, else True.
        """
        if not self.samples:
            return False

        df = pd.DataFrame(self.samples, columns=channel_names)
        df["timestamps"] = self.timestamps
        df["trial_onset"] = self.trial_onset_marker
        df["trial_offset"] = self.trial_offset_marker
        df["run_onset"] = self.run_onset_marker
        df["run_offset"] = self.run_offset_marker

        write_csv(df, file_path)
        return True