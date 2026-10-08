# core/simulators.py
"""Synthetic LSL outlets (fNIRS data and experiment markers) for testing without hardware."""
import time
import math
from random import random as rand
from PyQt5.QtCore import QThread
from pylsl import StreamInfo, StreamOutlet, local_clock


class SyntheticNIRSThread(QThread):
    """Publishes a synthetic 'NIRS' LSL stream of sine-plus-noise channels at a fixed rate."""
    def __init__(self, name='LUMO_Synthetic', n_channels=100, fs=6.6667):
        """
        Args:
            name: LSL stream name.
            n_channels: number of channels (kept low by default for plotting speed).
            fs: nominal sampling rate (Hz).
        """
        super().__init__()
        self.stream_name = name
        self.n_channels = n_channels
        self.fs = fs
        self.is_running = True

    def run(self):
        """
        Create the outlet with 'S(x)_D(y)_<wavelength>' channel labels, then push
        samples until stop() is called, catching up on however many samples
        are due since start so the average rate matches fs.
        """
        info = StreamInfo(self.stream_name, 'NIRS', self.n_channels, self.fs, 'float32', 'synth_nirs_01')
        chns = info.desc().append_child("channels")

        # Labels must use a single '_' between optode pair and wavelength:
        # core.signal_pipeline pairs wavelengths by stripping the trailing
        # '_<wavelength>', and core.lsl_utils.resolve_channel_names expects it.
        # Even channels are 735 nm, odd channels 850 nm.
        for i in range(self.n_channels):
            ch = chns.append_child("channel")
            source = (i // 10) + 1
            detector = (i % 10) + 1
            wl = 735 if i % 2 == 0 else 850
            ch.append_child_value("label", f"S({source})_D({detector})_{wl}")

        outlet = StreamOutlet(info)
        start_time = local_clock()
        sent_samples = 0

        while self.is_running:
            elapsed_time = local_clock() - start_time
            required_samples = int(self.fs * elapsed_time) - sent_samples

            for _ in range(required_samples):
                t = local_clock()
                sample = [(math.sin(t) + (rand() * 2)) for _ in range(self.n_channels)]
                outlet.push_sample(sample)

            sent_samples += required_samples
            time.sleep(0.01) # Avoid busy-waiting at 100% CPU

    def stop(self):
        """Signal the push loop to exit and block until the thread finishes."""
        self.is_running = False
        self.wait()


class SyntheticMarkerThread(QThread):
    """
    Publishes a synthetic 2-channel string 'Markers' LSL stream: one
    'run_onset', then repeating trials of 'trial_onset', 'trial_index',
    'target' and 'trial_offset' markers (about 8 s per cycle).
    """
    def __init__(self, name='OpenSesame_Markers'):
        """Store the LSL stream name for the marker outlet."""
        super().__init__()
        self.stream_name = name
        self.is_running = True

    def run(self):
        """Create the marker outlet and emit the run/trial marker sequence until stopped."""
        # 2 channels: [label, value]
        info = StreamInfo(self.stream_name, 'Markers', 2, 0, 'string', 'synth_markers_01')
        outlet = StreamOutlet(info)
        time.sleep(1.5) # Give consumers time to discover the outlet before the first marker

        if not self.is_running: return
        outlet.push_sample(['run_onset', '0'])

        trial_count = 0
        while self.is_running:
            time.sleep(3) # Inter-trial interval
            if not self.is_running: break

            outlet.push_sample(['trial_onset', '0'])
            trial_count += 1
            time.sleep(1)
            outlet.push_sample(['trial_index', str(trial_count)])
            time.sleep(1)
            outlet.push_sample(['target', '1'])

            time.sleep(3)
            if not self.is_running: break

            outlet.push_sample(['trial_offset', '0'])

    def stop(self):
        """Signal the marker loop to exit and block until the thread finishes."""
        self.is_running = False
        self.wait()
