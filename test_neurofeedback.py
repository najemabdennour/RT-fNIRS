"""
Synthetic LSL hardware + decoder test harness for the real-time
neurofeedback pipeline (see ui/neurofeedback_tab.py).

Streams an NIRS LSL outlet named NIRS_STREAM_NAME, prints probabilities read
from the "Neurofeedback_Out" stream, and, if decoders/<model-name>.pkl does not
exist yet, trains a dummy LogisticRegression and saves it there (its training
set is also written to data/synthetic_training_data.csv). Two data sources:

  1. Pure synthetic (default): N_CHANNELS channels of sine + Gaussian noise,
     and a model trained on random data. Only useful for exercising LSL plumbing.

  2. Real data (--real-data path/to/file.csv): a recorded or preprocessed
     session CSV is looped over the stream with its channel names, so the
     pipeline (SCI/SQA pairing, MBLL, ...) sees realistic fNIRS. Training
     windows and each replay loop are augmented (per-channel scaling, noise
     scaled to channel amplitude, small circular time shift).

Usage:
    python test_neurofeedback.py
    python test_neurofeedback.py --real-data data/preprocessed/grouped_sessions/Session_2026-01-01.csv --trials 30
    python test_neurofeedback.py --real-data some_session.csv --no-augment

With --no-augment the replay is verbatim and training windows get no time
shift, but training windows still get scaling and noise unless --noise-std 0.
"""
import argparse
import time, os, threading
import numpy as np, joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, local_clock

from core.io_utils import read_csv, write_csv

NIRS_STREAM_NAME = "LUMO_Synthetic"
N_CHANNELS = 100          # only used without --real-data
SAMPLING_RATE = 6.6

# Columns stripped from a real CSV before treating the rest as signal channels
# (the same list the offline pipeline and decoder tab use).
from config import NON_SIGNAL_COLUMNS

decoders_dir = os.path.join(os.path.abspath(os.getcwd()), "decoders")


def load_real_signal_source(csv_path):
    """Loads a recorded/preprocessed CSV and returns (signal_df, channel_names).

    Drops NON_SIGNAL_COLUMNS, non-numeric and all-NaN columns, and
    forward/back-fills remaining gaps.

    Raises:
        ValueError: if no numeric signal columns remain.
    """
    df = read_csv(csv_path)
    channel_cols = [c for c in df.columns if c not in NON_SIGNAL_COLUMNS]
    signal_df = df[channel_cols].select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    signal_df = signal_df.ffill().bfill()
    if signal_df.empty:
        raise ValueError(f"No usable numeric signal columns found in '{csv_path}'.")
    return signal_df, list(signal_df.columns)


def augment_signal_window(window, noise_std=0.05, scale_range=(0.9, 1.1), time_jitter=True):
    """Returns an augmented copy of a (samples x channels) signal window.

    Applies, in order: random per-channel amplitude scaling within scale_range,
    additive Gaussian noise with std noise_std times each channel's (scaled)
    std, and optionally a circular time shift of up to 10% of the window.
    The output has the same shape as the input.
    """
    data = np.asarray(window, dtype=float).copy()
    n_samples, n_channels = data.shape

    scales = np.random.uniform(scale_range[0], scale_range[1], size=n_channels)
    data = data * scales

    channel_std = data.std(axis=0)
    channel_std[channel_std == 0] = 1.0
    data = data + np.random.normal(0, noise_std, size=data.shape) * channel_std

    if time_jitter and n_samples > 1:
        max_shift = max(1, n_samples // 10)
        shift = np.random.randint(-max_shift, max_shift + 1)
        data = np.roll(data, shift, axis=0)

    return data


def build_training_set_from_real_data(signal_df, n_trials=20, window_size=None, noise_std=0.05,
                                       scale_range=(0.9, 1.1), time_jitter=True):
    """Builds a dummy (X, y) training set from random windows of a real recording.

    Each of n_trials windows (default length: a quarter of the recording,
    clamped to 10-60 samples) is augmented, unless both noise_std and
    time_jitter are falsy, and reduced to its per-channel mean, mirroring the
    live 'averaging (feature_vector)' step. Labels alternate 0/1 so the two
    classes are balanced.

    Returns:
        (X, y): DataFrame with the signal column names, and a label array.
    """
    values = signal_df.values
    n_samples_total, n_channels = values.shape
    window_size = window_size or max(10, min(60, n_samples_total // 4))
    window_size = max(1, min(window_size, n_samples_total))

    rows, targets = [], []
    for i in range(n_trials):
        start = np.random.randint(0, max(1, n_samples_total - window_size + 1))
        window = values[start:start + window_size]
        augmented = augment_signal_window(window, noise_std=noise_std, scale_range=scale_range,
                                           time_jitter=time_jitter) if (noise_std or time_jitter) else window
        rows.append(augmented.mean(axis=0))
        targets.append(i % 2)

    X = pd.DataFrame(rows, columns=signal_df.columns)
    y = np.array(targets)
    return X, y


def generate_dummy_model(filename="dummy_model", signal_df=None, n_trials=20,
                          noise_std=0.05, scale_range=(0.9, 1.1), time_jitter=True):
    """Trains a dummy LogisticRegression and saves it to decoders/<filename>.pkl.

    Uses augmented windows of signal_df when given, otherwise random features on
    N_CHANNELS "Ch-i" columns. The training set (with a 'target' column) is also
    written to data/synthetic_training_data.csv.
    """
    print(f"📦 Generating synthetic deployment model: '{filename}'...")

    if signal_df is not None:
        print(f"📈 Building {n_trials} augmented training trial(s) from the real data source...")
        X, y = build_training_set_from_real_data(
            signal_df, n_trials=n_trials, noise_std=noise_std, scale_range=scale_range, time_jitter=time_jitter
        )
    else:
        column_names = [f"Ch-{i}" for i in range(N_CHANNELS)]
        X = pd.DataFrame(np.random.randn(n_trials, N_CHANNELS), columns=column_names)
        y = np.random.randint(0, 2, size=n_trials)

    print("x shape:", X.shape)
    training_data = pd.concat([X, pd.Series(y, name='target')], axis=1)
    os.makedirs("data", exist_ok=True)
    write_csv(training_data, "data/synthetic_training_data.csv")
    print("Synthetic training data saved to:\n", "data/synthetic_training_data.csv")

    pipeline = LogisticRegression()
    pipeline.fit(X, y)
    os.makedirs(decoders_dir, exist_ok=True)
    export_target_path = os.path.join(decoders_dir, f"{filename}.pkl")
    joblib.dump(pipeline, export_target_path)
    print("✅ Dummy model ready.")


def nirs_hardware_simulator(stop_event, real_signal_values=None, channel_names=None,
                             augment=True, noise_std=0.03, scale_range=(0.95, 1.05)):
    """Streams simulated NIRS samples over LSL at SAMPLING_RATE until stop_event is set.

    Without real_signal_values, sends sine + Gaussian noise on N_CHANNELS
    unlabeled channels. With real_signal_values, loops that recording, re-augmenting
    the whole recording on each pass when augment is True. channel_names, if
    given, are added as LSL channel labels so SCI/SQA pairing sees real names.
    """
    n_channels = real_signal_values.shape[1] if real_signal_values is not None else N_CHANNELS
    info = StreamInfo(NIRS_STREAM_NAME, 'NIRS', n_channels, SAMPLING_RATE, 'float32', 'synth_hw')

    if channel_names:
        chns = info.desc().append_child("channels")
        for name in channel_names:
            chns.append_child("channel").append_child_value("label", name)

    outlet = StreamOutlet(info)
    t0 = local_clock()
    sent = 0

    replay_buffer = [None]  # mutable cell so the closure below can rebind it
    replay_idx = [0]

    def next_replay_chunk(n):
        """Returns the next n replay rows, starting a new (re-augmented) pass on wrap-around."""
        out_rows = []
        while len(out_rows) < n:
            if replay_buffer[0] is None or replay_idx[0] >= len(replay_buffer[0]):
                replay_buffer[0] = (
                    augment_signal_window(real_signal_values, noise_std=noise_std, scale_range=scale_range,
                                           time_jitter=True)
                    if augment else real_signal_values
                )
                replay_idx[0] = 0
            take = min(n - len(out_rows), len(replay_buffer[0]) - replay_idx[0])
            out_rows.extend(replay_buffer[0][replay_idx[0]:replay_idx[0] + take].tolist())
            replay_idx[0] += take
        return out_rows

    while not stop_event.is_set():
        req = int(SAMPLING_RATE * (local_clock() - t0)) - sent
        if req > 0:
            if real_signal_values is not None:
                for row in next_replay_chunk(req):
                    outlet.push_sample([float(v) for v in row])
            else:
                for _ in range(req):
                    outlet.push_sample([float(np.sin(local_clock()) + np.random.normal(0, 0.1)) for _ in range(n_channels)])
            sent += req
        time.sleep(0.01)


def feedback_monitor(stop_event):
    """Waits for the "Neurofeedback_Out" LSL stream and prints each 'probability' sample."""
    print("🎧 Feedback Monitor: Searching for Probability outlet...")
    inlet = None
    while not stop_event.is_set() and inlet is None:
        try:
            streams = resolve_byprop("name", "Neurofeedback_Out", timeout=1.0)
            if streams: inlet = StreamInlet(streams[0])
        except: pass
            
    while not stop_event.is_set() and inlet is not None:
        sample, _ = inlet.pull_sample(timeout=0.5)
        if sample and sample[0] == "probability":
            print(f"🎯 Output Probability: {float(sample[1])*100:.1f}% Match")


def parse_args():
    """Parses the command-line options of the test harness."""
    parser = argparse.ArgumentParser(
        description="Synthetic LSL hardware + decoder validation harness for the neurofeedback pipeline."
    )
    parser.add_argument("--real-data", type=str, default=None,
                         help="Path to a real recorded/preprocessed CSV to use as the signal source "
                              "instead of pure random data.")
    parser.add_argument("--trials", type=int, default=20,
                         help="Number of augmented training trials to draw when using a real data source (default: 20).")
    parser.add_argument("--no-augment", action="store_true",
                         help="Disable augmentation - train/replay the real data source verbatim (looped as-is).")
    parser.add_argument("--noise-std", type=float, default=0.05,
                         help="Relative Gaussian noise std added during augmentation (default: 0.05).")
    parser.add_argument("--scale-min", type=float, default=0.9,
                         help="Lower bound of the random per-channel amplitude scaling range (default: 0.9).")
    parser.add_argument("--scale-max", type=float, default=1.1,
                         help="Upper bound of the random per-channel amplitude scaling range (default: 1.1).")
    parser.add_argument("--model-name", type=str, default="dummy_model",
                         help="Filename (without .pkl) for the generated dummy decoder (default: dummy_model).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    augment = not args.no_augment
    scale_range = (args.scale_min, args.scale_max)

    real_signal_df = None
    channel_names = None
    if args.real_data:
        if not os.path.exists(args.real_data):
            print(f"⚠️  Real data path not found: '{args.real_data}'. Falling back to pure synthetic data.")
        else:
            real_signal_df, channel_names = load_real_signal_source(args.real_data)
            print(f"📊 Loaded real signal source: {real_signal_df.shape[0]} samples x "
                  f"{real_signal_df.shape[1]} channels from '{args.real_data}'")

    export_target_path = os.path.join(decoders_dir, f"{args.model_name}.pkl")
    if not os.path.exists(export_target_path):
        generate_dummy_model(
            filename=args.model_name,
            signal_df=real_signal_df,
            n_trials=args.trials,
            noise_std=args.noise_std,
            scale_range=scale_range,
            time_jitter=augment,
        )

    stop_event = threading.Event()
    threading.Thread(
        target=nirs_hardware_simulator,
        args=(stop_event,),
        kwargs={
            "real_signal_values": real_signal_df.values if real_signal_df is not None else None,
            "channel_names": channel_names,
            "augment": augment,
            "noise_std": args.noise_std,
            "scale_range": scale_range,
        }
    ).start()
    threading.Thread(target=feedback_monitor, args=(stop_event,)).start()

    print("⚡ Hardware Simulator Active. Open GUI and start processing.\n")
    try:
        while True: time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()
