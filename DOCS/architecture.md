# Architecture

## Overview

The toolbox splits cleanly into two layers:

- **`core/`** — backend logic. No PyQt imports except three `QThread` worker
  classes (`AdvancedStreamResolver`, `MarkerDiagnosticWorker` in
  `lsl_client.py`, and the synthetic-stream threads in `simulators.py`) that
  need to run off the GUI thread. Everything else is plain
  pandas/numpy/scikit-learn, callable and testable independently of the UI.
- **`ui/`** — PyQt5 tabs. Each tab is a thin shell that collects parameters
  from widgets, calls into `core/`, and renders the result. The one
  exception is `neurofeedback_tab.py`'s `NeurofeedbackWorker`, which runs
  the real-time decode loop on a background `QThread` but still delegates
  all actual signal processing to `core.signal_pipeline.SignalPreprocessor`.

The design goal that shows up everywhere: **the same channel-level signal
processing code runs in both the offline batch path and the real-time
path.** `core/signal_pipeline.py`'s `SignalPreprocessor` has no notion of
trials, runs, or behavioral data — it just takes a DataFrame of channel
columns and a params dict. `core/offline_processor.py` wraps it with
session-structural steps that only make sense for a full recorded session
(epoching, trial/block averaging); `ui/neurofeedback_tab.py` calls the same
engine directly on a live buffer plus one extra step
(`averaging (feature_vector)`) that collapses a trial buffer to the single
feature row a decoder expects.

## Data flow

```
                    ┌─────────────────────────────────────────────┐
                    │              Experiment Design tab            │
                    │  (generates a Pygame script, launches it as    │
                    │   a subprocess, pushes lifecycle + target-class │
                    │   markers over LSL)                              │
                    └───────────────────────┬───────────────────────┘
                                             │  "OpenSesame_Markers" stream
                                             ▼
  Real hardware / core.simulators   ┌──────────────────┐
  ───────────────────────────────▶ │   LSL network     │
        NIRS/EEG/MEG data stream    └─────────┬──────────┘
                                             │
                  ┌──────────────────────────┼──────────────────────────┐
                  ▼                          ▼                          ▼
          Online Streaming tab      Neurofeedback tab            (any LSL client)
          (core.data_manager.        (NeurofeedbackWorker:
           DataSession buffers        SignalPreprocessor on a
           samples+markers,           live buffer → decoder →
           exports CSV)               probability)
                  │                          │
                  ▼                          ▼
             Recorded CSV            "Neurofeedback_Out" stream
                  │                          │
                  ▼                          ▼
        Offline Analytical tab      Experiment Design tab's running
        (OfflineSessionCombiner      script (renders feedback: Text/
         merges runs + behavioral,   Progress Bar/Disk) and/or
         OfflineDataPreprocessor     core.channel_lists-based
         runs the shared pipeline,   co-adaptation loop
         epochs, averages)
                  │
                  ▼
        Decoder Training tab
        (benchmarks core.decoders.DECODER_REGISTRY,
         saves the winner to decoders/*.pkl)
                  │
                  ▼
        decoders/*.pkl ──────────▶ loaded by Neurofeedback tab
```

## Shared conventions

- **Non-signal columns.** `config.NON_SIGNAL_COLUMNS` is the single list of
  structural/marker/label columns that no step treats as a channel. The
  offline pipeline extends it with the active experiment profile's
  behavioral columns via `config.get_non_signal_columns(experiment)`.
- **Experiment profiles.** `config.EXPERIMENT_PROFILES` describes how each
  paradigm's behavioral log is aligned onto the signal (label column,
  carried columns, trial filter). The selected profile is passed explicitly
  through `params['active_experiment']` — there is no global "active
  experiment" state.
- **CSV I/O.** All reads/writes go through `core.io_utils`, which strips
  leaked pandas index columns (`Unnamed: 0`) so they never become features.
- **Warnings.** Non-fatal problems in the signal engine (e.g. a filter
  skipped on a too-short buffer) go through `SignalPreprocessor.warn()` and
  appear at WARNING level in the Offline and Neurofeedback logs.

## LSL streams and markers

| Stream | Type | Producer | Consumer |
|---|---|---|---|
| NIRS / EEG / MEG data | configurable `type` | real hardware, or `core.simulators.SyntheticNIRSThread` / `test_neurofeedback.py` | Online tab, Neurofeedback tab |
| `OpenSesame_Markers` (name configurable per-marker in the Experiment tab) | `Markers` (2-channel string) | Experiment Design tab's generated runner, or `core.simulators.SyntheticMarkerThread` | Online tab (via `DataSession`), Neurofeedback tab |
| `Neurofeedback_Out` | string, samples like `["probability", "0.73"]` | `NeurofeedbackWorker` | Experiment Design tab's running script (Feedback Window step), `test_neurofeedback.py`'s `feedback_monitor` |

Default marker labels (all customizable in the Experiment tab's "System LSL
Markers" panel, *except* `target_class` — see the caveat below):
`baseline_start`, `baseline_end`, `run_onset`, `run_offset`, `trial_index`,
`target_class`. Per-step markers (e.g. `trial_onset`, `trial_offset`,
`request_feedback`, `fixation_onset`) are set per-row in the step table and
are always free text.

> **Caveat:** `ui/neurofeedback_tab.py` matches the target-class marker
> using the hardcoded literal `"target_class"`, not the configurable name
> from the Experiment tab's marker panel. If you rename that marker there,
> the Neurofeedback tab will stop recognizing it.

## Channel naming convention

Every part of the pipeline that pairs up dual-wavelength channels (SCI
rejection, MBLL conversion) relies on one naming convention:

```
S(<source>)_D(<detector>)_<wavelength>
e.g.  S(1)_D(2)_735   S(1)_D(2)_850
```

The last underscore separates the optode-pair base name from the trailing
wavelength (or, after MBLL conversion, from `HbO`/`HbR`). Code that needs
the base name does `col.rsplit('_', 1)[0]`.

Real LUMO recordings use a double-underscore variant,
`S(N1/A)__D(N1/1)__735`. This still pairs correctly: the base name keeps a
trailing underscore (`S(N1/A)__D(N1/1)_`), and `f"{base}_{wavelength}"`
rebuilds the original name. Columns with no underscore at all (e.g. the
generic `C1`, `C2`… of older recordings) are never paired, so SCI and MBLL
leave them untouched. This convention is produced by:

- `core/lsl_utils.py`'s `resolve_channel_names()`, reading `source`/
  `detector`/`wavelen` child values from LSL stream metadata (falling back
  to a `label` child value, then to `Ch-<i>`)
- `core/simulators.py`'s `SyntheticNIRSThread` and `test_neurofeedback.py`'s
  `nirs_hardware_simulator`, which both write this same label format into
  the LSL stream's `channels`/`channel`/`label` metadata tree

If you bring in real hardware with a different metadata schema, either
adapt `resolve_channel_names()` or make sure its `label` fallback produces
`S(x)_D(y)_wavelength`-shaped names — the shared pipeline's SCI/MBLL steps
depend on it.

## Real-time preprocessing caveat

`SignalPreprocessor`'s SCI and SQA steps can drop different channels on
different live trial buffers (a channel that looks noisy in one short
window might look fine in the next), which changes the feature vector's
shape from trial to trial — a problem for a decoder trained on a fixed
feature set. `core/channel_lists.py` exists specifically to address this:
run SCI/SQA once offline, export the rejected channel names, then load that
fixed list into the Neurofeedback tab's "Manual Channel Filter" step
instead of (or alongside) live SCI/SQA, so the same channels are excluded
consistently on every trial.
