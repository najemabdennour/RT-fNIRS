# Real-Time fNIRS & Physiological Signals Toolbox

A PyQt5 desktop application for acquiring, preprocessing, and
running real-time neurofeedback experiments on fNIRS (and other LSL-streamed
physiological) signals — from live hardware or from LUMO-style optode
recordings — with a shared signal-processing engine used identically offline
and in real time.

## What it does

- **Acquire** live NIRS/EEG/MEG streams over [Lab Streaming Layer](https://labstreaminglayer.org/) (LSL), synced to experiment event markers.
- **Design** visual stimulus/neurofeedback paradigms without writing Pygame code, and run them as a generated, self-contained script.
- **Preprocess** recordings — SCI/SQA quality control, MBLL concentration conversion, filtering, detrending, scaling, epoching — through one engine shared by the offline and real-time paths.
- **Train** and benchmark decoders across a dozen scikit-learn classifiers, then deploy the winner.
- **Run real-time neurofeedback**: live prediction, probability feedback pushed back over LSL to the experiment display, and optional co-adaptive retraining during the session.

## Requirements

- Python 3.9–3.14
- [liblsl](https://github.com/sccn/liblsl) installed system-wide (`pylsl` is only a wrapper around it — it is **not** installed by `pip`)
- See `requirements.txt` for Python packages (PyQt5, pandas, numpy, scipy, scikit-learn, joblib, matplotlib, pylsl, pygame; `nilearn` is optional, used only by the detrend step)

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python main.py
```

This opens the main window with five tabs (see [`DOCS/ui_reference.md`](DOCS/ui_reference.md) for full detail on each). If you don't have real hardware or a running experiment handy, you can exercise the whole pipeline with synthetic data:

```bash
# Terminal 1: fake NIRS hardware + a dummy decoder to predict against
python test_neurofeedback.py

# Terminal 2: the app itself
python main.py
```

Then, in the app: **Online Streaming Monitor** to watch the fake signal arrive, **Neurofeedback Engine** to load the dummy model and start decoding, **Experiment Design** to launch a generated paradigm that requests feedback and displays it live.

## Project structure

```
main.py                    Entry point - installs the exception handler, launches the PyQt5 app
config.py                  Preprocessing defaults, paths, non-signal columns, experiment profiles
requirements.txt
experiment_skeleton.py     Hand-editable Pygame+LSL paradigm template
test_neurofeedback.py      Synthetic/real-data LSL harness for testing the neurofeedback pipeline

core/                      Backend logic - no PyQt imports except QThread workers
  math_kernel.py             Filtering, SCI, CV, MBLL inversion - pure numpy
  signal_pipeline.py          SignalPreprocessor: the shared channel-level pipeline engine
  channel_lists.py            Save/load named channel-inclusion/exclusion lists
  decoders.py                  Classifier registry + factory
  offline_processor.py         Session merging + full offline pipeline orchestration
  lsl_client.py                 Async LSL stream resolution workers
  lsl_utils.py                  Channel-name resolution from LSL stream metadata
  simulators.py                  Synthetic LSL data/marker generators (dev mode)
  data_manager.py                Real-time sample/marker buffering + CSV export
  io_utils.py                    CSV read/write that strips leaked index columns

ui/                         PyQt5 tabs and shared widgets
  main_window.py              Top-level window, tab registry, theming
  online_tab.py                 Live acquisition + plotting
  offline_tab.py                 Session merge + offline pipeline UI
  decoders_tab.py                 Decoder benchmarking + training UI
  neurofeedback_tab.py             Real-time decoding + co-adaptation engine
  experiment_tab.py                 Visual paradigm designer + runner
  components.py                       Shared widgets (PipelineStepTable, CollapsibleSection)
  exception_handler.py                 Reports uncaught exceptions (terminal + dialog) without exiting
  themes.py                            QSS/matplotlib theme presets

decoders/                  Trained model .pkl files land here
data/                       Default recordings/behavioral/preprocessed output tree
DOCS/                       Detailed documentation (see below)
```

## Documentation

- [`DOCS/architecture.md`](DOCS/architecture.md) — data flow, the shared pipeline engine, shared conventions, LSL stream/marker conventions, channel naming
- [`DOCS/core_reference.md`](DOCS/core_reference.md) — every module under `core/`
- [`DOCS/ui_reference.md`](DOCS/ui_reference.md) — every tab under `ui/`
- [`DOCS/configuration.md`](DOCS/configuration.md) — `config.py` field-by-field reference, including experiment profiles
- [`DOCS/dev_tools.md`](DOCS/dev_tools.md) — `test_neurofeedback.py` and `experiment_skeleton.py`
- [`DOCS/developer_guide.md`](DOCS/developer_guide.md) — how to work on the package: ground rules, adding signal-pipeline steps (with UI), offline steps, decoders, experiment profiles and tabs

## Known limitations

- `neurofeedback_tab.py` matches the target-class marker by the literal string `"target_class"`, while the Experiment tab's "System LSL Markers" panel lets you rename it — if you rename it there, the Neurofeedback tab won't recognize it. Leave it at the default unless you update both.
- `core/offline_processor.py`'s `discover_folders_by_date` silently falls back to the root recordings/behavioral folder when no date-matched subfolder is found, rather than raising — fine for a single flat folder, but can quietly process the wrong directory if you expected an error.
