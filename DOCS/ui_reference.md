# `ui/` Reference

## `main_window.py` — `LSLVisualizerApp`

The top-level `QMainWindow`. Owns the single shared
`core.data_manager.DataSession` (`self.global_session_cache`), the theme
switcher (backed by `ui.themes.THEMES`), a fullscreen toggle, and registers
the five tabs in this order: Online Streaming Monitor, Real-Time
Neurofeedback Engine, Experiment Design & Presentation, Offline Analytical
Processing, Decoder Training / Evaluation.

## `exception_handler.py`

Global handler for uncaught exceptions, installed by `main.py`
(`install_exception_hook()`, right after the `QApplication` is created).
Uncaught exceptions **never close the app**. Instead, each one is:

1. printed to the terminal (stderr) as a timestamped block with its origin —
   `GUI thread`, `QThread <WorkerClass>`, or `thread '<name>'` — and the full
   traceback;
2. shown in an **Unhandled Exception** dialog: exception type and origin,
   the message, and the full traceback under *Show Details...*.

It covers Qt slots on the GUI thread and `QThread.run()` (via
`sys.excepthook`) and plain Python threads (via `threading.excepthook`).
Dialogs are always opened from the GUI thread's event loop via a queued
signal, never from inside the hook itself. Only one dialog is open at a
time: exceptions raised while it is open (e.g. a timer slot failing
repeatedly) are printed to the terminal and summarized as
"N further exception(s)" when it closes. Ctrl+C (`KeyboardInterrupt`) keeps
Python's default behavior. `report_exception(exc_type, exc_value, exc_tb)`
can also be called directly to report a caught exception the same way.

The handler is a safety net, not error handling: expected failures (bad
input, missing files) should still be caught where they happen and shown
with a specific message, as the tabs already do.

## `themes.py`

Four QSS + matplotlib style presets in a flat `THEMES` dict: **Dark
Studio** (default), **Classic Light**, **Cyberpunk Neon**, **Matrix
Hacker**. Each entry supplies a `qss` stylesheet string, an `mpl` style
name, and `bg`/`fg` hex colors. Pure data — consumed by
`main_window.apply_theme_config`.

## `components.py`

Shared widgets used by more than one tab:

- **`CollapsibleSection(title)`** — an accordion widget, collapsed by
  default (`set_expanded(bool)` to force a state). Used to keep
  infrequently-touched panels (e.g. the real-time pipeline, advanced marker
  config) out of the way until needed.
- **`StepDescriptionWidget(title, subtitle)`** — compact title+subtitle
  label pair for a pipeline step row.
- **`PipelineStepTable(steps, show_checkpoint=True, row_height=56)`** — a
  reorderable table of pipeline steps, each with an Execute toggle and an
  optional Checkpoint (save) toggle, plus Move Up/Down buttons. Shared by
  the Offline tab and the Neurofeedback tab, so both drive the exact same
  step-table UI and `compiled_steps()` → `{'id', 'run'[, 'save']}` contract
  into `SignalPreprocessor.run_steps`. Rows are
  `(step_id, title, subtitle, default_run[, default_save])` tuples — see
  [`developer_guide.md`](developer_guide.md#44-the-ui-offline-tab-uioffline_tabpy).

---

## `online_tab.py` — `OnlineStreamingTab`

Live acquisition dashboard. Lets you pick a target signal type
(NIRS/EEG/MEG), optionally enable a marker stream, and resolve both via
`core.lsl_client.AdvancedStreamResolver`. A "Use Synthetic Data" mode spins
up `core.simulators.SyntheticNIRSThread`/`SyntheticMarkerThread` instead of
requiring real hardware. Once connected, a `QTimer` polls the inlet every
30ms, appends into `parent_win.global_session_cache` (the single shared
`DataSession`), and plots up to 15 selected channels with vertical marker
lines. Exports the accumulated session to CSV on demand.

---

## `offline_tab.py` — `OfflineAnalysisTab`

Three concerns in one tab:

1. **Session merge** — a thin UI over `core.offline_processor.OfflineSessionCombiner`: pick or auto-discover today's recordings/behavioral folders, combine per-run CSVs into one session file.
2. **Experiment profile selector** — the *Active Experiment* dropdown lists `config.EXPERIMENT_PROFILES` (default `config.DEFAULT_EXPERIMENT`). The selection is read from the dropdown and passed explicitly to the session merge and the pipeline (`params['active_experiment']`); it decides whether a behavioral log is mandatory and how `epoch_prep` aligns it. *Register Profile* adds a session-only profile from a name + comma-separated column list via `config.register_experiment_profile()` (label column = `condition` if listed).
3. **Pipeline** — a `PipelineStepTable` (channel_filter, sci, sqa, mbll, epoch_prep, filter, detrend, scaling, averaging, block_avg) driving `core.offline_processor.OfflineDataPreprocessor.run_preprocessing_pipeline`, with per-step checkpoint export. Engine warnings (e.g. a skipped filter) appear in the log terminal at WARNING level.

Also hosts the **Manual Channel Filter List** panel: type/paste channel
names or load a saved list, and — after a run where SCI/SQA actually
rejected something — **Save Rejected Channels...** exports them via
`core.channel_lists.save_channel_list` for reuse in the Neurofeedback tab.

---

## `decoders_tab.py` — `DecoderTrainingTab`

Loads one or more preprocessed CSVs (each needs `target` and `run`
columns; every column in `config.NON_SIGNAL_COLUMNS` is dropped from the
feature set, and leaked index columns are stripped on load), then benchmarks every entry in `core.decoders.DECODER_REGISTRY`
via cross-validation (`LeaveOneGroupOut` grouped by `run`, or
`StratifiedShuffleSplit` as a fallback), ranks by macro F1, and trains +
serializes the selected winner to `decoders/<name>.pkl` via
`decoder_selection(preprocessing=True, ...)`.

---

## `neurofeedback_tab.py` — `NeurofeedbackTab` / `NeurofeedbackWorker`

The real-time engine. `NeurofeedbackTab` is the UI shell; the actual decode
loop runs on a background `QThread`, `NeurofeedbackWorker`.

**UI panels** (top to bottom):

- **Signal Source** — target stream type selector (NIRS/EEG/MEG), mirroring `online_tab.py`.
- **Neurofeedback Decoder Configuration** — model path, co-adaptation toggle + decoder choice + base training data path, and a **"Use Co-adapted Decoder for Live Predictions"** checkbox (see Co-adaptation below).
- **Channel Filter List** — load a channel-exclusion list (e.g. one exported from the Offline tab) for the real-time `channel_filter` step.
- **Real-Time Preprocessing Pipeline** (`CollapsibleSection`, collapsed by default) — a `PipelineStepTable` (channel_filter, sci, sqa, mbll, filter, detrend, scaling, `averaging (feature_vector)`), every step unchecked by default, plus the numeric threshold/filter/scaler parameter fields.

**Worker lifecycle (`NeurofeedbackWorker.run()`):**

1. Loads the model and warns if it was trained with leaked CSV index
   columns (`Unnamed: N`) as features, since live predictions would then
   fail on a feature-name mismatch. Resolves the data stream and the marker stream, each via
   `_resolve_stream_with_retry()` — polls every 2s and checks the
   cooperative `is_running` flag between attempts, so starting the tab
   before the NIRS hardware or the Experiment Design script is running
   just waits (with live status/log feedback) instead of failing after a
   fixed timeout. Resolves channel names via `resolve_channel_names(data_inlet.info())`
   — the connected inlet's full metadata, not the lightweight discovery
   record.
2. Waits through the baseline period, buffering samples.
3. Per trial: buffers the trial window, and on `request_feedback`, calls
   `_process_and_predict()`, which concatenates baseline+trial buffers,
   runs them through the same `SignalPreprocessor.run_steps()` as the
   offline path, slices back to the trial portion, predicts, and pushes
   `["probability", <value>]` out over the `Neurofeedback_Out` LSL stream.

**Co-adaptation:** the co-adaptive model starts as the loaded base model.
Each trial whose target-class probability exceeds
`config.COADAPTATION_ACCEPTANCE_CRITERIA` is appended (labelled with the
current target class) to the base training CSV's rows; once at least two
classes are present, a fresh decoder from `core.decoders.decoder_selection()`
is refit on the accumulated set and saved as `<model>_coadapted.pkl`. This *training* is controlled solely by the
"Enable Co-adaptation" checkbox. Whether that retrained model is actually
**used** for this session's live predictions is a separate toggle
("Use Co-adapted Decoder for Live Predictions") — unchecking it keeps
predictions (and therefore trial-acceptance decisions) on the stable base
model while co-adaptation still trains and saves in the background.

**Stop button (`stop()`):** sets `is_running = False` and calls
`self.wait()`, blocking until the thread has actually exited before the UI
re-enables Start — this matters because the retry-resolve loop, the
baseline wait, and the trial loop all check `is_running` cooperatively.

**Automatic reset:** the worker's `QThread.finished` signal is connected to
`_on_worker_finished`, so Start is re-enabled (and Stop disabled) whenever
the worker ends on its own — a `run_offset` marker, a failed model load, or
an exception — not only when Stop is pressed. Signals from an older worker
are ignored, so a finished previous run can't reset a newly started one.

---

## `experiment_tab.py` — `ExperimentDesignTab`

A visual paradigm builder that emits a self-contained Pygame+LSL script
(`RUNNER_CODE`, written to a temp file and launched via `subprocess`) —
functionally the data-driven successor to `experiment_skeleton.py`.

**Step table** columns: Step Type, Content/Stimulus, Color, Size, Duration,
LSL Marker. The Content/Stimulus column's widget changes based on Step
Type:

| Step Type | Content column shows |
|---|---|
| Fixation Cross | (disabled — only Color/Size apply) |
| Text Prompt | free text field |
| Shape | shape picker (Circle/Square/Triangle/Rectangle/Diamond) |
| Image | file path + Browse button |
| Feedback Window | `FeedbackConfigCell`: style dropdown + two checkboxes (see below) |

**Feedback Window display** (per-step, via `FeedbackConfigCell`):
- **Style**: `Text`, `Progress Bar (Horizontal)`, `Progress Bar (Vertical)`, or `Disk`.
- **Show 'Feedback' Label** checkbox.
- **Show Percentage Text** checkbox (Progress Bar/Disk only — Text style always shows the percentage, since that's its entire content).

The Disk style draws three concentric circles: a filled red performance
circle sized by
`circle_radius(prob, chance_level, chance_level_radius) = prob * chance_level_radius / chance_level`
(exactly matching the yellow ring at chance level), a yellow unfilled
chance-level reference ring, and a white unfilled maximum-performance
reference ring. `CHANCE_LEVEL` (0.5), `CHANCE_LEVEL_RADIUS` (50px), and
`MAX_RADIUS` (100px) are fixed constants in the runner, not user-configurable.
The vertical progress bar fills bottom-to-top; both bar orientations draw a
chance-level tick mark. The "Feedback" label's vertical clearance is
computed per-style from that style's own half-height, so it never
overlaps whichever display is drawn below it.

**Trial Target Class Sequence** — a comma-separated sequence (e.g. `0,1,0,1`)
that cycles to fill however many trials are configured, with a **Shuffle**
button that expands it to the exact trial count and randomizes order
(preserving class balance). One `target_class` marker is pushed per trial,
before that trial's steps run. Leave it blank to skip sending a
`target_class` marker entirely.

**System LSL Markers** panel — customizes the names of the lifecycle
markers the runner fires on its own (`baseline_start`, `baseline_end`,
`run_onset`, `run_offset`, `trial_index`, `target_class`). Per-step markers
are always free text in the step table itself. See the `target_class`
naming caveat in [`architecture.md`](architecture.md).

Also includes an **`LSLStreamCatcher`** to preview `Neurofeedback_Out`
samples live inside the tab, and a **Custom Script Executor** sub-tab that
runs an arbitrary `.py` file as a subprocess (the more general-purpose
successor to always launching the generated runner).
