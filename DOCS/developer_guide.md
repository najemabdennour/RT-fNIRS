# Developer Guide

How to work on the toolbox and extend it: where things live, the rules the
code follows, and worked examples for the most common extensions (a new
signal-processing step, a new offline step, a new decoder, a new experiment
profile, a new tab).

Read [`architecture.md`](architecture.md) first for the big picture.

---

## 1. Setup

```bash
pip install -r requirements.txt   # plus liblsl on Linux, see README
python main.py                    # launch the app
python test_neurofeedback.py      # fake NIRS hardware + dummy decoder (separate terminal)
```

Without hardware, either tick **Enable LSL Simulators (Dev Mode)** in the
Online Streaming Monitor tab, or run `test_neurofeedback.py`
(see [`dev_tools.md`](dev_tools.md)).

There is no automated test suite yet. Verify changes with a small script
that calls the `core/` function directly (see [§9](#9-testing-your-change)),
then in the app.

---

## 2. Ground rules

| Rule | Why |
|---|---|
| **`core/` holds the logic, `ui/` only collects parameters and displays results.** `core/` imports no PyQt except for `QThread` workers. | Lets `core/` be scripted and tested without a GUI. |
| **One signal engine for offline and real time.** Channel-level processing lives only in `core/signal_pipeline.py`. | A model trained on offline features must see the same transformations live. |
| **Read/write CSVs only through `core.io_utils.read_csv` / `write_csv`.** | Strips leaked pandas index columns (`Unnamed: 0`) so they never become features. |
| **Non-signal columns come from `config`.** Use `config.NON_SIGNAL_COLUMNS` or `config.get_non_signal_columns(experiment)`; never write your own list. | One definition of "what is a channel". |
| **No mutable global state.** Don't write to `config` at runtime; pass the selected experiment/parameters explicitly as arguments. | Tabs can't silently affect each other. |
| **Report through callbacks, not `print`.** Use `self.log(...)` for progress and `self.warn(...)` for non-fatal problems (e.g. a skipped step). Raise exceptions for fatal ones. | Messages appear in the right tab's console at the right level. |
| **Catch expected errors where they happen.** Show a specific `QMessageBox` or log line. Anything uncaught is reported by `ui/exception_handler.py` (traceback in the terminal and an error dialog), and the app keeps running. | The global handler is a safety net, not a substitute for a clear error message. |
| **Every function has a docstring.** Comments explain *why*, not *what*. | Keeps the code readable for the next developer. |

---

## 3. How the signal pipeline works

`core/signal_pipeline.py` defines `SignalPreprocessor`. Each processing step
is a separate **strategy method** registered in a dispatch table:

```python
SIGNAL_STEP_ORDER = ['channel_filter', 'sci', 'sqa', 'mbll', 'filter', 'detrend', 'scaling']

class SignalPreprocessor:
    def __init__(self, log_callback=None, warning_callback=None):
        ...
        self._step_handlers = {
            'channel_filter': self._step_channel_filter,
            'sci': self._step_sci,
            'sqa': self._step_sqa,
            'mbll': self._step_mbll,
            'filter': self._step_filter,
            'detrend': self._step_detrend,
            'scaling': self._step_scaling,
            'averaging (feature_vector)': self._step_reduce_to_feature_vector,
        }

    def run_steps(self, data, pipeline_steps, params, excluded_cols=None):
        for step in pipeline_steps:                  # e.g. {'id': 'filter', 'run': True}
            ...
            data = self._step_handlers[step_id](data, params, excluded_cols)
        return data
```

The pipeline is built from a few small pieces:

- A **step id** (`'filter'`) is the string the UI tables, `pipeline_steps`
  lists and checkpoint filenames use.
- `run_steps` walks the caller's ordered list and calls each handler. It
  ignores ids it doesn't know, so the offline pipeline can mix these steps
  with its own structural steps in one list.
- Every handler has the same **contract**:

  ```python
  def _step_xxx(self, data, params, excluded_cols) -> pd.DataFrame
  ```

  | Argument | Meaning |
  |---|---|
  | `data` | DataFrame of channel columns, plus any non-signal columns |
  | `params` | flat dict of settings. Read with `params.get(key, default)` so missing keys never crash |
  | `excluded_cols` | columns the step must not touch (timestamps, markers, behavioral fields) |
  | returns | the processed DataFrame. Rows may be dropped only if the step is meant to do that, and columns only if the step drops channels, as SCI and SQA do |

- **Pure math goes in `core/math_kernel.py`.** These are numpy/scipy
  functions with no pandas, no logging and no state. The step method does the
  DataFrame plumbing, the logging and the edge-case handling around them.
- `SIGNAL_STEP_ORDER` documents the physically sensible default order.
  Users can still reorder steps in the UI.

The same engine runs in:

- the **Offline tab**, through `OfflineDataPreprocessor.run_preprocessing_pipeline`;
- the **Neurofeedback tab**, through `NeurofeedbackWorker._process_and_predict`, on a baseline + trial buffer.

---

## 4. Worked example: adding a signal processing step

As an example, we'll add **Savitzky–Golay smoothing** as a new step with id
`smooth`. The same five edits apply to any channel-level step.

### 4.1 The math: `core/math_kernel.py`

```python
from scipy.signal import savgol_filter

def savgol_smooth(data, window_length=11, polyorder=3):
    """
    Smooths each column with a Savitzky-Golay filter (local polynomial fit),
    preserving peak shape better than a moving average.

    window_length must be odd and greater than polyorder.
    """
    return savgol_filter(data, window_length, polyorder, axis=0)
```

### 4.2 The strategy: `core/signal_pipeline.py`

Import the function, write the handler, register it, and add it to the
documented order:

```python
from core.math_kernel import (
    ...,
    savgol_smooth,
)

SIGNAL_STEP_ORDER = ['channel_filter', 'sci', 'sqa', 'mbll', 'smooth', 'filter', 'detrend', 'scaling']

# in __init__:
        self._step_handlers = {
            ...
            'smooth': self._step_smooth,
        }

    def _step_smooth(self, data, params, excluded_cols):
        """
        Savitzky-Golay smoothing of every signal column.

        params:
          'smooth_window' (default 11): window length in samples, forced odd
          'smooth_polyorder' (default 3): polynomial order
        Skipped with a warning when the buffer is shorter than the window.
        """
        window = int(params.get('smooth_window', 11)) | 1   # force odd
        polyorder = int(params.get('smooth_polyorder', 3))
        signal_cols = [c for c in data.columns if c not in excluded_cols]
        if not signal_cols:
            return data

        if len(data) < window or polyorder >= window:
            self.warn(
                f"Smoothing SKIPPED - needs at least {window} samples and polyorder < window "
                f"(got {len(data)} samples, polyorder={polyorder}). Data passed through UNSMOOTHED."
            )
            return data

        self.log(f"[Pipeline Engine] Applying Savitzky-Golay smoothing (window={window}, order={polyorder})...")
        data = data.copy()
        data[signal_cols] = savgol_smooth(data[signal_cols].values, window, polyorder)
        return data
```

The handler follows these rules:

- **Exclusions.** It takes `signal_cols` from `excluded_cols`, so markers and
  labels are never smoothed.
- **Defaults.** It reads every parameter with a default.
- **Short buffers.** It checks the buffer length first and calls
  `self.warn(...)` instead of crashing or skipping silently. Real-time
  buffers can be short; see how `_step_filter` uses
  `bandpass_min_samples()` for the same purpose.
- **Copying.** It copies before writing, so the caller's DataFrame isn't
  changed in place.

At this point the step already works from Python:

```python
engine = SignalPreprocessor()
out = engine.run_steps(df, [{'id': 'smooth', 'run': True}], {'smooth_window': 9}, config.NON_SIGNAL_COLUMNS)
```

### 4.3 Defaults: `config.py`

Add the step's defaults to `DEFAULT_PREPROC_CONFIG`, so the UI pre-fills
from one place:

```python
    "enable_smooth": False,          # default 'Execute' state of the smoothing step
    "smooth_window": 11,             # Savitzky-Golay window (samples, odd)
    "smooth_polyorder": 3,           # Savitzky-Golay polynomial order
```

### 4.4 The UI: Offline tab (`ui/offline_tab.py`)

There are three edits in `init_ui` / `execute_preprocessing_only`.

**(a) Add a row to the step table.** Rows are
`(step_id, title, subtitle, default_run[, default_save])` tuples in
`default_steps`, and the order of the list is the default execution order:

```python
        default_steps = [
            ...
            ("mbll", "Convert to HbO/HbR (MBLL)", "...", defaults["enable_mbll"]),
            ("smooth", "Savitzky-Golay Smoothing", "Polynomial smoothing that preserves peak shape", defaults["enable_smooth"]),
            ("epoch_prep", ...),
            ...
        ]
```

`PipelineStepTable` (in `ui/components.py`) adds the Execute and Checkpoint
checkboxes and the Move Up/Down reordering automatically.
`step_table.compiled_steps()` then returns `{'id': 'smooth', 'run': ..., 'save': ...}`.

**(b) Add widgets for the parameters** to `param_grid` (rows 0–3 are already
used):

```python
        self.txt_smooth_window = QLineEdit(str(defaults["smooth_window"]))
        self.txt_smooth_poly = QLineEdit(str(defaults["smooth_polyorder"]))
        param_grid.addWidget(QLabel("Smoothing Window (samples):"), 4, 0); param_grid.addWidget(self.txt_smooth_window, 4, 1)
        param_grid.addWidget(QLabel("Smoothing Poly Order:"), 4, 2); param_grid.addWidget(self.txt_smooth_poly, 4, 3)
```

**(c) Pass the values** in the `params` dict built in
`execute_preprocessing_only`:

```python
                'smooth_window': int(self.txt_smooth_window.text()),
                'smooth_polyorder': int(self.txt_smooth_poly.text()),
```

`OfflineDataPreprocessor` needs no change: `run_preprocessing_pipeline`
sends every id the signal engine knows (`available_steps()`) to it
automatically, and handles checkpoint saving.

### 4.5 The UI: Neurofeedback tab (`ui/neurofeedback_tab.py`)

There are the same three edits:

1. Add the tuple to `pipeline_steps_spec`, in the same position as offline.
   Default it to `False`, like the other real-time steps:

   ```python
            ("smooth", "Savitzky-Golay Smoothing",
             "Polynomial smoothing. Enable only if the deployed model was trained on smoothed data.",
             False),
   ```

2. Add `self.txt_smooth_window` / `self.txt_smooth_poly` widgets to its
   `param_grid` (rows 0–2 are used, so use row 3).
3. Add both keys to the dict returned by `compile_preproc_params()`.

`NeurofeedbackWorker` needs no change, since it passes the compiled steps and
params straight to `SignalPreprocessor.run_steps`.

> **Keep offline and real time in sync.** A decoder only works live if the
> live buffer goes through the same steps, in the same order, with the same
> parameters as its training data. When you add a step offline, add it to
> the Neurofeedback pipeline too, and note the new step in the docs.

### 4.6 Document it

Add a row to the steps table in [`core_reference.md`](core_reference.md), the
new keys to [`configuration.md`](configuration.md), and the new row to both
pipeline lists in [`ui_reference.md`](ui_reference.md).

---

## 5. Adding an offline-only (session-structural) step

Steps that need the whole session or the behavioral log, such as epoching or
averaging across trials, live in `core/offline_processor.py`, not in the
signal engine. The signal engine must stay usable on a single live buffer.

```python
class OfflineDataPreprocessor:
    def __init__(...):
        self._structural_handlers = {
            'epoch_prep': self._step_epoch_prep,
            'averaging': self._step_averaging,
            'block_avg': self._step_block_avg,
            'baseline_correct': self._step_baseline_correct,   # new
        }

    def _step_baseline_correct(self, data, params, excluded_cols, behav_df):
        """Subtracts each epoch's pre-stimulus mean from its signal columns."""
        ...
        return data
```

The signature adds `behav_df` (may be `None`). `excluded_cols` here already
includes the active experiment's behavioral columns. Then add the row to the
Offline tab's `default_steps`, as in §4.4(a).

---

## 6. Adding a decoder

Add one entry to `DECODER_REGISTRY` in `core/decoders.py`:

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

DECODER_REGISTRY = {
    ...
    "lda": lambda: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
}
```

The value must be a **zero-argument factory** (a lambda), so every call
returns a fresh, unfitted model. The new key then appears automatically in:

- the Decoder tab's checklist and *Decoder to Train* dropdown;
- the Neurofeedback tab's co-adaptation dropdown.

The real-time engine calls `predict_proba`. For a classifier without it,
such as `LinearSVC`, wrap it in `CalibratedClassifierCV`, as the `svm` entry
does.

---

## 7. Adding an experiment profile

Experiment profiles tell the offline `epoch_prep` step how to line up a
behavioral log with the trial-onset markers. Add one to
`config.EXPERIMENT_PROFILES` (all keys are described in
[`configuration.md`](configuration.md#experiment-profiles)):

```python
    "MotorImagery_exp": {
        "columns": ['trial_index', 'hand', 'rt', 'run'],
        "target_column": "hand",               # becomes 'target', the decoder label
        "carry_columns": ['trial_index', 'rt'],
        "keep_trials_where": {},               # e.g. {"correct": True}
        "behavior_required": True,
    },
```

The profile appears in the Offline tab's *Active Experiment* dropdown. Its
columns are excluded from signal processing automatically, through
`config.get_non_signal_columns(experiment)`. To add a profile for the
current session only, use the *Register Profile* fields in the Offline tab.
They call `config.register_experiment_profile()`.

---

## 8. Adding a tab

1. Create `ui/my_tab.py` with a `QWidget` subclass. Build the widgets in
   `init_ui()`, and keep all logic in `core/`.
2. Register it in `ui/main_window.py` → `init_global_ui()`:

   ```python
   self.tab_mine = MyTab(self)
   self.tabs_container.addTab(self.tab_mine, "My Analysis")
   ```

3. For anything slow (LSL resolution, long computations), use a `QThread`
   that sends results back through `pyqtSignal`s, so the GUI never freezes.
   `core/lsl_client.py` and `NeurofeedbackWorker` are the patterns to copy.
   Never touch widgets directly from the worker thread.
4. For a log console, copy the `append_status_log(level, message)` pattern
   from the Offline or Decoder tab. Levels are SYSTEM / INFO / SUCCESS /
   WARNING / ERROR.

---

## 9. Testing your change

Exercise the `core/` piece directly on a small synthetic DataFrame. Two
habits help:

- **Don't leave anything behind.** Set `PYTHONDONTWRITEBYTECODE=1` so no
  `__pycache__` folders are written into the project, and put throwaway
  scripts outside the repo.
- **Check against a known answer.** For example, compare a smoothed signal
  with `scipy` called directly. Also check edge cases: a buffer shorter than
  the filter needs, an empty DataFrame, and missing `params` keys.

```python
import numpy as np, pandas as pd, config
from core.signal_pipeline import SignalPreprocessor

df = pd.DataFrame(np.random.rand(200, 2) + 1, columns=['S1_D1_735', 'S1_D1_850'])
warnings = []
engine = SignalPreprocessor(log_callback=print, warning_callback=warnings.append)
out = engine.run_steps(df, [{'id': 'smooth', 'run': True}], {'smooth_window': 9}, config.NON_SIGNAL_COLUMNS)
short = engine.run_steps(df.iloc[:5], [{'id': 'smooth', 'run': True}], {}, [])
assert warnings, "short buffer should warn"
```

Then run it end to end in the app:

1. Offline tab on a recorded session, with a checkpoint saved for your step;
   open the checkpoint CSV to check the result.
2. For real-time steps, run `test_neurofeedback.py` against the
   Neurofeedback tab.

---

## 10. Where things are

| I want to... | Go to |
|---|---|
| change filter / SCI / SQA / MBLL math | `core/math_kernel.py` |
| add or change a channel-level step | `core/signal_pipeline.py` |
| change epoching or averaging | `core/offline_processor.py` |
| change default parameters | `config.py` → `DEFAULT_PREPROC_CONFIG` |
| change which columns count as non-signal | `config.py` → `NON_SIGNAL_COLUMNS` |
| change how CSVs are read or written | `core/io_utils.py` |
| change how channel names are built from LSL metadata | `core/lsl_utils.py` |
| change live acquisition / recording | `ui/online_tab.py`, `core/data_manager.py` |
| change the real-time decode loop | `ui/neurofeedback_tab.py` → `NeurofeedbackWorker` |
| change the stimulus runner (pygame) | `ui/experiment_tab.py` → `RUNNER_CODE` |
| change the reorderable step table widget | `ui/components.py` → `PipelineStepTable` |
