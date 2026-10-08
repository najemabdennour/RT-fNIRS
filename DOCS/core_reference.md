# `core/` Reference

Backend logic, largely free of PyQt. See [`architecture.md`](architecture.md)
for how these pieces fit together.

---

## `math_kernel.py`

Pure numpy/scipy math, no pandas, no state.

| Function | Purpose |
|---|---|
| `butter_bandpass_filter(data, lowcut, highcut, fs, order)` | Zero-phase Butterworth bandpass via SOS coefficients + `sosfiltfilt`. |
| `bandpass_min_samples(lowcut, highcut, fs, order)` | Minimum buffer length `butter_bandpass_filter` accepts (mirrors scipy's `sosfiltfilt` padding rule) — e.g. 34 samples for the order-5, 0.05–0.5 Hz, 6.6 Hz default. |
| `calculate_scalp_coupling_index(w1_series, w2_series)` | Pearson correlation between two wavelength channels of an optode pair; 0.0 if either has zero variance. |
| `calculate_coefficient_of_variation(series)` | `std / mean * 100`; `inf` if mean is 0. |
| `apply_mbll_inversion(od_w1, od_w2, e_inv)` | Applies the inverted extinction-coefficient matrix to two optical-density series, returning stacked HbO/HbR concentration rows. |

---

## `signal_pipeline.py`

`SignalPreprocessor` — the shared channel-level pipeline engine, used
unmodified by both `offline_processor.py` and `neurofeedback_tab.py`.

```python
engine = SignalPreprocessor(log_callback=my_logger, warning_callback=my_warning_logger)
result = engine.run_steps(dataframe, pipeline_steps, params, excluded_cols=[...])
```

- `log_callback` receives progress messages; `warning_callback` receives
  non-fatal problems (e.g. a step skipped on a too-short buffer) via
  `engine.warn()`. Without a `warning_callback`, warnings go to
  `log_callback` with a `[WARNING]` prefix; with neither, they are raised as
  Python `RuntimeWarning`s.

- `pipeline_steps`: ordered list of `{'id': str, 'run': bool}` dicts (an
  optional `'save'` key is ignored by this engine but used by
  `offline_processor.py` for checkpointing).
- `params`: flat dict of thresholds/settings (see
  [`configuration.md`](configuration.md) for the full key list).
- `excluded_cols`: non-signal columns (timestamps, markers, behavioral
  fields) that every step leaves untouched.
- Unrecognized step ids are silently skipped, so callers can mix these with
  offline-only structural steps in one ordered list.

### Steps (`SIGNAL_STEP_ORDER`)

| id | What it does |
|---|---|
| `channel_filter` | Manual keep/exclude by name (`manual_exclude_channels` / `manual_keep_channels` in `params`). Matches both the exact column name and its base name (pre-`rsplit('_', 1)`), so a name saved before or after MBLL still matches. Independent of SCI/SQA's live thresholds — for reproducing a fixed exclusion list. |
| `sci` | Drops both wavelength columns of an optode pair if their cross-wavelength correlation is below `sci_threshold`. |
| `sqa` | Drops individual channels whose coefficient of variation exceeds `cv_threshold`. |
| `mbll` | Converts paired wavelength columns to `_HbO`/`_HbR` concentration columns via the Modified Beer-Lambert Law. Requires `mbll_wavelengths` and matching `extinction_coefficients` entries. |
| `filter` | Butterworth bandpass (`low_cut`, `high_cut`, `fs`, `butterworth_order`). Checks the buffer length against `bandpass_min_samples()` first; if it is too short, or the parameters are invalid (e.g. high cut above Nyquist), the data passes through **unfiltered** and a warning naming the required length is issued via `warn()`. |
| `detrend` | Linear batch detrending via `nilearn.signal._detrend` (optional dependency — skipped with a warning if not installed). Batch count is clamped to the buffer length. |
| `scaling` | `StandardScaler` or `MinMaxScaler` (`scaler_type`) across signal columns. |
| `averaging (feature_vector)` | Real-time-only: collapses a processed buffer to its single-row mean — the feature vector a trial-averaged decoder expects. |

Each step is a separate `_step_<name>(data, params, excluded_cols)` method
registered in `self._step_handlers`; see
[`developer_guide.md`](developer_guide.md#4-worked-example-adding-a-signal-step)
for adding one.

### Rejection tracking

`sci` and `sqa` append to `self.rejected_channels` as they drop columns
(SCI logs the shared base name since it drops a pair together; SQA logs the
exact column name since it can drop one wavelength independently).
`get_rejected_channel_names()` returns the deduplicated, sorted list;
`reset_rejections()` clears it — call this at the start of each fresh run.
`OfflineDataPreprocessor.get_rejected_channels()` is a passthrough to this.

---

## `channel_lists.py`

Two functions, no pandas/UI dependency:

- `save_channel_list(path, channels, notes="")` — writes a sorted,
  de-duplicated JSON file (`{"channels": [...], "metadata": {...}}`).
  Returns the list actually written.
- `load_channel_list(path)` — reads it back. Also tolerates a bare JSON
  list, or a flat comma/newline-separated text file.

Used by `offline_tab.py` (to export SCI/SQA rejections) and
`neurofeedback_tab.py` (to load that same list into the real-time
`channel_filter` step) — see the real-time preprocessing caveat in
[`architecture.md`](architecture.md).

---

## `decoders.py`

`DECODER_REGISTRY`: a dict mapping string keys to zero-arg classifier
factories, all seeded with `RANDOM_STATE = 12345`:

`svm`, `svmlinear`, `decisiontree`, `extratree`, `randomforest`,
`extratrees`, `bagging`, `gradientboosting`, `adaboost`, `naivebayes`,
`kneighbors`, `mlp`, `sgd`, `logisticregression`.

`decoder_selection(preprocessing=False, selected_method="extratrees")`
builds the pipeline for a given key. If `preprocessing=True`, wraps the
classifier in `make_pipeline(VarianceThreshold(), classifier)`. Raises
`ValueError` for an unregistered key.

---

## `offline_processor.py`

Two classes:

### `OfflineSessionCombiner` (static methods)

- `discover_folders_by_date(recordings_root, behavioral_root, target_date=None)`
  — finds subfolders matching a date string (defaults to today) and returns
  the **most recently modified** match for each root; falls back to the root
  folder itself if no match is found. (Not currently called by the UI.)
- `process_and_combine(rec_dir, beh_dir, out_rec_dir, out_beh_dir, experiment=config.DEFAULT_EXPERIMENT)` —
  concatenates every CSV in `rec_dir` (trimming each to start at its first
  valid `run_onset`, tagging with a `run` column), optionally merges
  matching behavioral CSVs, and writes `Session_<date>.csv` /
  `behav_<date>.csv`. `experiment` only sets the columns of the empty
  behavioral frame returned when no logs are found.

### `OfflineDataPreprocessor`

Wraps a `SignalPreprocessor` instance (`self.signal_engine`) and adds three
session-structural steps that need a full session + optional behavioral
alignment:

| id | What it does |
|---|---|
| `epoch_prep` | Computes `peak_onset`/`peak_offset` windows (`peak_onset_buffer`/`peak_offset_buffer` seconds after each `trial_onset`), aligns the behavioral log onto trial-onset rows as described by the experiment profile (`params['active_experiment']` → `config.EXPERIMENT_PROFILES`: `target_column` → `target`, `carry_columns` copied and forward-filled, `keep_trials_where` filtering), then slices the session into per-trial epochs numbered in an `index` column. Raises `ValueError` if the profile requires a behavioral log (or its target column) and none is given; warns about other missing profile columns. |
| `averaging` | Collapses each trial's epoch to one mean row per `(trial, target, run)`. Labels keep their type, so text conditions (e.g. `left`/`right`) work; whole-number floats such as `2.0` are normalized to `2`. |
| `block_avg` | Groups consecutive same-target rows within a run into blocks and averages each block. |

`run_preprocessing_pipeline(session_df, behav_df, params, output_dir, today_str)`
walks an ordered step list, routing each step id to either the signal
engine or the structural handlers above, optionally checkpointing to
`data_checkpoint_<step_id>_<date>.csv` when a step's `'save'` flag is set.
Calls `signal_engine.reset_rejections()` at the start of every run. The
excluded (non-signal) columns passed to every step are
`config.get_non_signal_columns(params['active_experiment'])`.
`OfflineDataPreprocessor(log_callback=None, warning_callback=None)` passes
both callbacks through to its signal engine.

---

## `io_utils.py`

CSV read/write helpers that keep leaked DataFrame index columns out of the
toolbox's data. A CSV written with pandas' default `index=True` gains a
nameless leading column that reads back as `Unnamed: 0`; left in place it
becomes a "channel" or decoder feature. **All CSV I/O in the package goes
through these helpers.**

| Function | Purpose |
|---|---|
| `find_index_columns(columns)` | Names matching `^Unnamed: \d+$`. |
| `drop_index_columns(df, log_callback=None, source="")` | Drops them, reporting what was removed. |
| `read_csv(path, log_callback=None, **kwargs)` | `pd.read_csv` + `drop_index_columns`. |
| `write_csv(df, path, log_callback=None, **kwargs)` | Drops index columns, then writes with `index=False` (always). |

The epoch-id column named `index` created by `epoch_prep` is a real column
and is deliberately not matched.

---

## `lsl_client.py`

Two `QThread` workers (the only PyQt dependency in `core/`):

- `AdvancedStreamResolver(sig_type, use_markers, marker_name)` — resolves a
  primary data stream by `type` and, optionally, a marker stream by `name`,
  concurrently. Emits `success(primary_info, marker_info, status_msg)` or
  `failure(error_msg)`.
- `MarkerDiagnosticWorker(marker_name)` — probes for a named marker stream
  and reports its metadata (`name`, `type`, `hostname`, `uid`,
  `channels`) or a timeout failure.

---

## `lsl_utils.py`

`resolve_channel_names(stream_info)` — builds channel labels from an LSL
`StreamInfo`'s metadata tree. Preferred: `S(<source>)_D(<detector>)_<wavelength>`
from structured `source`/`detector`/`wavelen` child values; falls back to a
`label` child value, then to `Ch-<i>`.

**Important:** call this on a fully-connected `StreamInlet`'s `.info()`,
not on the lightweight record `resolve_byprop()` itself returns —
`resolve_byprop` doesn't carry an outlet's full custom XML metadata; only
`inlet.info()` pulls the complete description. (Both `online_tab.py` and `neurofeedback_tab.py` do this.)

---

## `simulators.py`

Two `QThread`s for development without real hardware, both launched from
`online_tab.py`'s synthetic-data checkbox:

- `SyntheticNIRSThread(name='LUMO_Synthetic', n_channels=100, fs=6.6667)` —
  streams sine+noise samples, with `S(x)_D(y)_<735|850>` channel-label
  metadata attached.
- `SyntheticMarkerThread(name='OpenSesame_Markers')` — streams a repeating
  fake trial cycle (`run_onset` → `trial_onset`/`trial_index`/`target` →
  `trial_offset`, looped every 3s).

---

## `data_manager.py`

`DataSession` — the real-time sample/marker buffer, owned as a single
shared instance by `main_window.py` (`self.global_session_cache`).

- `process_and_append(sample, timestamp, marker_inlet)` — appends a data
  sample, and non-blockingly pulls one marker sample per call, sorting it
  into `trial_onset_marker`/`trial_offset_marker`/`run_onset_marker`/
  `run_offset_marker` (null-padding the columns that didn't fire) plus a
  `cached_visual_markers` list of `(timestamp, label)` pairs for plotting
  vertical marker lines.
- `export_to_csv(file_path, channel_names)` — writes the buffered arrays to
  a DataFrame with those four marker columns plus `timestamps` (via
  `core.io_utils.write_csv`), matching exactly what `offline_processor.py`
  expects to read back in.
- `reset()` — clears all buffers for a fresh recording session.
