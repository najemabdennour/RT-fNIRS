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
| `optical_density(intensity)` | Raw intensity → optical density, `-ln(I / mean(I))` per column. If any value is ≤ 0, values are replaced by their absolute value and clipped at the smallest non-zero channel minimum, so the log is always defined. Returns `(od, n_non_positive)`. |

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
| `optical_density` | Converts raw intensity to optical density (`math_kernel.optical_density`). Reads no params. Column names are kept; the result is tagged `data.attrs['fnirs_signal'] = 'od'` so later steps adapt (see below). Skipped with a warning if the data is already OD or HbO/HbR. |
| `channel_filter` | Manual keep/exclude by name (`manual_exclude_channels` / `manual_keep_channels` in `params`). Matches both the exact column name and its base name (pre-`rsplit('_', 1)`), so a name saved before or after MBLL still matches. Independent of SCI/SQA's live thresholds — for reproducing a fixed exclusion list. |
| `sci` | Drops both wavelength columns of an optode pair if their cross-wavelength correlation is below `sci_threshold`. |
| `sqa` | Drops individual channels whose coefficient of variation exceeds `cv_threshold`. On OD input it uses `exp(-OD) = I / mean(I)`, which has exactly the CV of the original intensity, so rejections are the same as running SQA before OD. |
| `mbll` | Converts paired wavelength columns to `_HbO`/`_HbR` concentration columns via the Modified Beer-Lambert Law. Requires `mbll_wavelengths` and matching `extinction_coefficients` entries. Accepts raw intensity (converted to OD internally) or OD from the `optical_density` step (used as is; results are identical). Tags its output `'hb'`. Warns when supposed raw intensity contains values ≤ 0, which usually means an OD checkpoint CSV was reloaded without re-running the OD step. |
| `filter` | Butterworth bandpass (`low_cut`, `high_cut`, `fs`, `butterworth_order`). Checks the buffer length against `bandpass_min_samples()` first; if it is too short, or the parameters are invalid (e.g. high cut above Nyquist), the data passes through **unfiltered** and a warning naming the required length is issued via `warn()`. |
| `detrend` | Linear batch detrending via `nilearn.signal._detrend` (optional dependency — skipped with a warning if not installed). Batch count is clamped to the buffer length. |
| `scaling` | `StandardScaler` or `MinMaxScaler` (`scaler_type`) across signal columns. |
| `averaging (feature_vector)` | Real-time-only: collapses a processed buffer to its single-row mean — the feature vector a trial-averaged decoder expects. |

Each step is a separate `_step_<name>(data, params, excluded_cols)` method
registered in `self._step_handlers`; see
[`developer_guide.md`](developer_guide.md#4-worked-example-adding-a-signal-step)
for adding one.

### Signal type tracking

`DataFrame.attrs['fnirs_signal']` (`SIGNAL_TYPE_ATTR`) records what the
signal columns hold: absent = raw intensity, `'od'` = optical density,
`'hb'` = HbO/HbR. pandas carries `attrs` through every operation the
pipeline uses, but **not through a CSV**: a checkpoint saved after the OD
step reloads as "raw intensity", so reloaded OD data must not be fed to
`mbll` (it warns).

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

## `montage.py`

Reads the optode montage from an LSL stream's metadata (no Qt). Built for
the layout LUMO publishes in its stream `desc`:

| Element | Contents |
|---|---|
| `channels/channel` | `label`, `source` (e.g. `N1/A`), `detector` (`N1/1`), `wavelen` (nm), `length` (source-detector distance, mm) |
| `probes/probe` + `location` | optode `label`, `module` (tile), `function` (`Source`/`Detector`) and X/Y/Z position (mm); LUMO writes `location` as the probe's next sibling |
| `modules/module` | tile `label` (e.g. `N1`) and hardware `serial` |
| `fiducial/fiducial` | `Nasion`, `Inion`, `Al`, `Ar`, `Cz` positions |

Every part is optional. On a recorded LUMO header (48 tiles, 144 sources,
192 detectors, 7,056 channels) each channel's `length` equals the 3D
distance between its optodes' positions.

| Function | Purpose |
|---|---|
| `parse_montage(stream_xml, channel_names=None)` | Parses `StreamInfo.as_xml()` (from a connected inlet's `info()`) into a `Montage` (`channels`, `probes`, `modules`, `fiducials`, `system`; `has_positions`, `summary()`, `system_summary()`, `tiles()`). `system` holds the stream name and, for LUMO, `device`/`hub_serial`/`cap_id` (from a name like `LUMO HA00053/GA00396`, see `parse_system_name`), plus `acq_software`, `acq_version` and `sampling_rate` from `<lumo>`. `channel_names` sets each channel's display name. |
| `project_2d(montage, extra_points=None)` | 2D layout, returning `(positions, method)`. `extra_points` (e.g. empty helmet slots) are projected in the same frame; for a saved layout, the azimuthal projection that produced it is recovered by least squares from the shared optodes, so extras land exactly where a fresh projection would put them. `'azimuthal'`: with Nasion/Inion/Al/Ar, azimuthal equidistant projection about the vertex (Cz), nose up (+y), right ear right (+x), radius = angle from the vertex × head radius (mm). `'saved'`: else the montage's saved 2D layout (`layout_2d`). `'sphere'`: else a sphere is fitted to the optodes and they are projected about the montage's own centre (orientation arbitrary). |
| `head_outline(montage, positions, method, extra_points=None)` | `(centre_xy, radius, approximate)`: exact through the projected fiducials for `'azimuthal'` layouts; otherwise centred on the layout origin with radius = ¼ circumference of a sphere fitted to the 3D optodes (`approximate=True`). |
| `montage_from_table(table, channel_names=None)` | Rebuilds a `Montage` from a saved layout table, including its 2D layout (`layout_*` columns) as `layout_2d`, and `system` / `fiducials` from the `system_*` / `fiducial_*` columns when present. With `channel_names`, follows the stream's channel order (unmatched names get no optodes). |
| `write_lsl_desc(montage, desc)` | Writes a montage into an LSL `StreamInfo.desc()` in LUMO's layout, plus a `<layout>` X/Y child per probe for the saved 2D position, so `parse_montage` recovers it. Also writes `<lumo>` (acquisition software) and the fiducials when known. Used by the Dev Mode simulator. |
| `montage_table(montage, positions_2d=None, tile_slots=None)` | One row per channel: `channel`, `wavelength_nm`, `source`, `detector`, `source_tile`, `detector_tile`, their serials, `same_tile`, `distance_mm` (the stream's `length`, else computed from positions), source/detector/midpoint X/Y/Z (mm) and their 2D layout coordinates; `source_slot`/`detector_slot` with `tile_slots`; then constant `system_*` columns and `fiducial_<name>_x/y/z`, so the table can recreate the stream's metadata. |

Note: the 2D projection is for display. Distances in the flattened layout
differ from true distances (on average about 0.84× near the optodes); the
table's `distance_mm` and 3D coordinates are the true values.

---

## `config_io.py`

Reads and edits `config.py` for the Settings tab (no Qt).
`ConfigFile(path)` executes the file and holds `values` (the settings in
`EDITABLE`), `comment_for(name, key=None)` / `heading_for(name, key)` (the
documenting comments), `diff(new_values)` and `save(new_values)`. Saving
patches only changed values at their source position (AST spans): dicts
with the same keys and equal-length lists are patched entry by entry,
added/removed entries of one-entry-per-line dicts are inserted/deleted as
lines, anything else is rewritten in the file's style; paths under the
project become `os.path.join(BASE_DIR, ...)`. The patched source is
executed and compared to the intended values before an atomic replace, a
`.bak` backup is written, and `ConfigChangedError` is raised if the file
changed on disk since loading. `apply_to_module(module, values)` updates the
imported `config` module in place.

---

## `helmet_templates.py`

Helmet templates: every tile slot of a helmet, learned from LSL stream
metadata (a stream only lists the tiles plugged in), so the 2D layout can
draw the empty slots. Stored as `<key>.json` per helmet; the key is the cap
id (`template_key(montage)`, else the stream name). A template holds
`slots` (`id`, `optodes` {short id: 3D position}, `tiles_seen`,
`serials_seen`), `fiducials`, `system`, `reference`, `sources` and dates.

| Function | Purpose |
|---|---|
| `from_montage(key, montage, source="")` | Option A: a reference template made only from this montage (record once with every slot populated). |
| `merge_montage(template, montage, source="")` | Option B: adds the montage's tiles - matched slots fill in missing optodes, other tiles become new slots. Returns the number of new slots. |
| `match_tiles(template, montage)` | `{tile: slot id}` by optode position (median distance ≤ `MATCH_TOLERANCE_MM` = 5 mm, closest first, one tile per slot), independent of tile names. |
| `empty_slot_points(template, matched)` | `{'slot<id>/<optode>': position}` of the unoccupied slots, for `project_2d(..., extra_points=...)`. |
| `load_template` / `save_template` / `list_templates` / `new_template` | Storage in a folder (`config.HELMET_TEMPLATE_DIR`). |

---

## `lsl_client.py`

Three `QThread` workers (the only PyQt dependency in `core/`):

- `AdvancedStreamResolver(sig_type, use_markers, marker_name)` — resolves a
  primary data stream by `type` and, optionally, a marker stream by `name`,
  concurrently. Emits `success(primary_info, marker_info, status_msg)` or
  `failure(error_msg)`.
- `MarkerDiagnosticWorker(marker_name)` — probes for a named marker stream
  and reports its metadata (`name`, `type`, `hostname`, `uid`,
  `channels`) or a timeout failure.
- `StreamAcquisitionWorker(inlet, marker_inlet, session)` — records a live
  stream on a background thread: pulls data in chunks and passes each sample
  to `session.process_and_append` (which polls the marker inlet once per
  sample, as before). Emits `error(msg)` and exits if the inlet fails. It
  owns both inlets while running, so `stop()` it (it waits for the thread)
  before closing them.

`release_inlets_async(inlets)` closes and releases LSL inlets on a
background thread, because liblsl blocks for about 0.5 s per inlet while
tearing it down. It takes ownership: pass a list holding the only remaining
references (it is emptied), so the last reference, and the teardown, always
belong to the background thread.

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

- `SyntheticNIRSThread(name='LUMO_Synthetic', n_channels=100, fs=6.6667, montage=None)` —
  streams sine+noise samples. With a `montage` (the Online tab passes the
  one in `config.SIMULATOR_LAYOUT_FILE`) it has one channel per montage
  channel and publishes the full montage metadata like LUMO hardware;
  otherwise `n_channels` channels with generic `S(x)_D(y)_<735|850>` labels.
- `SyntheticMarkerThread(name='OpenSesame_Markers')` — streams a repeating
  fake trial cycle (`run_onset` → `trial_onset`/`trial_index`/`target` →
  `trial_offset`, about 8 s per cycle). Its waits are interruptible, so
  `stop()` returns immediately.

---

## `data_manager.py`

`DataSession` — the real-time sample/marker buffer, owned as a single
shared instance by `main_window.py` (`self.global_session_cache`).
**Thread-safe:** the acquisition thread appends while the GUI reads, so
every access takes one internal lock. Read through `snapshot()` / `len()`
rather than the raw lists, which could be caught mid-append.

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
- `snapshot(n_recent)` — a consistent copy for display:
  `(last n_recent samples, their timestamps, all cached markers, total sample count)`.
- `len(session)` — number of samples recorded.
- `unsaved_count()` — samples recorded since the last successful export
  (`saved_count` is set by `export_to_csv` and cleared by `reset()`); the
  Online tab uses it to confirm before Start discards a recording.
- `export_to_csv` builds its DataFrame under the lock, so exporting while
  recording writes a consistent cut of everything recorded so far.
