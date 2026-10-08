# `config.py` Reference

Global, process-wide **defaults**. UI tabs read from here to pre-fill their
fields; nothing in the app writes back into this module at runtime (the one
exception is `register_experiment_profile()`, an explicit API for adding a
session-only experiment profile).

## Top-level

| Name | Default | Used by |
|---|---|---|
| `DEFAULT_EXPERIMENT` | `"new_experiment"` | Experiment profile pre-selected in the Offline tab at startup. Read-only; the selected profile is passed explicitly to the processing code. |
| `COADAPTATION_ACCEPTANCE_CRITERIA` | `0.8` | `neurofeedback_tab.py` — probability threshold above which a trial is accepted into the co-adaptation training set |

## `DEFAULT_PREPROC_CONFIG`

The canonical defaults for every `SignalPreprocessor` parameter. UI tabs
read individual keys to pre-fill their own fields — none of the UI writes
back into this dict, so editing a field in the app only affects that
session's run.

| Key | Default | Meaning |
|---|---|---|
| `butterworth_filter` | `True` | Offline pipeline's default "run" state for the `filter` step |
| `detrend` | `True` | Offline pipeline's default "run" state for the `detrend` step |
| `averaging` | `True` | Offline pipeline's default "run" state for the `averaging` step |
| `block_avg` | `True` | Offline pipeline's default "run" state for the `block_avg` step |
| `enable_sci` | `False` | Offline pipeline's default "run" state for the `sci` step |
| `enable_sqa` | `True` | Offline pipeline's default "run" state for the `sqa` step |
| `sqa_cv_threshold` | `7.5` | SQA coefficient-of-variation rejection limit (%) |
| `sci_correlation_min` | `0.5` | SCI minimum cross-wavelength correlation |
| `enable_mbll` | `True` | Offline pipeline's default "run" state for the `mbll` step |
| `filter_low_band` | `0.05` | Butterworth low cutoff (Hz) |
| `filter_high_band` | `0.5` | Butterworth high cutoff (Hz) |
| `butterworth_order` | `5` | Filter order |
| `detrend_batches` | `10` | Linear detrend batch count (clamped to buffer length at run time) |
| `scaling` | `True` | Offline pipeline's default "run" state for the `scaling` step |
| `scaler_type` | `"standard"` | `"standard"` or `"minmax"` |
| `peak_onset_time_buffer` | `4` | Seconds after `trial_onset` where the epoch window starts |
| `peak_offset_time_buffer` | `9` | Seconds after `trial_onset` where the epoch window ends |
| `sampling_rate` | `6.6` | Reference `fs` for filtering — **note:** `core/simulators.py`'s synthetic stream defaults to `6.6667` (`test_neurofeedback.py` uses `6.6`); reconcile manually if you rely on the exact value matching |
| `mbll_wavelengths` | `[735, 850]` | Wavelength pair MBLL expects to find per optode |
| `extinction_coefficients` | per-wavelength `[e_HbO, e_HbR]` dict (690–850nm) | MBLL absorption coefficients (Cope et al. 1991) |
| `max_plot_trial_traces` | `5` | Cap on overlay trace lines in some diagnostic plots |

## Paths

Computed relative to `config.py`'s own directory (`BASE_DIR`):

| Name | Path |
|---|---|
| `DEFAULT_RAW_RECORDINGS` | `data/raw/recordings` |
| `DEFAULT_RAW_BEHAVIORAL` | `data/raw/behavs` |
| `DEFAULT_OUT_SESSIONS` | `data/preprocessed/grouped_sessions` |
| `DEFAULT_OUT_BEHAVIORAL` | `data/preprocessed/grouped_behavs` |

## `NON_SIGNAL_COLUMNS`

The single shared list of structural / marker / label columns that are never
treated as signal channels:
`timestamps`, `trial_onset`, `trial_offset`, `run_onset`, `run_offset`,
`run`, `target`, `trial_index`, `peak_onset`, `peak_offset`, `correct`,
`response`, `index`, `block_id`.

Used by the offline pipeline, decoder training (feature selection), and
`test_neurofeedback.py`.

`get_non_signal_columns(experiment=None)` returns this list, plus — when an
experiment name is given — that profile's behavioral `columns`. The offline
pipeline uses this form, because `epoch_prep` merges behavioral columns into
the session and they must never be filtered, scaled, or averaged as channels.

## Experiment profiles

`EXPERIMENT_PROFILES` describes, per paradigm, how the offline `epoch_prep`
step aligns a behavioral log (one row per trial, in order) onto the
session's `trial_onset` rows:

| Key | Meaning |
|---|---|
| `columns` | Every column the behavioral log may contain (also excluded from signal processing). |
| `target_column` | Behavioral column copied into `target` — the decoder label. `None` → `target` defaults to `1` when absent. |
| `carry_columns` | Behavioral columns copied onto each trial-onset row and forward-filled across the trial. |
| `keep_trials_where` | `{column: value}` — only matching trials are kept (e.g. `{"correct": True}`); `{}` keeps all. |
| `behavior_required` | `True` → the pipeline refuses to run without a behavioral log (and without its `target_column`). |

Built-in profiles:

| Profile | `target_column` | `carry_columns` | `keep_trials_where` | `behavior_required` |
|---|---|---|---|---|
| `WM_exp` | `WM_load` | `trial_index`, `correct`, `response` | `{"correct": True}` | `True` |
| `Visual_perception_exp` | `condition` | `trial_index`, `duration`, `timestamp` | `{}` | `False` |
| `new_experiment` | `None` | — | `{}` | `False` |

Helpers:

- `get_experiment_profile(name)` — the named profile, or `new_experiment`'s if unknown.
- `get_behavioral_columns(name)` — that profile's `columns`.
- `register_experiment_profile(name, columns, target_column=None, carry_columns=None, keep_trials_where=None, behavior_required=False)`
  — adds a profile for the current session (used by the Offline tab's
  *Register Profile* button). If no `target_column` is given, `condition` is
  used when present; `carry_columns` defaults to every other column except
  `run` (which the session merge assigns itself).

See [`developer_guide.md`](developer_guide.md#7-adding-an-experiment-profile)
for adding a permanent profile.
