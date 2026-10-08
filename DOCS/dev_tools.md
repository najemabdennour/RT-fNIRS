# Development & Testing Tools

Two standalone scripts at the project root, outside the `core`/`ui` package
structure — neither is imported by the main app.

## `test_neurofeedback.py`

A synthetic (or real-data-replay) LSL hardware + decoder validation harness
for the real-time neurofeedback pipeline, so you can exercise
`ui/neurofeedback_tab.py` end-to-end without real hardware.

```bash
python test_neurofeedback.py                          # pure synthetic
python test_neurofeedback.py --real-data path/to/session.csv --trials 30
python test_neurofeedback.py --real-data session.csv --no-augment
```

| Flag | Default | Meaning |
|---|---|---|
| `--real-data PATH` | none | Use a real recorded/preprocessed CSV as the signal source instead of pure random data (both for training the dummy model and for streaming). |
| `--trials N` | `20` | Number of augmented training trials to draw when using `--real-data`. |
| `--no-augment` | off | Disable augmentation — train/replay the real data source verbatim (looped as-is). |
| `--noise-std X` | `0.05` | Relative Gaussian noise std added during augmentation. |
| `--scale-min X` / `--scale-max X` | `0.9` / `1.1` | Range for random per-channel amplitude scaling during augmentation. |
| `--model-name NAME` | `dummy_model` | Filename (without `.pkl`) for the generated decoder in `decoders/`. |

**What it does:**

1. If `decoders/<model-name>.pkl` doesn't already exist, trains one. With
   no `--real-data`, trains a `LogisticRegression` on pure random data
   (fast, but the model has learned nothing meaningful — useful only for
   exercising LSL plumbing). With `--real-data`, draws augmented windows
   from the real recording (`build_training_set_from_real_data`) —
   per-channel amplitude scaling, noise scaled to each channel's own
   amplitude, and a small circular time-shift, collapsed to mean feature
   vectors with alternating binary labels.
2. Streams a `LUMO_Synthetic` NIRS LSL stream. With no real data, this is
   sine + Gaussian noise on 100 unlabeled channels. With `--real-data`, it
   loops the recording (re-augmenting on each wrap-around so repeated
   playback isn't identical) and attaches the real column names as LSL
   channel-label metadata — so SCI/SQA base-channel pairing sees real
   names instead of falling back to generic `Ch-i` placeholders.
3. Runs a `feedback_monitor` thread that prints every `probability` sample
   it receives on `Neurofeedback_Out`.

Real-data mode strips non-signal columns using the shared
`config.NON_SIGNAL_COLUMNS` list (see [`configuration.md`](configuration.md#non_signal_columns))
and reads/writes CSVs through `core.io_utils`, so leaked index columns are
dropped. The training set is saved to `data/synthetic_training_data.csv`.

## `experiment_skeleton.py`

A hand-editable template for writing a custom Pygame + LSL paradigm from
scratch, predating (and structurally mirroring) the Experiment Design
tab's generated runner. Useful if you want full manual control over the
stimulus code rather than working through the step-table UI — copy this
file and edit the constants/render logic directly.

Structure: baseline recording phase → N trials, each a fixation cross →
stimulus presentation (with a pulsing circle placeholder for a motor
imagery cue) → feedback request + display (polls `Neurofeedback_Out` for
up to 2.5s) → inter-trial rest → run-complete marker.

Configuration is via module-level constants at the top of the file
(`MARKER_STREAM_NAME`, `FEEDBACK_STREAM_NAME`, `SCREEN_WIDTH`/`HEIGHT`,
`FULLSCREEN`, `BASELINE_DURATION`, `TOTAL_TRIALS`, `TRIAL_DURATION`,
`ITI_DURATION`) rather than a JSON spec or CLI flags — edit them directly
and re-run.
