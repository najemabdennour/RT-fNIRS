# Documentation

Detailed documentation for the Real-Time fNIRS & Physiological Signals
Toolbox. For installation and a quick start, see the
[project README](../README.md).

| Document | Contents |
|---|---|
| [`architecture.md`](architecture.md) | How the pieces fit together: data flow, the shared offline/real-time signal engine, shared conventions, LSL streams and markers, channel naming |
| [`core_reference.md`](core_reference.md) | Every module under `core/` — signal steps, offline steps, CSV I/O, LSL helpers, simulators |
| [`ui_reference.md`](ui_reference.md) | Every tab under `ui/` and the shared widgets |
| [`configuration.md`](configuration.md) | `config.py` field by field: preprocessing defaults, non-signal columns, experiment profiles |
| [`dev_tools.md`](dev_tools.md) | `test_neurofeedback.py` (synthetic/replay LSL harness) and `experiment_skeleton.py` |
| [`developer_guide.md`](developer_guide.md) | Working on the package: ground rules, and step-by-step examples for adding a signal step (core + UI), an offline step, a decoder, an experiment profile, a tab |

New to the code? Read `architecture.md`, then `developer_guide.md`.
