# core/lsl_utils.py
"""Shared helpers for turning LSL stream metadata into channel labels."""


def resolve_channel_names(stream_info):
    """
    Builds channel labels from an LSL StreamInfo's channel description tree.

    Preferred convention: 'S(<source>)_D(<detector>)_<wavelength>', built from
    structured NIRS metadata child values. Falls back to a generic 'label'
    child value, then to a positional 'Ch-i' placeholder.

    A single underscore separates the optode-pair base name from the
    wavelength on purpose: core.signal_pipeline groups paired-wavelength
    channels by stripping the trailing '_<wavelength>' segment, so every
    stream source (real hardware or core.simulators) needs to agree on
    this format for SCI/SQA/MBLL to correctly pair up channels.
    """
    names = []
    ch = stream_info.desc().child("channels").child("channel")
    for i in range(stream_info.channel_count()):
        try:
            name = f"S({ch.child_value('source')})_D({ch.child_value('detector')})_{int(float(ch.child_value('wavelen')))}"
        except Exception:
            try:
                name = ch.child_value("label") or f"Ch-{i}"
            except Exception:
                name = f"Ch-{i}"
        names.append(name)
        ch = ch.next_sibling()
    return names