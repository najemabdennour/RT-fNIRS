# core/montage.py
"""
Optode montage read from an LSL stream's metadata: tiles (modules), sources
and detectors with their 3D positions, head fiducials, and per-channel
source/detector/wavelength/length. Also provides a 2D projection of the
montage for display and a per-channel table for export.

Follows the metadata LUMO publishes (desc/channels, desc/probes,
desc/modules, desc/fiducial). Every part is optional: streams without
positions still yield channel/tile information, just no layout.
No Qt dependency.
"""
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Fiducials that define the head frame for the 2D projection.
_HEAD_FIDUCIALS = ('Nasion', 'Inion', 'Al', 'Ar')

# LUMO names its stream '<device> <hub serial>/<cap id>', e.g. 'LUMO HA00053/GA00396'.
_SYSTEM_NAME = re.compile(r'^\s*(\S+)\s+([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)')

# Layout-table columns holding stream-level system information (constant per table).
SYSTEM_COLUMNS = {
    'stream_name': 'system_stream_name', 'device': 'system_device', 'hub_serial': 'system_hub_serial',
    'cap_id': 'system_cap_id', 'acq_software': 'system_acq_software',
    'acq_version': 'system_acq_version', 'sampling_rate': 'system_sampling_rate_hz',
}


def parse_system_name(stream_name):
    """Splits a LUMO-style stream name into {'device', 'hub_serial', 'cap_id'} ({} if it doesn't match)."""
    match = _SYSTEM_NAME.match(stream_name or '')
    if not match:
        return {}
    return {'device': match.group(1), 'hub_serial': match.group(2), 'cap_id': match.group(3)}


@dataclass
class Montage:
    """
    Montage description parsed from stream metadata.

    Attributes:
        channels: one dict per stream channel, in stream order, with keys
            'name', 'label', 'source', 'detector', 'wavelength' (nm) and
            'length' (source-detector distance in mm); missing values are None.
        probes: optode label -> {'module', 'function' ('Source'/'Detector'),
            'position' (3-element array, mm, or None)}.
        modules: tile label -> hardware serial ('' if unknown).
        fiducials: fiducial label -> 3-element position array (mm).
        layout_2d: optode label -> saved 2D layout (x, y), e.g. from a layout
            table or a '<layout>' probe child; used when no fiducials are known.
        system: stream-level information - 'stream_name', and when known
            'device', 'hub_serial', 'cap_id' (parsed from a LUMO stream name),
            'acq_software', 'acq_version' and 'sampling_rate' (Hz).
    """
    channels: list = field(default_factory=list)
    probes: dict = field(default_factory=dict)
    modules: dict = field(default_factory=dict)
    fiducials: dict = field(default_factory=dict)
    layout_2d: dict = field(default_factory=dict)
    system: dict = field(default_factory=dict)

    @property
    def has_positions(self):
        """True if at least one optode has a 3D position."""
        return any(p['position'] is not None for p in self.probes.values())

    def summary(self):
        """One-line description: counts of tiles, sources, detectors and channels."""
        functions = [p['function'] for p in self.probes.values()]
        return (f"{len(self.modules) or len({p['module'] for p in self.probes.values() if p['module']})} tiles, "
                f"{functions.count('Source')} sources, {functions.count('Detector')} detectors, "
                f"{len(self.channels)} channels")

    def tiles(self):
        """Tile label -> {short optode id (e.g. 'A', '1'): 3D position} for optodes with positions."""
        tiles = {}
        for label, probe in self.probes.items():
            if probe['position'] is not None and probe['module']:
                tiles.setdefault(probe['module'], {})[label.split('/')[-1]] = probe['position']
        return tiles

    def system_summary(self):
        """One-line system description, e.g. 'LUMO - hub HA00053 - cap GA00396 - lumox 2.2.3 - 6.67 Hz'."""
        sys_ = self.system
        parts = [sys_.get('device') or sys_.get('stream_name') or '']
        if sys_.get('hub_serial'):
            parts.append(f"hub {sys_['hub_serial']}")
        if sys_.get('cap_id'):
            parts.append(f"cap {sys_['cap_id']}")
        if sys_.get('acq_software'):
            parts.append(f"{sys_['acq_software']} {sys_.get('acq_version') or ''}".strip())
        if sys_.get('sampling_rate'):
            parts.append(f"{float(sys_['sampling_rate']):.2f} Hz")
        return " \u00b7 ".join(p for p in parts if p)


def _float(text):
    """Parses a float, returning None for missing or malformed text."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _position(location):
    """Reads an <X>/<Y>/<Z> location element into a 3-element array, or None."""
    if location is None:
        return None
    coords = [_float(location.findtext(axis)) for axis in ('X', 'Y', 'Z')]
    return None if any(c is None for c in coords) else np.array(coords)


def parse_montage(stream_xml, channel_names=None):
    """
    Parses an LSL stream info XML document (StreamInfo.as_xml()) into a Montage.

    Args:
        stream_xml: the full stream info XML - take it from a connected
            inlet's info(), since the resolver's discovery record lacks desc.
        channel_names: optional display names in stream order (e.g. from
            core.lsl_utils.resolve_channel_names) used as each channel's 'name';
            the metadata <label> is used otherwise.
    Returns:
        Montage (empty if the XML has no <desc>).
    """
    montage = Montage()
    root = ET.fromstring(stream_xml)
    name = root.findtext('name') or ''
    montage.system = {'stream_name': name, **parse_system_name(name)}
    srate = _float(root.findtext('nominal_srate'))
    if srate:
        montage.system['sampling_rate'] = srate
    desc = root.find('desc')
    if desc is None:
        return montage
    lumo = desc.find('lumo')
    if lumo is not None:
        montage.system['acq_software'] = lumo.findtext('acqsw_name') or ''
        montage.system['acq_version'] = lumo.findtext('acqsw_ver') or ''
        fps = _float(lumo.findtext('fps'))
        if fps:
            montage.system['sampling_rate'] = fps

    channels = desc.find('channels')
    for i, ch in enumerate(channels if channels is not None else []):
        label = ch.findtext('label')
        montage.channels.append({
            'name': channel_names[i] if channel_names and i < len(channel_names) else (label or f"Ch-{i}"),
            'label': label,
            'source': ch.findtext('source'),
            'detector': ch.findtext('detector'),
            'wavelength': _float(ch.findtext('wavelen')),
            'length': _float(ch.findtext('length')),
        })

    probes = desc.find('probes')
    if probes is not None:
        elements = list(probes)
        for i, el in enumerate(elements):
            if el.tag != 'probe':
                continue
            # LUMO writes each <location> as the probe's next sibling; also accept it as a child.
            location = el.find('location')
            if location is None and i + 1 < len(elements) and elements[i + 1].tag == 'location':
                location = elements[i + 1]
            label = el.findtext('label')
            if label:
                montage.probes[label] = {
                    'module': el.findtext('module') or label.split('/')[0],
                    'function': el.findtext('function') or '',
                    'position': _position(location),
                }
                # Optional saved 2D layout (written by write_lsl_desc, not part of LUMO's metadata).
                layout = el.find('layout')
                xy = [_float(layout.findtext(a)) for a in ('X', 'Y')] if layout is not None else [None]
                if None not in xy:
                    montage.layout_2d[label] = (xy[0], xy[1])

    modules = desc.find('modules')
    for module in (modules if modules is not None else []):
        label = module.findtext('label')
        if label:
            montage.modules[label] = module.findtext('serial') or ''

    fiducials = desc.find('fiducial')
    if fiducials is not None:
        for fid in fiducials.iter('fiducial'):
            label, position = fid.findtext('label'), _position(fid.find('location'))
            if label and position is not None:
                montage.fiducials[label] = position
    return montage


def project_2d(montage, extra_points=None):
    """
    Flattens the optode (and fiducial) positions to 2D for display.

    extra_points: optional {label: 3D position} projected in the same frame,
    e.g. the empty slots of a helmet template (core.helmet_templates).

    In order of preference:
      'azimuthal' - with Nasion, Inion, Al and Ar: azimuthal equidistant
          projection about the vertex (as for EEG topomaps). The head centre
          is the mean of those four fiducials, the vertex direction points to
          Cz (or is perpendicular to the ear-ear and nasion-inion axes), and
          each point sits at its angular distance from the vertex times the
          head radius. Nose up (+y), right ear right (+x).
      'saved' - the montage's saved 2D layout (montage.layout_2d), e.g. from
          a layout table exported while the fiducials were known.
      'sphere' - without either: a sphere is fitted to the optodes and they
          are projected azimuthally about the montage's own centre, which
          keeps distances near the optodes close to the true ones.
    Returns:
        (positions, method): positions maps labels to 2D (x, y) in mm;
        method is one of the names above, or 'none' without positions.
    """
    points = {label: p['position'] for label, p in montage.probes.items() if p['position'] is not None}
    extra_points = extra_points or {}
    fid = montage.fiducials
    if points and all(name in fid for name in _HEAD_FIDUCIALS):
        points.update(fid)
        points.update(extra_points)
        center = np.mean([fid[name] for name in _HEAD_FIDUCIALS], axis=0)
        right = fid['Ar'] - fid['Al']
        front = fid['Nasion'] - fid['Inion']
        up = fid['Cz'] - center if 'Cz' in fid else np.cross(right, front)
        radius = np.mean([np.linalg.norm(fid[name] - center) for name in _HEAD_FIDUCIALS])
        return _azimuthal(points, center, up, right, radius), 'azimuthal'

    if montage.layout_2d:
        positions = dict(montage.layout_2d)
        if extra_points:
            positions.update(_align_to_saved(points, montage.layout_2d, extra_points))
        return positions, 'saved'

    points.update(extra_points)
    if len(points) >= 4:
        xyz = np.array(list(points.values()))
        center, radius = _fit_sphere(xyz)
        up = np.mean(xyz - center, axis=0)
        # Orient with the montage's widest extent left-right.
        tangent = (xyz - center) - np.outer((xyz - center) @ (up / np.linalg.norm(up)), up / np.linalg.norm(up))
        right = np.linalg.svd(tangent - tangent.mean(axis=0), full_matrices=False)[2][0]
        return _azimuthal(points, center, up, right, radius), 'sphere'
    return {}, 'none'


def _align_to_saved(points, saved_2d, extra_points):
    """
    Places extra 3D points into a saved 2D layout.

    Saved layouts are azimuthal projections (see project_2d), so the
    projection that produced this one - centre, vertex direction, rotation,
    radius and a 2D shift - is recovered by least squares from the optodes
    present in both, starting from a sphere fitted to them, and then applied
    to the extra points. Returns {} if fewer than 4 optodes are shared.
    """
    from scipy.optimize import least_squares

    shared = [lbl for lbl in points if lbl in saved_2d]
    if len(shared) < 4:
        return {}
    xyz = np.array([points[lbl] for lbl in shared])
    target = np.array([saved_2d[lbl] for lbl in shared], dtype=float)
    center0, radius0 = _fit_sphere(xyz)

    def frame(params):
        """(center, up, right, radius, shift) from the parameter vector."""
        center, (theta, phi, psi), radius, shift = params[:3], params[3:6], params[6], params[7:9]
        up = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        helper = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(up, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        return center, up, np.cos(psi) * e1 + np.sin(psi) * e2, radius, shift

    def project(params, pts):
        """Azimuthal projection of an (n, 3) array under the parameters."""
        center, up, right, radius, shift = frame(params)
        front = np.cross(up, right)
        v = pts - center
        norm = np.linalg.norm(v, axis=1)
        angle = np.arccos(np.clip((v @ up) / np.maximum(norm, 1e-12), -1, 1))
        planar = np.column_stack([v @ right, v @ front])
        planar /= np.maximum(np.linalg.norm(planar, axis=1, keepdims=True), 1e-12)
        return planar * (angle * radius)[:, None] + shift

    best = None
    mean_dir = np.mean(xyz - center0, axis=0)
    mean_dir /= np.linalg.norm(mean_dir)
    for up0 in (mean_dir, np.array([0, 0, 1.0])):
        theta0, phi0 = np.arccos(np.clip(up0[2], -1, 1)), np.arctan2(up0[1], up0[0])
        for psi0 in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            x0 = np.r_[center0, theta0, phi0, psi0, radius0, 0.0, 0.0]
            for mirror in (1, -1):
                # A reflected layout is fitted by mirroring x before solving.
                tgt = target * [mirror, 1]
                fit = least_squares(lambda p: (project(p, xyz) - tgt).ravel(), x0, method='lm', max_nfev=2000)
                if best is None or fit.cost < best[0]:
                    best = (fit.cost, fit.x, mirror)
                if best[0] < 1e-6 * len(xyz):  # exact fit found: stop searching
                    break
            if best[0] < 1e-6 * len(xyz):
                break
        if best[0] < 1e-6 * len(xyz):
            break
    _, params, mirror = best
    labels = list(extra_points)
    projected = project(params, np.array([extra_points[lbl] for lbl in labels])) * [mirror, 1]
    return {lbl: (float(x), float(y)) for lbl, (x, y) in zip(labels, projected)}


def head_outline(montage, positions, method, extra_points=None):
    """
    Head outline for the 2D layout: (centre_xy, radius, approximate).

    With the head fiducials ('azimuthal' layouts) the outline passes through
    the projected Nasion/Inion/ears and is exact. Otherwise the layout's
    origin is taken as the top of the head and the radius is a quarter of
    the circumference of a sphere fitted to all 3D optode positions (the
    ear level of a projection about the vertex), so approximate=True.
    Returns None without enough positions.
    """
    if method == 'azimuthal':
        fid = [positions[name] for name in _HEAD_FIDUCIALS if name in positions]
        return (0.0, 0.0), float(np.mean([np.hypot(x, y) for x, y in fid])), False
    xyz = [p['position'] for p in montage.probes.values() if p['position'] is not None]
    xyz += list((extra_points or {}).values())
    if len(xyz) >= 4:
        _, radius = _fit_sphere(np.array(xyz))
        return (0.0, 0.0), float(np.pi / 2 * radius), True
    if positions:
        xy = np.array(list(positions.values()))
        return (0.0, 0.0), float(np.max(np.hypot(xy[:, 0], xy[:, 1])) * 1.2), True
    return None


def _fit_sphere(xyz):
    """Least-squares sphere through 3D points: returns (centre, radius)."""
    A = np.column_stack([2 * xyz, np.ones(len(xyz))])
    b = (xyz ** 2).sum(axis=1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    center = sol[:3]
    return center, float(np.sqrt(sol[3] + center.dot(center)))


def _azimuthal(points, center, up, right, radius):
    """
    Azimuthal equidistant projection of 3D points about the `up` direction
    from `center`: radius in the plane = angle from `up` x `radius`, with
    `right` as +x (orthogonalized against `up`) and up x right as +y.
    """
    up = up / np.linalg.norm(up)
    right = right - right.dot(up) * up
    right /= np.linalg.norm(right)
    front = np.cross(up, right)
    positions = {}
    for label, pos in points.items():
        v = pos - center
        norm = np.linalg.norm(v)
        if norm == 0:
            positions[label] = (0.0, 0.0)
            continue
        angle = np.arccos(np.clip(v.dot(up) / norm, -1.0, 1.0))
        planar = np.array([v.dot(right), v.dot(front)])
        planar_norm = np.linalg.norm(planar)
        direction = planar / planar_norm if planar_norm > 0 else np.zeros(2)
        x, y = angle * radius * direction
        positions[label] = (float(x), float(y))
    return positions


def montage_from_table(table, channel_names=None):
    """
    Rebuilds a Montage from a layout table saved by montage_table (CSV).

    Args:
        table: DataFrame with at least 'channel', 'source', 'detector'; the
            tile, serial, wavelength, distance, 3D and layout_* columns are
            used when present.
        channel_names: optional stream channel names. When given, the montage
            follows this order and only these channels; names missing from
            the table keep no source/detector. Without it, the table order is used.
    Returns:
        Montage (with layout_2d from the layout_* columns, and system and
        fiducials from the system_* / fiducial_* columns, if present).
    """
    def value(row, col):
        """Cell value, or None if the column is missing or the cell is empty."""
        v = row.get(col)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else v

    montage = Montage()
    first = table.iloc[0].to_dict() if len(table) else {}
    for key, column in SYSTEM_COLUMNS.items():
        v = value(first, column)
        if v is not None:
            montage.system[key] = float(v) if key == 'sampling_rate' else str(v)
    for column in table.columns:
        match = re.match(r'^fiducial_(.+)_x$', str(column))
        if match:
            xyz = [value(first, f"fiducial_{match.group(1)}_{a}") for a in 'xyz']
            if None not in xyz:
                montage.fiducials[match.group(1)] = np.array(xyz, dtype=float)
    rows = {str(r['channel']): r for r in table.to_dict('records')}
    names = list(channel_names) if channel_names is not None else list(rows)
    for name in names:
        row = rows.get(name, {})
        wl, length = value(row, 'wavelength_nm'), value(row, 'distance_mm')
        montage.channels.append({
            'name': name, 'label': name,
            'source': value(row, 'source'), 'detector': value(row, 'detector'),
            'wavelength': float(wl) if wl is not None else None,
            'length': float(length) if length is not None else None,
        })

    for row in rows.values():
        for role, function in (('source', 'Source'), ('detector', 'Detector')):
            label = value(row, role)
            if label is None or label in montage.probes:
                continue
            tile = value(row, f'{role}_tile') or str(label).split('/')[0]
            xyz = [value(row, f'{role}_{a}') for a in 'xyz']
            montage.probes[label] = {
                'module': tile, 'function': function,
                'position': np.array(xyz, dtype=float) if None not in xyz else None,
            }
            serial = value(row, f'{role}_tile_serial')
            if serial is not None:
                montage.modules[tile] = str(serial)
            xy = [value(row, f'layout_{role}_{a}') for a in 'xy']
            if None not in xy:
                montage.layout_2d[label] = (float(xy[0]), float(xy[1]))
    return montage


def write_lsl_desc(montage, desc):
    """
    Writes a montage into an LSL stream description (pylsl XMLElement,
    StreamInfo.desc()) in LUMO's layout - channels, probes with a sibling
    <location>, modules - plus a <layout> child per probe holding its saved
    2D position, so parse_montage on the receiving side recovers everything.
    Also writes the acquisition software (<lumo>) and head fiducials when known.
    """
    if montage.system.get('acq_software'):
        lumo = desc.append_child("lumo")
        lumo.append_child_value("acqsw_name", str(montage.system['acq_software']))
        lumo.append_child_value("acqsw_ver", str(montage.system.get('acq_version') or ''))
        if montage.system.get('sampling_rate'):
            lumo.append_child_value("fps", f"{float(montage.system['sampling_rate']):.6f}")
    chns = desc.append_child("channels")
    for ch in montage.channels:
        el = chns.append_child("channel")
        el.append_child_value("label", ch['label'] or ch['name'])
        el.append_child_value("type", "Intensity")
        el.append_child_value("measure", "Amplitude")
        if ch['source'] is not None:
            el.append_child_value("source", str(ch['source']))
            el.append_child_value("detector", str(ch['detector']))
        if ch['wavelength'] is not None:
            el.append_child_value("wavelen", f"{ch['wavelength']:.6f}")
        if ch['length'] is not None:
            el.append_child_value("length", f"{ch['length']:.6f}")

    probes = desc.append_child("probes")
    for label, probe in montage.probes.items():
        el = probes.append_child("probe")
        el.append_child_value("modality", "CW")
        el.append_child_value("label", label)
        el.append_child_value("module", probe['module'])
        el.append_child_value("function", probe['function'])
        if label in montage.layout_2d:
            layout = el.append_child("layout")
            layout.append_child_value("X", f"{montage.layout_2d[label][0]:.6f}")
            layout.append_child_value("Y", f"{montage.layout_2d[label][1]:.6f}")
        if probe['position'] is not None:
            loc = probes.append_child("location")
            for axis, v in zip('XYZ', probe['position']):
                loc.append_child_value(axis, f"{v:.6f}")

    modules = desc.append_child("modules")
    for label, serial in montage.modules.items():
        el = modules.append_child("module")
        el.append_child_value("label", label)
        el.append_child_value("serial", serial)

    if montage.fiducials:
        fiducials = desc.append_child("fiducial")
        for label, position in montage.fiducials.items():
            fid = fiducials.append_child("fiducial")
            fid.append_child_value("label", label)
            loc = fid.append_child("location")
            for axis, v in zip('XYZ', position):
                loc.append_child_value(axis, f"{v:.6f}")


def montage_table(montage, positions_2d=None, tile_slots=None):
    """
    Builds one row per stream channel for export.

    tile_slots: optional {tile: helmet-template slot id} adding
    source_slot/detector_slot columns (core.helmet_templates.match_tiles).

    Columns: channel, wavelength_nm, source, detector, source_tile,
    detector_tile, source_tile_serial, detector_tile_serial, same_tile,
    distance_mm, then 3D source/detector/midpoint coordinates (mm) and the
    2D layout coordinates of the source, detector and midpoint. Ends with
    stream-level columns, constant down the table: the system_* columns
    (SYSTEM_COLUMNS) and fiducial_<name>_x/y/z for each head fiducial, so a
    saved table can reproduce the stream's metadata (e.g. for Dev Mode).

    distance_mm is the source-detector separation: the stream's <length>
    when present, else the 3D distance between the optode positions, else
    empty. Coordinates are empty when positions are unknown.
    """
    positions_2d = positions_2d or {}
    nan3 = np.full(3, np.nan)
    rows = []
    for ch in montage.channels:
        src, det = ch['source'], ch['detector']
        src_probe, det_probe = montage.probes.get(src, {}), montage.probes.get(det, {})
        src_tile = src_probe.get('module') or (src.split('/')[0] if src else None)
        det_tile = det_probe.get('module') or (det.split('/')[0] if det else None)
        src_pos = src_probe.get('position')
        det_pos = det_probe.get('position')
        src_pos = nan3 if src_pos is None else src_pos
        det_pos = nan3 if det_pos is None else det_pos
        distance = ch['length']
        if distance is None and not np.isnan(src_pos).any() and not np.isnan(det_pos).any():
            distance = float(np.linalg.norm(src_pos - det_pos))
        mid = (src_pos + det_pos) / 2
        src_2d = positions_2d.get(src, (np.nan, np.nan))
        det_2d = positions_2d.get(det, (np.nan, np.nan))
        rows.append({
            'channel': ch['name'],
            'wavelength_nm': ch['wavelength'],
            'source': src, 'detector': det,
            'source_tile': src_tile, 'detector_tile': det_tile,
            'source_tile_serial': montage.modules.get(src_tile, ''),
            'detector_tile_serial': montage.modules.get(det_tile, ''),
            'same_tile': (src_tile == det_tile) if src_tile and det_tile else None,
            'distance_mm': distance,
            'source_x': src_pos[0], 'source_y': src_pos[1], 'source_z': src_pos[2],
            'detector_x': det_pos[0], 'detector_y': det_pos[1], 'detector_z': det_pos[2],
            'midpoint_x': mid[0], 'midpoint_y': mid[1], 'midpoint_z': mid[2],
            'layout_source_x': src_2d[0], 'layout_source_y': src_2d[1],
            'layout_detector_x': det_2d[0], 'layout_detector_y': det_2d[1],
            'layout_midpoint_x': (src_2d[0] + det_2d[0]) / 2, 'layout_midpoint_y': (src_2d[1] + det_2d[1]) / 2,
        })
        if tile_slots is not None:
            rows[-1]['source_slot'] = tile_slots.get(src_tile)
            rows[-1]['detector_slot'] = tile_slots.get(det_tile)
    table = pd.DataFrame(rows)
    for key, column in SYSTEM_COLUMNS.items():
        if montage.system.get(key) not in (None, ''):
            table[column] = montage.system[key]
    for name, position in montage.fiducials.items():
        for axis, v in zip('xyz', position):
            table[f"fiducial_{name}_{axis}"] = float(v)
    return table
