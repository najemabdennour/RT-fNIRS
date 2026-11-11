# core/helmet_templates.py
"""
Helmet templates: every tile slot (dock) of a helmet, learned from LSL stream
metadata, so the 2D layout can also draw the slots that are empty in the
current montage.

An LSL stream only describes the tiles that are plugged in. A template
collects their optode positions across sessions:
  - from a reference recording with every slot populated (from_montage), or
  - by accumulating the tiles of each stream seen (merge_montage).

Tiles are matched to slots by optode position, not by tile name: LUMO
numbers tiles ('N13') by how they are connected, which can change between
sessions, while positions on the same helmet stay fixed. Templates are
stored per helmet as JSON, keyed by the cap id in the stream name
(e.g. 'GA00396' from 'LUMO HA00053/GA00396'). No Qt dependency.
"""
import json
import os
import re
from datetime import datetime

import numpy as np

# Max median distance (mm) between a tile's optodes and a slot's for them to be the same slot.
MATCH_TOLERANCE_MM = 5.0


def template_key(montage):
    """Template name for a montage: its cap id, else its sanitized stream name, else None."""
    key = montage.system.get('cap_id') or montage.system.get('stream_name')
    if not key:
        return None
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(key)).strip('_') or None


def new_template(key):
    """An empty template."""
    now = datetime.now().isoformat(timespec='seconds')
    return {'key': key, 'reference': False, 'system': {}, 'fiducials': {}, 'slots': [],
            'created': now, 'updated': now, 'sources': []}


def _slot_distance(tile, slot):
    """Median distance (mm) between same-id optodes of a tile and a slot, or inf if under 2 are shared."""
    shared = [o for o in tile if o in slot['optodes']]
    if len(shared) < 2:
        return np.inf
    return float(np.median([np.linalg.norm(np.asarray(tile[o]) - np.asarray(slot['optodes'][o])) for o in shared]))


def match_tiles(template, montage):
    """
    Matches the montage's tiles to template slots by optode position.

    Returns:
        {tile label: slot id} for tiles within MATCH_TOLERANCE_MM of a slot
        (each slot used at most once, closest pairs first).
    """
    candidates = []
    for tile, optodes in montage.tiles().items():
        for slot in template['slots']:
            d = _slot_distance(optodes, slot)
            if d <= MATCH_TOLERANCE_MM:
                candidates.append((d, tile, slot['id']))
    matched, used_slots = {}, set()
    for _, tile, slot_id in sorted(candidates):
        if tile not in matched and slot_id not in used_slots:
            matched[tile] = slot_id
            used_slots.add(slot_id)
    return matched


def merge_montage(template, montage, source=""):
    """
    Adds the montage's tiles to the template (option B, accumulation).

    Tiles matching a slot fill in any optodes the slot was missing and are
    recorded in its 'tiles_seen'/'serials_seen'; other tiles become new
    slots. System info and fiducials are taken from the montage if the
    template has none yet.

    Returns:
        Number of new slots added (the template is modified in place).
    """
    matched = match_tiles(template, montage)
    slots = {slot['id']: slot for slot in template['slots']}
    added = 0
    for tile, optodes in sorted(montage.tiles().items(), key=lambda t: _natural_key(t[0])):
        if tile in matched:
            slot = slots[matched[tile]]
            for o, pos in optodes.items():
                slot['optodes'].setdefault(o, [float(v) for v in pos])
        else:
            slot = {'id': max(slots, default=0) + 1, 'optodes': {o: [float(v) for v in p] for o, p in optodes.items()},
                    'tiles_seen': [], 'serials_seen': []}
            template['slots'].append(slot)
            slots[slot['id']] = slot
            added += 1
        if tile not in slot['tiles_seen']:
            slot['tiles_seen'].append(tile)
        serial = montage.modules.get(tile)
        if serial and serial not in slot['serials_seen']:
            slot['serials_seen'].append(serial)

    if not template['system']:
        template['system'] = {k: v for k, v in montage.system.items() if v not in (None, '')}
    if not template['fiducials'] and montage.fiducials:
        template['fiducials'] = {k: [float(v) for v in p] for k, p in montage.fiducials.items()}
    if added or source:
        template['updated'] = datetime.now().isoformat(timespec='seconds')
        if source and source not in template['sources']:
            template['sources'].append(source)
    return added


def from_montage(key, montage, source=""):
    """
    A reference template made only from this montage (option A): meant for
    a stream recorded with every slot of the helmet populated.
    """
    template = new_template(key)
    merge_montage(template, montage, source)
    template['reference'] = True
    return template


def empty_slot_points(template, matched):
    """
    3D optode positions of the slots no live tile occupies, as
    {'slot<id>/<optode id>': position} for core.montage.project_2d.
    """
    used = set(matched.values())
    return {f"slot{slot['id']}/{o}": np.asarray(pos, dtype=float)
            for slot in template['slots'] if slot['id'] not in used
            for o, pos in slot['optodes'].items()}


def template_path(directory, key):
    """File path of a template."""
    return os.path.join(directory, f"{key}.json")


def list_templates(directory):
    """Names of the templates saved in directory (sorted)."""
    if not os.path.isdir(directory):
        return []
    return sorted(f[:-5] for f in os.listdir(directory) if f.endswith('.json'))


def load_template(directory, key):
    """Loads a template, or returns None if it doesn't exist."""
    path = template_path(directory, key)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_template(directory, template):
    """Writes a template to <directory>/<key>.json (creating the directory) and returns the path."""
    os.makedirs(directory, exist_ok=True)
    path = template_path(directory, template['key'])
    with open(path, 'w') as f:
        json.dump(template, f, indent=1)
    return path


def _natural_key(name):
    """Sort key putting 'N2' before 'N10'."""
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', str(name))]
