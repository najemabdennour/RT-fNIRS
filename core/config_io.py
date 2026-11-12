# core/config_io.py
"""
Reads and edits config.py for the Settings tab, keeping the file clean for
developers who edit it by hand.

Saving patches only the values that changed, at their exact place in the
source: comments, citations, blank lines and formatting elsewhere are left
untouched. Folder paths inside the project stay written as
os.path.join(BASE_DIR, ...). Before the file is replaced, the new source is
executed and checked to produce exactly the intended values, a backup is
written (config.py.bak), and the save is refused if the file changed on disk
since it was loaded. No Qt dependency.
"""
import ast
import hashlib
import io
import json
import math
import os
import re
import runpy
import tokenize

# Settings the Settings tab edits (top-level names in config.py).
EDITABLE = [
    'DEFAULT_EXPERIMENT', 'COADAPTATION_ACCEPTANCE_CRITERIA', 'DEFAULT_PREPROC_CONFIG',
    'DEFAULT_RAW_RECORDINGS', 'DEFAULT_RAW_BEHAVIORAL', 'DEFAULT_OUT_SESSIONS', 'DEFAULT_OUT_BEHAVIORAL',
    'SIMULATOR_LAYOUT_FILE', 'HELMET_TEMPLATE_DIR', 'NON_SIGNAL_COLUMNS', 'EXPERIMENT_PROFILES',
]
PATH_SETTINGS = ['DEFAULT_RAW_RECORDINGS', 'DEFAULT_RAW_BEHAVIORAL', 'DEFAULT_OUT_SESSIONS',
                 'DEFAULT_OUT_BEHAVIORAL', 'SIMULATOR_LAYOUT_FILE', 'HELMET_TEMPLATE_DIR']


class ConfigChangedError(RuntimeError):
    """config.py was modified on disk after it was loaded, so saving would overwrite those edits."""


class ConfigFile:
    """
    One loaded config.py: its current values, the comments documenting
    them, and save() to write changed values back in place.
    """

    def __init__(self, path):
        """Loads and executes the file at `path`."""
        self.path = os.path.abspath(path)
        self.reload()

    def reload(self):
        """Re-reads the file: its source, values (by executing it) and comments."""
        with open(self.path, 'rb') as f:
            self.source = f.read()
        self.digest = hashlib.sha256(self.source).hexdigest()
        namespace = runpy.run_path(self.path)
        self.base_dir = namespace.get('BASE_DIR', os.path.dirname(self.path))
        self.values = {name: namespace[name] for name in EDITABLE if name in namespace}
        self.comments, self.headings = _collect_comments(self.source)

    def changed_on_disk(self):
        """True if the file's content differs from what was loaded."""
        with open(self.path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest() != self.digest

    def comment_for(self, name, key=None):
        """
        The comment documenting a setting: the trailing comment on its line,
        else the comment block right above it ('' if none).
        """
        return self.comments.get((name, key), '')

    def heading_for(self, name, key):
        """
        The section heading a dict entry starts, i.e. a comment block above it
        that follows a blank line or the opening brace ('' if none).
        """
        return self.headings.get((name, key), '')

    def diff(self, new_values):
        """Names (and 'NAME[key]' for DEFAULT_PREPROC_CONFIG entries) whose value would change."""
        changes = []
        for name, value in new_values.items():
            old = self.values.get(name)
            if isinstance(old, dict) and isinstance(value, dict) and name == 'DEFAULT_PREPROC_CONFIG':
                changes += [f"{name}[{k!r}]" for k in value if not _equal(old.get(k), value[k])]
            elif not _equal(old, value):
                changes.append(name)
        return changes

    def save(self, new_values, backup=True):
        """
        Writes the changed values into the file in place.

        Args:
            new_values: {setting name: new value} (unchanged ones are ignored).
            backup: write '<config>.bak' with the previous content first.
        Returns:
            The list of changed settings (see diff); [] writes nothing.
        Raises:
            ConfigChangedError if the file changed on disk since it was loaded;
            ValueError if the patched file would not reproduce the values.
        """
        changes = self.diff(new_values)
        if not changes:
            return []
        if self.changed_on_disk():
            raise ConfigChangedError(f"{self.path} was modified since it was loaded.")

        tree = ast.parse(self.source)
        assignments = {node.targets[0].id: node for node in tree.body
                       if isinstance(node, ast.Assign) and len(node.targets) == 1
                       and isinstance(node.targets[0], ast.Name)}
        edits = []
        for name, value in new_values.items():
            if name not in assignments:
                raise ValueError(f"'{name}' is not assigned at the top level of {self.path}.")
            self._patch(assignments[name].value, self.values.get(name), value, edits, is_path=name in PATH_SETTINGS)

        new_source = _apply_edits(self.source, edits)
        self._verify(new_source, new_values)
        if backup:
            with open(self.path + '.bak', 'wb') as f:
                f.write(self.source)
        tmp = self.path + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(new_source)
        os.replace(tmp, self.path)
        self.reload()
        return changes

    # --- Patching ---

    def _patch(self, node, old, new, edits, is_path=False):
        """
        Collects the minimal source edits turning `node` (whose value is
        `old`) into `new`: dicts with the same keys and lists of the same
        length are patched entry by entry, so comments inside them survive;
        anything else that differs is rewritten as a whole.
        """
        if _equal(old, new):
            return
        if isinstance(node, ast.Dict) and isinstance(old, dict) and isinstance(new, dict) \
                and all(isinstance(k, ast.Constant) for k in node.keys) \
                and (set(old) == set(new) or self._one_entry_per_line(node)):
            for key_node, value_node in zip(node.keys, node.values):
                if key_node.value in new:
                    self._patch(value_node, old[key_node.value], new[key_node.value], edits)
                else:
                    self._delete_entry(key_node, value_node, edits)
            added = [k for k in new if k not in old]
            if added:
                self._insert_entries(node, {k: new[k] for k in added}, edits)
            return
        if isinstance(node, (ast.List, ast.Tuple)) and isinstance(old, list) and isinstance(new, list) \
                and len(old) == len(new) and not any(isinstance(v, (dict, list)) for v in new):
            for element, o, n in zip(node.elts, old, new):
                self._patch(element, o, n, edits)
            return
        indent = self._line_indent(node.lineno)
        text = self._path_expression(new) if is_path else _literal(new, indent)
        edits.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, text))

    def _lines(self):
        """The source split into lines (bytes)."""
        return self.source.split(b'\n')

    def _one_entry_per_line(self, node):
        """
        True for a multi-line dict whose entries each start on their own line
        and whose closing brace is on its own line, so entries can be
        inserted or deleted without touching the others.
        """
        lines = self._lines()
        if not node.keys or node.lineno == node.end_lineno:
            return False
        if lines[node.end_lineno - 1][:node.end_col_offset - 1].strip():
            return False
        starts = [k.lineno for k in node.keys]
        return len(set(starts)) == len(starts) and all(
            not lines[k.lineno - 1][:k.col_offset].strip() for k in node.keys)

    def _delete_entry(self, key_node, value_node, edits):
        """Deletes a dict entry's whole lines (key through value and its trailing comma/comment)."""
        rest = self._lines()[value_node.end_lineno - 1][value_node.end_col_offset:]
        if not re.fullmatch(rb'\s*,?\s*(#.*)?', rest):
            raise ValueError("Cannot remove a dict entry that shares its line with another entry.")
        edits.append((key_node.lineno, 0, value_node.end_lineno + 1, 0, ""))

    def _insert_entries(self, node, entries, edits):
        """Inserts new dict entries on their own lines before the closing brace (adding a missing comma)."""
        lines = self._lines()
        last_value = node.values[-1]
        after = lines[last_value.end_lineno - 1][last_value.end_col_offset:]
        if not after.lstrip().startswith(b','):
            edits.append((last_value.end_lineno, last_value.end_col_offset,
                          last_value.end_lineno, last_value.end_col_offset, ","))
        inner = self._line_indent(node.keys[0].lineno)
        text = "".join(f"{inner}{_literal(k, inner)}: {_literal(v, inner)},\n" for k, v in entries.items())
        edits.append((node.end_lineno, 0, node.end_lineno, 0, text))

    def _path_expression(self, value):
        """A path inside the project as os.path.join(BASE_DIR, ...); other paths and None as literals."""
        if not value:
            return "None"
        value = os.path.abspath(value)
        rel = os.path.relpath(value, self.base_dir)
        if rel == '.' or rel.startswith('..') or os.path.isabs(rel):
            return json.dumps(value)
        return "os.path.join(BASE_DIR, " + ", ".join(json.dumps(p) for p in rel.split(os.sep)) + ")"

    def _line_indent(self, lineno):
        """Leading whitespace of a source line (1-based)."""
        line = self.source.split(b'\n')[lineno - 1].decode('utf-8')
        return line[:len(line) - len(line.lstrip())]

    def _verify(self, new_source, new_values):
        """Executes the patched source and checks every edited setting has exactly the intended value."""
        ast.parse(new_source)
        namespace = {'__file__': self.path, '__name__': '_config_check'}
        exec(compile(new_source, self.path, 'exec'), namespace)
        for name, value in new_values.items():
            expected = os.path.abspath(value) if name in PATH_SETTINGS and value else value
            got = namespace.get(name)
            got = os.path.abspath(got) if name in PATH_SETTINGS and got else got
            if not _equal(got, expected if expected != '' else None):
                raise ValueError(f"Patched config.py would not reproduce '{name}' - nothing was written.")


def _equal(a, b):
    """Deep equality with float tolerance (spin boxes round floats), treating '' and None alike."""
    if a in ('', None) and b in ('', None):
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _literal(value, indent=""):
    """
    Python source for a value in config.py's style: double-quoted strings,
    short lists on one line, dicts one entry per line indented 4 spaces.
    """
    inner = indent + "    "
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        items = [_literal(v, inner) for v in value]
        one_line = "[" + ", ".join(items) + "]"
        if len(one_line) + len(indent) <= 96 and "\n" not in one_line:
            return one_line
        lines, current = [], inner
        for item in items:
            piece = item + ", "
            if len(current) + len(piece) > 96 and current.strip():
                lines.append(current.rstrip())
                current = inner
            current += piece
        lines.append(current.rstrip().rstrip(','))
        return "[\n" + "\n".join(lines) + "\n" + indent + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        one_line = "{" + ", ".join(f"{_literal(k)}: {_literal(v)}" for k, v in value.items()) + "}"
        if len(one_line) + len(indent) <= 80 and "\n" not in one_line and not any(
                isinstance(v, (dict, list)) for v in value.values()):
            return one_line
        entries = [f"{inner}{_literal(k, inner)}: {_literal(v, inner)}," for k, v in value.items()]
        return "{\n" + "\n".join(entries) + "\n" + indent + "}"
    raise TypeError(f"Unsupported setting value type: {type(value).__name__}")


def _apply_edits(source, edits):
    """Replaces (start line, start col, end line, end col) byte spans with new text, from last to first."""
    lines = source.split(b'\n')
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)
    out = source
    for l1, c1, l2, c2, text in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        start, end = offsets[l1 - 1] + c1, offsets[l2 - 1] + c2
        new = text.encode('utf-8')
        # Keep a trailing comment on the same line at its column.
        delta = len(new) - (end - start)
        if l1 == l2 and b'\n' not in new and delta:
            match = re.match(rb'(,?)( +)#', out[end:])
            if match:
                spaces = max(1, len(match.group(2)) - delta)
                cut = end + len(match.group(1))
                out = out[:cut] + b' ' * spaces + out[cut + len(match.group(2)):]
        out = out[:start] + new + out[end:]
    return out


def _collect_comments(source):
    """
    Collects the comments documenting settings.

    Returns:
        (comments, headings): comments maps (name, None) for top-level
        settings and (name, key) for entries of top-level dicts to the comment
        on the same line, else the comment lines directly above. For dict
        entries, a comment block above that follows a blank line or the
        opening brace is a section heading instead (headings[(name, key)]),
        and only a block attached to the previous line documents the entry.
    """
    text = source.decode('utf-8')
    lines = text.split('\n')
    trailing = {}
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT and lines[tok.start[0] - 1][:tok.start[1]].strip():
            trailing[tok.start[0]] = tok.string.lstrip('#').strip()

    def above(lineno):
        """(comment lines directly above a 1-based line, whether a blank line or '{' precedes them)."""
        block, i = [], lineno - 2
        while i >= 0 and lines[i].strip().startswith('#'):
            block.insert(0, lines[i].strip().lstrip('#').strip())
            i -= 1
        starts_section = i < 0 or not lines[i].strip() or lines[i].rstrip().endswith('{')
        return " ".join(block), starts_section

    comments, headings = {}, {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            comments[(name, None)] = trailing.get(node.lineno) or above(node.lineno)[0]
            if isinstance(node.value, ast.Dict):
                for key in node.value.keys:
                    if not isinstance(key, ast.Constant):
                        continue
                    block, starts_section = above(key.lineno)
                    if block and starts_section:
                        headings[(name, key.value)] = block
                        block = ''
                    note = trailing.get(key.lineno, '')
                    comments[(name, key.value)] = " ".join(p for p in (block, note) if p)
    return comments, headings


def apply_to_module(module, values):
    """
    Updates an imported config module in place with saved values: dicts and
    lists are updated in place so references held elsewhere see the change
    (session-registered experiment profiles are kept), scalars are replaced.
    """
    for name, value in values.items():
        current = getattr(module, name, None)
        if isinstance(current, dict) and isinstance(value, dict):
            current.update(value)
        elif isinstance(current, list) and isinstance(value, list):
            current[:] = value
        else:
            setattr(module, name, value)
