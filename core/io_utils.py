# core/io_utils.py
"""
Shared CSV read/write helpers that keep stray DataFrame index columns out of
the toolbox's data files.

A CSV written with pandas' default index=True gains a leading nameless
column, which pandas reads back as 'Unnamed: 0'. Left in place it is treated
as just another signal channel / feature - e.g. a decoder ends up trained on
the row number. Every CSV the toolbox loads or saves should go through
read_csv()/write_csv() here so such columns are detected and dropped in one
place instead of being re-checked (or forgotten) at each call site.

Note: the 'index' column the offline epoching step creates is a real epoch id,
not a leaked index, and is deliberately NOT matched here.
"""
import re

import pandas as pd

# pandas' name for a column whose header cell is empty - what a saved index becomes
INDEX_COLUMN_PATTERN = re.compile(r"^Unnamed: \d+$")


def find_index_columns(columns):
    """Returns the column names that look like a leaked CSV index ('Unnamed: N')."""
    return [c for c in columns if isinstance(c, str) and INDEX_COLUMN_PATTERN.match(c)]


def drop_index_columns(df, log_callback=None, source=""):
    """
    Drops any leaked CSV index columns from `df`, reporting what was removed
    through log_callback (or print). Returns the cleaned DataFrame.
    """
    index_cols = find_index_columns(df.columns)
    if index_cols:
        where = f" from '{source}'" if source else ""
        message = f"[Index Guard] Dropped leaked CSV index column(s){where}: {index_cols}"
        if log_callback:
            log_callback(message)
        else:
            print(message)
        df = df.drop(columns=index_cols)
    return df


def read_csv(path, log_callback=None, **kwargs):
    """pd.read_csv wrapper that strips leaked index columns from the result."""
    df = pd.read_csv(path, **kwargs)
    return drop_index_columns(df, log_callback=log_callback, source=path)


def write_csv(df, path, log_callback=None, **kwargs):
    """DataFrame.to_csv wrapper that never writes the index, nor any leaked index columns."""
    kwargs["index"] = False
    df = drop_index_columns(df, log_callback=log_callback, source=path)
    df.to_csv(path, **kwargs)
