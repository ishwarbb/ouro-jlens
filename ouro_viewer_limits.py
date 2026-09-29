"""Viewer resource limits; generation length is independent of grid paging."""

MAX_NEW_TOKENS = 16819
DEFAULT_NEW_TOKENS = MAX_NEW_TOKENS
MAX_PROMPT_TOKENS = 4096
MAX_SEQUENCE_TOKENS = MAX_PROMPT_TOKENS + MAX_NEW_TOKENS
READOUT_WINDOW = 512
PREFILL_CHUNK = 128
READOUT_BATCH = 64


def window_bounds(start, length):
    if type(start) is not int or start < 0 or start >= length or start % READOUT_WINDOW:
        raise ValueError('Select a valid token window')
    return start, min(start + READOUT_WINDOW, length)
