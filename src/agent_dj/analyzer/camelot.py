"""Camelot wheel key mapping for DJ-compatible key notation.

The Camelot wheel maps musical keys to a number+letter system where
adjacent codes are harmonically compatible (safe to mix).
Compatible transitions: same code, ±1 number (same letter), or switch letter (same number).
"""

# Standard key names → Camelot codes
KEY_TO_CAMELOT: dict[str, str] = {
    # Minor keys (A column)
    "A minor": "8A",
    "E minor": "9A",
    "B minor": "10A",
    "F# minor": "11A",
    "Db minor": "12A",
    "Ab minor": "1A",
    "Eb minor": "2A",
    "Bb minor": "3A",
    "F minor": "4A",
    "C minor": "5A",
    "G minor": "6A",
    "D minor": "7A",
    # Major keys (B column)
    "C major": "8B",
    "G major": "9B",
    "D major": "10B",
    "A major": "11B",
    "E major": "12B",
    "B major": "1B",
    "Gb major": "2B",
    "Db major": "3B",
    "Ab major": "4B",
    "Eb major": "5B",
    "Bb major": "6B",
    "F major": "7B",
}

# Reverse lookup
CAMELOT_TO_KEY: dict[str, str] = {v: k for k, v in KEY_TO_CAMELOT.items()}

# Enharmonic equivalents for essentia output normalization
ENHARMONIC: dict[str, str] = {
    "C# minor": "Db minor",
    "D# minor": "Eb minor",
    "F# minor": "F# minor",
    "G# minor": "Ab minor",
    "A# minor": "Bb minor",
    "C# major": "Db major",
    "D# major": "Eb major",
    "F# major": "Gb major",
    "G# major": "Ab major",
    "A# major": "Bb major",
}


def key_to_camelot(key: str, scale: str) -> str:
    """Convert a key name + scale to Camelot notation.

    Args:
        key: Note name (e.g., 'C', 'F#', 'Bb')
        scale: 'major' or 'minor'

    Returns:
        Camelot code (e.g., '8B' for C major)
    """
    full_key = f"{key} {scale}"
    full_key = ENHARMONIC.get(full_key, full_key)
    return KEY_TO_CAMELOT.get(full_key, "?")


def camelot_compatible(code_a: str, code_b: str) -> bool:
    """Check if two Camelot codes are harmonically compatible.

    Compatible means: same code, ±1 number (same letter), or same number (different letter).
    """
    if code_a == "?" or code_b == "?":
        return False
    if code_a == code_b:
        return True

    num_a, letter_a = int(code_a[:-1]), code_a[-1]
    num_b, letter_b = int(code_b[:-1]), code_b[-1]

    # Same letter, adjacent numbers (wrapping 1-12)
    if letter_a == letter_b:
        diff = abs(num_a - num_b)
        return diff == 1 or diff == 11  # wrap around

    # Same number, different letter (relative major/minor)
    if num_a == num_b:
        return True

    return False


def camelot_distance(code_a: str, code_b: str) -> int:
    """Calculate the 'distance' between two Camelot codes.

    0 = same key, 1 = compatible, higher = less compatible.
    """
    if code_a == "?" or code_b == "?":
        return 99
    if code_a == code_b:
        return 0

    num_a, letter_a = int(code_a[:-1]), code_a[-1]
    num_b, letter_b = int(code_b[:-1]), code_b[-1]

    # Number distance on the wheel (circular)
    num_diff = min(abs(num_a - num_b), 12 - abs(num_a - num_b))

    # Letter change adds 1 step
    letter_diff = 0 if letter_a == letter_b else 1

    return num_diff + letter_diff
