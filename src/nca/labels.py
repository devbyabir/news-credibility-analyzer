"""Single source of truth for the dataset's label meaning.

BanFakeNews-2.0 (Mendeley Data, doi:10.17632/kjh887ct4j.1) encodes:
    0 = fake       (manually annotated fake / misleading / satirical articles)
    1 = authentic  (articles from mainstream news portals)

The legacy notebook treated 1 as "Fake"; every module must import from here.
"""

FAKE = 0
AUTHENTIC = 1

LABEL_NAMES = {FAKE: "fake", AUTHENTIC: "authentic"}
DISPLAY_NAMES = {FAKE: "Fake-like", AUTHENTIC: "Authentic-like"}
VALID_LABELS = frozenset(LABEL_NAMES)


def label_name(label: int) -> str:
    """Return the canonical name for a label, raising on unknown values."""
    try:
        return LABEL_NAMES[int(label)]
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Unknown label {label!r}; expected one of {sorted(VALID_LABELS)}") from exc
