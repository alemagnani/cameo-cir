"""
Canonical model IDs and shorthand aliases for the CAMEO eval stack.
Add new models here; everything else picks them up automatically.
"""

ALIASES = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-6",
}

# Ordered from cheapest/fastest to most capable
DEFAULT_JUDGE_MODELS = ["haiku", "sonnet", "opus"]


def resolve(name: str) -> str:
    """Return the full model ID for an alias or pass through if already a full ID."""
    return ALIASES.get(name, name)


def display_name(model_id: str) -> str:
    """Short label for tables — alias if known, else last segment of ID."""
    reverse = {v: k for k, v in ALIASES.items()}
    return reverse.get(model_id, model_id.split("-")[-1])
