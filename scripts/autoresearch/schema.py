"""
Experiment config + ledger schema for autoresearch.

A config is a dict with these (optional) keys; an experiment_id is the SHA1
of the canonical-JSON form, so the same config always produces the same id
and the ledger trivially de-duplicates.

Schema (all keys optional with sane defaults):

  id:               str — human label, ignored when hashing
  notes:            str — free text, ignored when hashing
  backbone:         "ViT-L/14" | "ViT-B/32"           (default "ViT-L/14")
  prompt:           str template with $ and {caption}  (default "a photo of $ that {caption}")
  caption_source:   "translated_question" | "question" (default "translated_question")
  fuse_text:
    kind:           "none" | "name" | "brand_only" | "structured"  (default "none")
    alpha:          0.0 ≤ float ≤ 1.0                  (default 0.0)
  oti_tokens:       null | path-to-OTI-bundle          (default null)
  rerank:
    enabled:        bool                                (default false)
    model:          "haiku" | "sonnet" | "opus"         (default "sonnet")
    topk:           int                                 (default 20)
  holdout_judge:    bool — also run Sonnet/Haiku judge on official-20
                    (default false; loop sets this every Nth experiment)
  judge_model:      "haiku" | "sonnet"                  (default "haiku")
  requires:         "cpu" | "gpu"                       (default "cpu")
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

DEFAULTS: dict[str, Any] = {
    "backbone": "ViT-L/14",
    "prompt": "a photo of $ that {caption}",
    "caption_source": "translated_question",
    "fuse_text": {"kind": "none", "alpha": 0.0},
    "oti_tokens": None,
    "rerank": {"enabled": False, "model": "sonnet", "topk": 20},
    "holdout_judge": False,
    "judge_model": "sonnet",
    "judge_use_images": True,
    "requires": "cpu",
}

# Fields excluded from the hash (don't change the experiment identity).
HASH_EXCLUDED = {"id", "notes"}


def merge_defaults(cfg: dict) -> dict:
    """Return cfg with defaults filled in (deep merge for nested dicts)."""
    out = dict(DEFAULTS)
    for k, v in cfg.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            merged = dict(out[k])
            merged.update(v)
            out[k] = merged
        else:
            out[k] = v
    return out


def canonical(cfg: dict) -> dict:
    """Hash-stable view of the config (defaults filled, hash-excluded fields stripped)."""
    full = merge_defaults(cfg)
    return {k: v for k, v in full.items() if k not in HASH_EXCLUDED}


def experiment_id(cfg: dict) -> str:
    blob = json.dumps(canonical(cfg), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def short_label(cfg: dict) -> str:
    """Human-friendly one-line label."""
    c = merge_defaults(cfg)
    parts = [c["backbone"]]
    if c["fuse_text"]["kind"] != "none":
        parts.append(f"fuse={c['fuse_text']['kind']}({c['fuse_text']['alpha']})")
    if c["oti_tokens"]:
        parts.append("oti")
    if c["rerank"]["enabled"]:
        parts.append(f"rerank={c['rerank']['model']}@{c['rerank']['topk']}")
    if c["caption_source"] != "translated_question":
        parts.append(f"capsrc={c['caption_source']}")
    parts.append(f"prompt='{c['prompt'][:30]}...'" if len(c["prompt"]) > 30 else f"prompt='{c['prompt']}'")
    return " | ".join(parts)
