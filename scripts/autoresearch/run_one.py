"""
Run a single autoresearch experiment end-to-end.

Pipeline (each step skipped if its output already exists or the config doesn't ask for it):

  1. ensure CLIP corpus index for the config's backbone (image + text caches)
  2. run inference: scripts/run_searle.py → data/runs/<exp_id>_dev.json
  3. (optional) rerank: scripts/rerank_sonnet.py → data/runs/<exp_id>_dev.rerank.json
  4. eval dev (in-process; per-rule + overall NDCG/Recall/MAP)
  5. (if holdout_judge) run official-20 + Haiku judge → judge metrics
  6. append result row to scripts/autoresearch/experiments.jsonl

The ledger key is sha1(canonical(config))[:12]; if already in ledger, this is a no-op.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.autoresearch.ledger import LEDGER_PATH_DEFAULT, append_row, completed_ids
from scripts.autoresearch.metrics_helper import per_rule_metrics
from scripts.autoresearch.schema import experiment_id, merge_defaults, short_label


def validate_prompt(prompt: str) -> str | None:
    """Tokenize the rendered prompt and verify CLIP token 259 ($) survives. Return error str or None."""
    try:
        import clip
        rendered = prompt.replace("{caption}", "a sample modification text")
        tok = clip.tokenize([rendered], truncate=True)[0]
        if (tok == 259).sum().item() == 0:
            return f"prompt loses the $ token after CLIP tokenization: {prompt!r}"
    except Exception as e:
        return f"prompt validation failed: {type(e).__name__}: {e}"
    return None

DEV_FULL = ROOT / "data" / "queries_dev_full.json"
DEV_BASE = ROOT / "data" / "queries_dev.json"
DEV_QUERIES = DEV_FULL if DEV_FULL.exists() else DEV_BASE
OFFICIAL_QUERIES = ROOT / "data" / "queries_official_cir.json"
RUNS_DIR = ROOT / "data" / "runs"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=8", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _run(cmd: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    if env_extra:
        env.update(env_extra)
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, env=env, text=True)


def run_inference(cfg: dict, exp_id: str, queries_path: Path, out_path: Path) -> None:
    if out_path.exists():
        print(f"  [skip] inference cache hit → {out_path.name}")
        return
    kind = cfg["fuse_text"]["kind"]
    alpha = cfg["fuse_text"]["alpha"] if kind != "none" else 0.0
    fuse_source = kind if kind in ("brand", "brand_name") else "name"
    cmd = [
        sys.executable, "-u", str(ROOT / "scripts" / "run_searle.py"),
        "--queries", str(queries_path),
        "--out", str(out_path),
        "--backbone", cfg["backbone"],
        "--batch-size", "8",
        "--fuse-text", str(alpha),
        "--fuse-text-source", fuse_source,
        "--caption-field", cfg.get("caption_source", "translated_question"),
        "--prompt", cfg["prompt"],
    ]
    if cfg.get("oti_tokens"):
        cmd += ["--oti-tokens", cfg["oti_tokens"]]
    _run(cmd)


def run_rerank(cfg: dict, in_run: Path, queries_path: Path, out_path: Path) -> None:
    if out_path.exists():
        print(f"  [skip] rerank cache hit → {out_path.name}")
        return
    cmd = [
        sys.executable, "-u", str(ROOT / "scripts" / "rerank_sonnet.py"),
        "--run", str(in_run),
        "--queries", str(queries_path),
        "--out", str(out_path),
        "--topk", str(cfg["rerank"]["topk"]),
        "--model", cfg["rerank"]["model"],
    ]
    _run(cmd)


def run_judge_eval(run_path: Path, judge_model: str, use_images: bool = True) -> dict:
    cmd = [
        sys.executable, "-u", str(ROOT / "scripts" / "evaluate.py"),
        "--run", str(run_path),
        "--queries", str(OFFICIAL_QUERIES),
        "--use-judge",
        "--judge-model", judge_model,
        "--judge-k", "20",
        "--k", "5", "10", "20",
    ]
    if use_images:
        cmd.append("--use-images")
    _run(cmd)
    eval_path = run_path.with_suffix(".eval.json")
    return json.loads(eval_path.read_text())


def run_one(cfg: dict, ledger_path: Path = LEDGER_PATH_DEFAULT) -> dict:
    cfg = merge_defaults(cfg)
    exp_id = experiment_id(cfg)
    label = short_label(cfg)
    print(f"\n=== experiment {exp_id}  {label} ===", flush=True)

    if exp_id in completed_ids(ledger_path):
        print(f"  [skip] already in ledger", flush=True)
        return {"id": exp_id, "status": "skip"}

    started = datetime.now(timezone.utc)
    t0 = time.time()
    row: dict = {
        "id": exp_id,
        "label": label,
        "config": cfg,
        "metrics": {"dev": None, "judge": None},
        "cost_usd": 0.0,            # rough; loop computes from rerank/judge calls
        "wall_seconds": 0.0,
        "git_sha": git_sha(),
        "started_at": started.isoformat(),
        "finished_at": None,
        "status": "ok",
        "error": None,
    }

    try:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        dev_run = RUNS_DIR / f"ar_{exp_id}_dev.json"

        # Note: we intentionally use the cfg-as-is. Brand-only / structured fusion
        # would need run_searle.py to learn new flags; for v0 only "name" fusion
        # uses --fuse-text. Other fusion kinds raise NotImplemented.
        kind = cfg["fuse_text"]["kind"]
        if kind not in ("none", "name", "brand", "brand_name"):
            raise NotImplementedError(f"fuse_text.kind={kind!r} not wired yet")

        err = validate_prompt(cfg["prompt"])
        if err:
            raise ValueError(err)

        run_inference(cfg, exp_id, DEV_QUERIES, dev_run)

        if cfg["rerank"]["enabled"]:
            dev_rerank = RUNS_DIR / f"ar_{exp_id}_dev.rerank.json"
            run_rerank(cfg, dev_run, DEV_QUERIES, dev_rerank)
            dev_eval_input = dev_rerank
        else:
            dev_eval_input = dev_run

        # Eval dev (in-process; deterministic; free)
        queries = json.loads(DEV_QUERIES.read_text())
        run = json.loads(dev_eval_input.read_text())
        row["metrics"]["dev"] = per_rule_metrics(run, queries)
        print(f"  dev overall NDCG@10 = {row['metrics']['dev']['overall']['ndcg@10']:.4f}  "
              f"MAP = {row['metrics']['dev']['overall']['map']:.4f}")

        # Holdout judge (rare; costs API)
        if cfg["holdout_judge"]:
            off_run = RUNS_DIR / f"ar_{exp_id}_official.json"
            run_inference(cfg, exp_id, OFFICIAL_QUERIES, off_run)
            if cfg["rerank"]["enabled"]:
                off_rerank = RUNS_DIR / f"ar_{exp_id}_official.rerank.json"
                run_rerank(cfg, off_run, OFFICIAL_QUERIES, off_rerank)
                judge_input = off_rerank
            else:
                judge_input = off_run
            row["metrics"]["judge"] = run_judge_eval(judge_input, cfg["judge_model"],
                                                     use_images=cfg.get("judge_use_images", True))
            print(f"  judge NDCG@10 = {row['metrics']['judge']['ndcg@10']:.4f}")

    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
        print(f"  [error] {row['error']}", flush=True)

    row["wall_seconds"] = round(time.time() - t0, 1)
    row["finished_at"] = datetime.now(timezone.utc).isoformat()
    append_row(row, ledger_path)
    print(f"  [done] {row['wall_seconds']:.0f}s  status={row['status']}", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML or JSON file with the experiment config")
    args = parser.parse_args()

    raw = Path(args.config).read_text()
    cfg = yaml.safe_load(raw)
    run_one(cfg)


if __name__ == "__main__":
    main()
