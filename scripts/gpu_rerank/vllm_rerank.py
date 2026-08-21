"""
Phase B — listwise VLM reranker on GPU via vLLM (offline batched).

Drop-in open-weights replacement for the Haiku reranker. Same task: ref image +
K candidate images + modification text -> best-to-worst ordering of the K candidates.
Reads the manifest + image store built by build_manifest.py; writes a reranked run
{qid: [reordered cand_pids]} per model. Splicing with the Stage-1 tail + evaluation
happen locally (free).

Self-contained: no repo imports, so it runs on a bare GPU box after `pip install vllm`.

Usage (on the box):
  python vllm_rerank.py --manifest manifest.jsonl --images-dir images \
      --model Qwen/Qwen2.5-VL-32B-Instruct --tensor-parallel 2 \
      --out reranked_qwen32b.json
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import string
import time
from pathlib import Path

LETTERS = string.ascii_uppercase


def data_uri(path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def label(i: int) -> str:
    return LETTERS[i] if i < 26 else LETTERS[i // 26 - 1] + LETTERS[i % 26]


def build_prompt(modification: str, cand_names: list[str]) -> str:
    n = len(cand_names)
    labels = [label(i) for i in range(n)]
    listing = "\n".join(f"  ({L}) {cand_names[i]}" for i, L in enumerate(labels))
    return f"""You are reranking products for a Composed Image Retrieval task.

The user shows a REFERENCE product image and says:
  MODIFICATION: "{modification}"

You will then see {n} CANDIDATE product images, labeled ({labels[0]})..({labels[-1]}).

For each candidate, you also have its English product name (description noisy, prefer the image):
{listing}

Rank the {n} candidates from BEST match for "reference + modification" to WORST.
A candidate is a better match when its visual+textual identity matches the *reference* in
every way the modification does NOT change, AND satisfies the change the modification asks for.

Reply with ONLY the {n} labels separated by single spaces, best first, no other text.
Labels are one or two letters (A, B, ..., Z, AA, AB, ...)."""


def parse_ranking(text: str, n: int) -> list[int]:
    label_to_idx = {label(i): i for i in range(n)}
    tokens = re.findall(r"[A-Z]{1,2}", text.upper())
    seen: list[int] = []
    for t in tokens:
        if t in label_to_idx and label_to_idx[t] not in seen:
            seen.append(label_to_idx[t])
        elif len(t) == 2 and t[0] in label_to_idx and label_to_idx[t[0]] not in seen:
            seen.append(label_to_idx[t[0]])
    for i in range(n):
        if i not in seen:
            seen.append(i)
    return seen[:n]


def build_messages(job: dict, img_dir: Path):
    """Multimodal chat messages: ref image, then candidate images, then the prompt."""
    content = [{"type": "text", "text": "REFERENCE PRODUCT IMAGE:"},
               {"type": "image_url", "image_url": {"url": data_uri(img_dir / f"{job['ref_pid']}.jpg")}}]
    for i, pid in enumerate(job["cand_pids"]):
        content.append({"type": "text", "text": f"CANDIDATE ({label(i)}):"})
        content.append({"type": "image_url", "image_url": {"url": data_uri(img_dir / f"{pid}.jpg")}})
    content.append({"type": "text", "text": build_prompt(job["modification"], job["cand_names"])})
    return [{"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-images", type=int, default=21)
    ap.add_argument("--max-tokens", type=int, default=128, help="output tokens; raise for reasoning models")
    ap.add_argument("--quantization", default=None, help="vLLM quantization, e.g. fp8, awq")
    ap.add_argument("--mm-max-patches", type=int, default=0,
                    help="InternVL dynamic-tiling cap (max_dynamic_patch); >0 limits tokens/image to fit context")
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--enforce-eager", action="store_true", help="disable CUDA graphs to free memory")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N jobs")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import SamplingParams as SP  # noqa

    jobs = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    if args.limit:
        jobs = jobs[:args.limit]
    img_dir = Path(args.images_dir)
    print(f"jobs={len(jobs)}  model={args.model}  tp={args.tensor_parallel}")

    llm_kwargs = dict(model=args.model, tensor_parallel_size=args.tensor_parallel,
                      max_model_len=args.max_model_len, trust_remote_code=True,
                      limit_mm_per_prompt={"image": args.max_images},
                      gpu_memory_utilization=args.gpu_mem_util, enforce_eager=args.enforce_eager)
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    if args.mm_max_patches > 0:
        llm_kwargs["mm_processor_kwargs"] = {"max_dynamic_patch": args.mm_max_patches}
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    messages = [build_messages(j, img_dir) for j in jobs]
    t0 = time.time()
    outputs = llm.chat(messages, sp)
    dt = time.time() - t0

    reranked: dict[str, list[str]] = {}
    n_parse_fail = 0
    for job, out in zip(jobs, outputs):
        reply = out.outputs[0].text.strip()
        order = parse_ranking(reply, len(job["cand_pids"]))
        if order[:len(job["cand_pids"])] == list(range(len(job["cand_pids"]))):
            n_parse_fail += 0  # identity is a valid (if unhelpful) order; not necessarily a failure
        reranked[job["qid"]] = [job["cand_pids"][i] for i in order]

    Path(args.out).write_text(json.dumps(reranked, indent=2))
    print(f"reranked {len(reranked)} queries in {dt:.1f}s ({len(jobs)/dt:.2f} q/s) -> {args.out}")


if __name__ == "__main__":
    main()
