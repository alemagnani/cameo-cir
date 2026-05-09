"""
Claude-as-judge relevance scorer for CAMEO CIR.

Design decisions
----------------
* Cache is model-specific: data/eval_cache/<model_name>.json
  Key: "<query_id>||<ret_product_id>" → integer score 0/1/2
  Written atomically after every new score — no work is lost on crash.

* Prompt uses 3 calibrated few-shot examples drawn from CIR-1 ground truth:
    score=2  product 1654  (Korean, vegetable-based cleanser — perfect match)
    score=1  product 13    (COSRX, Korean brand but no vegetables — partial)
    score=0  product 45    (Obagi, US brand, salicylic-acid focus — wrong)
  These anchor the 0/1/2 scale before any new pair is judged.

* Ground-truth pairs are NEVER sent to the API — they are injected directly
  into qrels as score=2 before this class is called.

Usage
-----
    from src.eval.judge import ClaudeJudge
    judge = ClaudeJudge()                      # loads cache automatically
    qrels = judge.score_run(queries, run, products_df, k=20)
"""

import base64
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "data" / "eval_cache"


# ---------------------------------------------------------------------------
# Few-shot examples (anchored to CIR-1 ground truth)
# ---------------------------------------------------------------------------

FEW_SHOT = """
Below are three calibration examples so you understand the scale.
All examples are for the same modification query.

---EXAMPLE START---
MODIFICATION QUERY: "I don't like these ones. What I really want is a set of body wash made from vegetables, and I think they're from Korea."

REFERENCE PRODUCT:
  Name: 3W CLINIC Vitamin C/Snail/Brown Tea/Coenzyme Q10 Facial Cleanser 100ml
  Brand: 3W CLINIC
  Description: A facial cleanser combining Vitamin C, Snail Mucin, Brown Tea and Coenzyme Q10. Brightening, antioxidant formula for mixed and oily skin.

CANDIDATE A (score=2 — fully relevant):
  Name: Byvibes Wonder Bath Super Vegitoks Cleanser 300ml Facial Cleanser with Vegetable Extract
  Brand: BYVIBES WONDER BATH
  Description: A Korean facial/body cleanser made from vegetable extracts (broccoli, spinach, kale). Suitable for all skin types. 300ml set format.
  → Score: 2
  → Reason: Korean brand, vegetable-based ingredients, cleanser product. All aspects of the query are satisfied.

CANDIDATE B (score=1 — partially relevant):
  Name: Cosrx Tea Tree 0.5% BHA Low pH Good Morning Gel Cleanser 150ml
  Brand: COSRX
  Description: Gentle gel cleanser from Korean brand COSRX. Contains tea tree extract and 0.5% BHA. pH 4.5-5.5. Suitable for all skin types including sensitive and acne-prone. Origin: South Korea.
  → Score: 1
  → Reason: Korean brand (partial match), cleanser product (right type), but does NOT contain vegetable-based ingredients — the core request is not satisfied.

CANDIDATE C (score=0 — not relevant):
  Name: Acne-reducing facial cleanser for oily skin Obagi Clenziderm MD Daily Care Foaming Cleanser 118ml
  Brand: OBAGI
  Description: American dermatologist brand. Contains 2% Salicylic Acid and Menthol for acne treatment. Origin: USA. No vegetable extracts. Single bottle, 118ml.
  → Score: 0
  → Reason: Not Korean, no vegetable ingredients. Modification not satisfied at all.
---EXAMPLE END---
""".strip()

RUBRIC = """
Score the CANDIDATE PRODUCT for relevance to the user's modification query on this scale:
  0 — Not relevant: wrong product type OR modification not satisfied at all.
  1 — Partially relevant: right category but modification only partially met.
  2 — Fully relevant: correct product type AND modification fully satisfied.

Reply with ONLY a single integer: 0, 1, or 2. No explanation.
""".strip()


# ---------------------------------------------------------------------------
# Judge class
# ---------------------------------------------------------------------------

class ClaudeJudge:
    def __init__(self, model: str = "haiku", use_images: bool = False):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        from src.eval.models import resolve
        self.model = resolve(model)
        self.use_images = use_images
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        # Image mode gets a separate cache — scores may differ when model sees images
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = self.model.replace("/", "_").replace(":", "_")
        suffix = "_vision" if use_images else ""
        self.cache_path = CACHE_DIR / f"{safe_name}{suffix}.json"
        self._cache: Dict[str, int] = self._load_cache()

        print(f"Judge model  : {self.model}  (images={'on' if use_images else 'off'})")
        print(f"Cache file   : {self.cache_path.relative_to(ROOT)}")
        print(f"Cached scores: {len(self._cache)}")

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> Dict[str, int]:
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        tmp = self.cache_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._cache, f, indent=2)
        tmp.replace(self.cache_path)   # atomic on POSIX

    @staticmethod
    def _cache_key(query_id: str, ret_product_id: str) -> str:
        return f"{query_id}||{ret_product_id}"

    # ------------------------------------------------------------------
    # Single-pair scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image_path: str) -> Optional[str]:
        """Load image from data/raw/<image_path> and return base64-encoded JPEG."""
        full = ROOT / "data" / "raw" / image_path
        if not full.exists():
            return None
        with open(full, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def score(
        self,
        question: str,
        ref_product: Dict,
        ret_product: Dict,
        query_id: str,
        ret_product_id: str,
        retries: int = 3,
    ) -> int:
        """
        Return relevance score 0/1/2 for a single (query, retrieved product) pair.
        Checks cache first; writes cache after a new API call.
        """
        key = self._cache_key(query_id, ret_product_id)
        if key in self._cache:
            return self._cache[key]

        content = self._build_content(question, ref_product, ret_product)

        for attempt in range(retries):
            try:
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=8,
                    messages=[{"role": "user", "content": content}],
                )
                text = msg.content[0].text.strip()
                result = int(text[0])
                if result in (0, 1, 2):
                    self._cache[key] = result
                    self._save_cache()
                    return result
            except Exception as e:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)

        return 0

    def _build_content(self, question: str, ref_product: Dict, ret_product: Dict):
        """Build the message content list, optionally including images."""
        text = self._build_prompt(question, ref_product, ret_product)

        if not self.use_images:
            return text

        # Build multimodal content: interleave images with text labels
        content = []
        ref_img = self._load_image(ref_product.get("image_path", ""))
        ret_img = self._load_image(ret_product.get("image_path", ""))

        if ref_img:
            content.append({"type": "text", "text": "REFERENCE PRODUCT IMAGE:"})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": ref_img
            }})
        if ret_img:
            content.append({"type": "text", "text": "CANDIDATE PRODUCT IMAGE:"})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": ret_img
            }})

        content.append({"type": "text", "text": text})
        return content

    def _build_prompt(self, question: str, ref_product: Dict, ret_product: Dict) -> str:
        def fmt(p: Dict) -> str:
            return (
                f"  Name: {p.get('translated_name', 'N/A')}\n"
                f"  Brand: {p.get('brand', 'N/A')}\n"
                f"  Description: {(p.get('translated_description') or 'N/A')[:400]}"
            )

        return f"""{FEW_SHOT}

---NEW QUERY TO JUDGE---
MODIFICATION QUERY: "{question}"

REFERENCE PRODUCT:
{fmt(ref_product)}

CANDIDATE PRODUCT:
{fmt(ret_product)}

{RUBRIC}"""

    # ------------------------------------------------------------------
    # Batch scoring of a full run
    # ------------------------------------------------------------------

    def score_run(
        self,
        queries: Dict,
        run: Dict,
        products_df,
        k: int = 20,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, int]]:
        """
        Score all retrieved products for all queries.

        queries      — loaded queries JSON dict (query_id → query fields)
        run          — {query_id: [product_id, ...]} ranked list
        products_df  — pandas DataFrame with product metadata, indexed by product_id
        k            — only judge top-k per query

        Ground-truth product IDs (answer_product_id) are injected as score=2
        WITHOUT calling the API.

        Returns qrels: {query_id: {product_id: score}}
        """
        prod_lookup = (
            products_df.drop_duplicates(subset="product_id")
            .set_index("product_id")
            .to_dict("index")
        )
        qrels: Dict[str, Dict[str, int]] = {}
        new_calls = 0

        for qid, query in queries.items():
            ranked = [str(pid) for pid in run.get(qid, [])[:k]]
            ref_pid = str(query["product_id"])
            ref_info = prod_lookup.get(ref_pid, {})
            question = query["question"]

            qrels[qid] = {}

            # Inject ground-truth scores directly (no API call)
            for gt_pid in query.get("answer_product_id", []):
                qrels[qid][str(gt_pid)] = 2

            for pid in ranked:
                if pid in qrels[qid]:
                    continue   # already have score (GT or previous API call)

                cached = self._cache.get(self._cache_key(qid, pid))
                if cached is not None:
                    qrels[qid][pid] = cached
                    if verbose:
                        print(f"  [{qid}] {pid} → {cached} (cached)  "
                              f"{prod_lookup.get(pid,{}).get('translated_name','?')[:55]}")
                    continue

                ret_info = prod_lookup.get(pid, {})
                score = self.score(
                    question=question,
                    ref_product=ref_info,
                    ret_product=ret_info,
                    query_id=qid,
                    ret_product_id=pid,
                )
                qrels[qid][pid] = score
                new_calls += 1
                if verbose:
                    print(f"  [{qid}] {pid} → {score} (new)     "
                          f"{ret_info.get('translated_name','?')[:55]}")

        print(f"\nJudge done. New API calls: {new_calls}  "
              f"Total cache size: {len(self._cache)}")
        return qrels
