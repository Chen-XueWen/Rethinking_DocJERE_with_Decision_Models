"""Remote decision engine: hosted decision models via the LiteLLM /v1/systemone route.

Drop-in replacement for src.models.decision_engine.DecisionEngine: same
`predict` / `predict_rows` / `n_calls` / `n_questions` interface, so the
existing stage 1-3 inference code and evaluation run unchanged.

Rows are sent concurrently (they are independent within a stage), under a
shared requests-per-minute limiter. 429s and 5xx are retried with
exponential backoff, honouring Retry-After.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common.utils import ROOT, load_yaml


def load_env(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE lines from .env into os.environ (existing vars win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class RateLimiter:
    """Thread-safe pacing: at most `rpm` request starts per rolling minute."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self.lock = threading.Lock()
        self.next_at = time.monotonic()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_at)
            self.next_at = start + self.interval
        if start > now:
            time.sleep(start - now)


class RemoteDecisionEngine:
    def __init__(self, model_key: str):
        load_env()
        cfg = load_yaml("configs/pretrained.yaml")
        self.model = cfg["models"][model_key]["litellm_model"]
        rl = cfg["rate_limit"]
        self.max_concurrency = int(rl["max_concurrency"])
        self.max_retries = int(rl["max_retries"])
        self.timeout = float(rl["timeout_s"])
        self.limiter = RateLimiter(int(os.environ.get("PRETRAINED_RPM", rl["rpm"])))
        try:
            self.url = os.environ["LITELLM_API_BASE"].rstrip("/") + "/v1/systemone"
            self._key = os.environ["LITELLM_KEY"]
        except KeyError as e:
            raise RuntimeError(f"{e.args[0]} not set (expected in {ROOT / '.env'})") from None
        self.n_calls = 0      # requests (rows) for throughput reporting
        self.n_questions = 0  # decisions answered across those requests
        self.usage = {"input_tokens": 0, "output_tokens": 0, "retries": 0,
                      "waf_blocks": 0, "outages": 0}
        self._count_lock = threading.Lock()
        self.waf_cooldown_s = float(rl.get("waf_cooldown_s", 300))
        self._waf_lock = threading.Lock()
        self._waf_clear_at = 0.0
        self.max_outage_s = float(rl.get("max_outage_s", 1800))
        self._outage_since: float | None = None
        self._last_outage = 0.0

    # -- public API (mirrors DecisionEngine) ---------------------------------
    def predict(self, state: str, questions: dict) -> dict:
        """Score one row. Returns {question_id: {"answer", "probabilities"}}."""
        return self._call(state, questions)

    def predict_rows(self, rows: list[dict], desc: str = "rows") -> list[dict]:
        """Score rows concurrently; results come back in input order."""
        if not rows:
            return []
        from tqdm import tqdm
        with ThreadPoolExecutor(min(self.max_concurrency, len(rows))) as ex:
            futs = [ex.submit(self._call, r["state"], r["questions"]) for r in rows]
            return [f.result() for f in tqdm(futs, desc=desc, unit="row", leave=False)]

    # -- internals -----------------------------------------------------------
    def _outage_cooldown(self, code: int) -> None:
        """One thread waits out a load-balancer block/outage; the others queue
        behind it. Gives up after `max_outage_s` of back-to-back failures."""
        with self._waf_lock:
            now = time.monotonic()
            if now < self._waf_clear_at:
                return  # another thread already waited out this outage
            if self._outage_since is None or now - self._last_outage > self.waf_cooldown_s + 60:
                self._outage_since = now  # a new outage, not a continuation
            if now - self._outage_since > self.max_outage_s:
                raise RuntimeError(
                    f"{self.model}: proxy unavailable (HTTP {code}) for "
                    f"{(now - self._outage_since) / 60:.0f} min; giving up "
                    f"(resume the run later, finished docs are kept)")
            kind = "WAF block" if code == 403 else "proxy unavailable"
            print(f"[{self.model}] HTTP {code} ({kind}) at {time.strftime('%H:%M:%S')}: "
                  f"cooling down {self.waf_cooldown_s:.0f}s", flush=True)
            self.usage["waf_blocks" if code == 403 else "outages"] += 1
            time.sleep(self.waf_cooldown_s)
            self._last_outage = time.monotonic()
            self._waf_clear_at = self._last_outage + 5

    def _call(self, state: str, questions: dict) -> dict:
        body = json.dumps({"model": self.model, "state": state,
                           "questions": questions}).encode()
        attempt = 0
        while True:
            self.limiter.wait()
            req = urllib.request.Request(self.url, data=body, headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                detail = e.read()[:500].decode(errors="replace")
                if e.code in (403, 502, 503, 504) and "<html" in detail.lower():
                    # An HTML error page comes from the proxy's AWS load balancer,
                    # not LiteLLM (which answers in JSON): 403 = WAF burst block,
                    # 502/503/504 = LiteLLM backend down or restarting. Both clear
                    # on their own (FINDINGS.md 2026-11-18). Wait it out; this
                    # doesn't count as a retry.
                    self._outage_cooldown(e.code)
                    continue
                retryable = e.code == 429 or e.code >= 500
                if not retryable or attempt == self.max_retries:
                    raise RuntimeError(f"{self.model}: HTTP {e.code}: {detail}") from None
                delay = float(e.headers.get("Retry-After") or 0) or min(60, 2 ** attempt)
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"{self.model}: {e}") from None
                delay = min(60, 2 ** attempt)
            with self._count_lock:
                self.usage["retries"] += 1
            attempt += 1
            time.sleep(delay + random.uniform(0, 1))

        with self._count_lock:
            self.n_calls += 1
            self.n_questions += len(questions)
            u = out.get("usage", {})
            self.usage["input_tokens"] += u.get("input_tokens", 0)
            self.usage["output_tokens"] += u.get("output_tokens", 0)
        return {qid: _normalize(questions[qid], ans)
                for qid, ans in out["answers"].items()}


def _normalize(question: dict, ans: dict) -> dict:
    """Map a System One answer onto the FastDecisionModel.predict shape the
    stage code reads: {"answer": ..., "probabilities": {...}}."""
    t = ans.get("type")
    if t == "noul":
        p = float(ans["noul"])
        return {"answer": p >= 0.5, "probabilities": {"true": p, "false": 1 - p}}
    if t == "choice":
        return {"answer": ans["choice"], "probabilities": ans["probabilities"]}
    if t == "score":
        probs = ans.get("probabilities", {})
        return {"answer": ans.get("score"), "probabilities": probs}
    return {"answer": None, "probabilities": {}, "raw": ans}  # e.g. a refusal
