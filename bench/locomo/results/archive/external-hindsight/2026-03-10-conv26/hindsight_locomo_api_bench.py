#!/usr/bin/env python3
"""
Standalone LoCoMo runner for benchmarking a local Hindsight API server.

This does not modify Hindsight or Elephant. It:
- ingests one LoCoMo conversation into a fresh Hindsight bank
- optionally waits for consolidation to finish
- runs Hindsight reflect on all answerable questions
- judges answers with Hindsight's existing LLM judge prompt
- captures /metrics before and after the run
- emits a normalized JSON artifact with timing + token summaries
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def add_repo_paths(repo_root: Path) -> None:
    """Allow imports from a local Hindsight checkout."""
    for path in (
        repo_root / "hindsight-dev",
        repo_root / "hindsight-clients" / "python",
        repo_root / "hindsight-api",
    ):
        sys.path.insert(0, str(path))


METRIC_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_metric_labels(text: str) -> dict[str, str]:
    if not text:
        return {}
    return {m.group(1): bytes(m.group(2), "utf-8").decode("unicode_escape") for m in LABEL_RE.finditer(text)}


def parse_prometheus(text: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    parsed: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE_RE.match(line)
        if not match:
            continue
        name = match.group(1)
        labels = tuple(sorted(parse_metric_labels(match.group(2) or "").items()))
        parsed[(name, labels)] = float(match.group(3))
    return parsed


def diff_metrics(
    before: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    after: dict[tuple[str, tuple[tuple[str, str], ...]], float],
) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    out: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    keys = set(before) | set(after)
    for key in keys:
        delta = after.get(key, 0.0) - before.get(key, 0.0)
        if abs(delta) > 1e-12:
            out[key] = delta
    return out


def fetch_text(url: str, timeout: float = 15.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def fetch_json(url: str, timeout: float = 15.0) -> Any:
    return json.loads(fetch_text(url, timeout=timeout))


def model_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [model_to_dict(v) for v in value]
    if isinstance(value, dict):
        return {k: model_to_dict(v) for k, v in value.items()}
    return value


def fmt_s(seconds: float) -> str:
    seconds = int(round(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes}:{sec:02d}"


CATEGORY_NAMES = {
    1: "multi-hop",
    2: "single-hop",
    3: "temporal",
    4: "open-domain",
}


@dataclass
class RunConfig:
    repo_root: Path
    api_url: str
    metrics_url: str
    dataset_path: Path
    conversation: str
    question_workers: int
    judge_workers: int
    wait_consolidation: bool
    output_path: Path
    bank_id: str
    timeout_s: float
    max_tokens: int
    budget: str


def build_reflect_query(question: str) -> str:
    return f"""
# CONTEXT:
You have access to facts and entities from a conversation.

# INSTRUCTIONS:
1. Search thoroughly across all available memories before answering - do not stop at the first result
2. Keep searching with different queries until you have a comprehensive answer
3. Carefully analyze all provided memories
4. Pay special attention to the timestamps to determine the answer
5. If the question asks about a specific event or fact, look for direct evidence in the memories
6. If the memories contain contradictory information or multiple instances of an event, say them all
7. Always convert relative time references to specific dates, months, or years.
8. Be as specific as possible when talking about people, places, and events
9. If the answer is not explicitly stated in the memories, use logical reasoning based on the information available to answer (e.g. calculate duration of an event from different memories).

Question: {question}
""".strip()


def summarize_metric_delta(delta: dict[tuple[str, tuple[tuple[str, str], ...]], float]) -> dict[str, Any]:
    llm_totals = {
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "calls": 0.0,
        "duration_s": 0.0,
    }
    llm_by_scope: dict[str, dict[str, float]] = defaultdict(
        lambda: {"input_tokens": 0.0, "output_tokens": 0.0, "calls": 0.0, "duration_s": 0.0}
    )
    op_by_name: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0.0, "duration_s": 0.0})

    for (name, labels_tuple), value in delta.items():
        labels = dict(labels_tuple)
        normalized = name.replace(".", "_")
        scope = labels.get("scope", "unknown")
        operation = labels.get("operation", "unknown")

        if normalized.startswith("hindsight_llm_tokens_input") and normalized.endswith("_total"):
            llm_totals["input_tokens"] += value
            llm_by_scope[scope]["input_tokens"] += value
        elif normalized.startswith("hindsight_llm_tokens_output") and normalized.endswith("_total"):
            llm_totals["output_tokens"] += value
            llm_by_scope[scope]["output_tokens"] += value
        elif normalized.startswith("hindsight_llm_calls_total"):
            llm_totals["calls"] += value
            llm_by_scope[scope]["calls"] += value
        elif normalized.startswith("hindsight_llm_duration") and normalized.endswith("_sum"):
            llm_totals["duration_s"] += value
            llm_by_scope[scope]["duration_s"] += value
        elif normalized.startswith("hindsight_operation_total"):
            op_by_name[operation]["calls"] += value
        elif normalized.startswith("hindsight_operation_duration") and normalized.endswith("_sum"):
            op_by_name[operation]["duration_s"] += value

    return {
        "llm_total": {
            "input_tokens": int(round(llm_totals["input_tokens"])),
            "output_tokens": int(round(llm_totals["output_tokens"])),
            "total_tokens": int(round(llm_totals["input_tokens"] + llm_totals["output_tokens"])),
            "calls": int(round(llm_totals["calls"])),
            "duration_s": llm_totals["duration_s"],
        },
        "llm_by_scope": {
            scope: {
                "input_tokens": int(round(values["input_tokens"])),
                "output_tokens": int(round(values["output_tokens"])),
                "total_tokens": int(round(values["input_tokens"] + values["output_tokens"])),
                "calls": int(round(values["calls"])),
                "duration_s": values["duration_s"],
            }
            for scope, values in sorted(llm_by_scope.items())
        },
        "operation_by_name": {
            op: {
                "calls": int(round(values["calls"])),
                "duration_s": values["duration_s"],
            }
            for op, values in sorted(op_by_name.items())
        },
        "raw_delta": [
            {
                "name": name,
                "labels": dict(labels),
                "value": value,
            }
            for (name, labels), value in sorted(delta.items())
        ],
    }


async def wait_for_consolidation(config: RunConfig) -> dict[str, Any]:
    if not config.wait_consolidation:
        return {
            "waited": False,
            "elapsed_s": 0.0,
            "pending_samples": [],
        }

    stats_url = f"{config.api_url.rstrip('/')}/v1/default/banks/{urllib.parse.quote(config.bank_id)}/stats"
    start = time.perf_counter()
    pending_samples: list[dict[str, Any]] = []

    while True:
        stats = fetch_json(stats_url, timeout=config.timeout_s)
        pending = int(stats.get("pending_consolidation", 0))
        sample = {
            "elapsed_s": time.perf_counter() - start,
            "pending_consolidation": pending,
            "last_consolidated_at": stats.get("last_consolidated_at"),
        }
        pending_samples.append(sample)
        if pending <= 0:
            return {
                "waited": True,
                "elapsed_s": time.perf_counter() - start,
                "pending_samples": pending_samples,
                "final_stats": stats,
            }
        if time.perf_counter() - start > config.timeout_s:
            raise TimeoutError(
                f"Timed out waiting for consolidation for bank {config.bank_id} after {config.timeout_s}s"
            )
        await asyncio.sleep(2.0)


def count_sessions_and_turns(item: dict[str, Any]) -> tuple[int, int]:
    conv = item["conversation"]
    sessions = 0
    turns = 0
    for key, value in conv.items():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        if isinstance(value, list):
            sessions += 1
            turns += len(value)
    return sessions, turns


async def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone Hindsight LoCoMo API benchmark")
    parser.add_argument("--repo-root", default="/tmp/hindsight", help="Path to local Hindsight checkout")
    parser.add_argument("--api-url", default="http://localhost:8888", help="Hindsight API base URL")
    parser.add_argument("--metrics-url", default=None, help="Prometheus metrics URL (defaults to <api-url>/metrics)")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to locomo10.json (defaults to hindsight-dev benchmark dataset)",
    )
    parser.add_argument("--conversation", required=True, help="Conversation/sample id, e.g. conv-26")
    parser.add_argument("--question-workers", type=int, default=5, help="Concurrent reflect questions")
    parser.add_argument("--judge-workers", type=int, default=4, help="Concurrent LLM judge requests")
    parser.add_argument("--wait-consolidation", action="store_true", help="Poll bank stats until pending=0")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Reflect max_tokens")
    parser.add_argument("--budget", default="high", choices=["low", "mid", "high"], help="Reflect budget")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Timeout for wait-consolidation/stat fetches")
    parser.add_argument("--bank-id", default=None, help="Explicit bank id (defaults to generated unique id)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    add_repo_paths(repo_root)

    from benchmarks.common.benchmark_runner import LLMAnswerEvaluator
    from benchmarks.locomo.locomo_benchmark import LoComoDataset
    from hindsight_client import Hindsight

    metrics_url = args.metrics_url or f"{args.api_url.rstrip('/')}/metrics"
    dataset_path = (
        Path(args.dataset).resolve()
        if args.dataset
        else repo_root / "hindsight-dev" / "benchmarks" / "locomo" / "datasets" / "locomo10.json"
    )
    bank_id = args.bank_id or f"locomo-{args.conversation}-{int(time.time())}-{random.randint(1000,9999)}"
    output_path = (
        Path(args.output).resolve()
        if args.output
        else Path("/tmp") / f"hindsight-{args.conversation}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    config = RunConfig(
        repo_root=repo_root,
        api_url=args.api_url,
        metrics_url=metrics_url,
        dataset_path=dataset_path,
        conversation=args.conversation,
        question_workers=args.question_workers,
        judge_workers=args.judge_workers,
        wait_consolidation=args.wait_consolidation,
        output_path=output_path,
        bank_id=bank_id,
        timeout_s=args.timeout,
        max_tokens=args.max_tokens,
        budget=args.budget,
    )

    dataset = LoComoDataset()
    items = dataset.load(config.dataset_path)
    item = next((it for it in items if dataset.get_item_id(it) == config.conversation), None)
    if item is None:
        raise SystemExit(f"Conversation {config.conversation} not found in {config.dataset_path}")

    qa_pairs = [qa for qa in dataset.get_qa_pairs(item) if qa.get("category") != 5 and qa.get("answer")]
    sessions, turns = count_sessions_and_turns(item)
    session_items = dataset.prepare_sessions_for_ingestion(item)

    client = Hindsight(base_url=config.api_url)
    evaluator = LLMAnswerEvaluator()
    judge_semaphore = asyncio.Semaphore(config.judge_workers)
    question_semaphore = asyncio.Semaphore(config.question_workers)

    metrics_before_text = ""
    metrics_after_text = ""
    metrics_summary: dict[str, Any] | None = None
    metrics_error: str | None = None

    overall_start = time.perf_counter()
    try:
        try:
            metrics_before_text = fetch_text(config.metrics_url, timeout=10.0)
        except Exception as exc:  # noqa: BLE001
            metrics_error = f"before snapshot failed: {exc}"

        # Fresh bank
        try:
            await client.adelete_bank(config.bank_id)
        except Exception:
            pass
        await client.acreate_bank(config.bank_id)

        ingest_start = time.perf_counter()
        ingest_items = []
        for session in session_items:
            ingest_items.append(
                {
                    "content": session["content"],
                    "timestamp": session.get("event_date"),
                    "context": session.get("context"),
                    "document_id": session.get("document_id"),
                }
            )
        await client.aretain_batch(config.bank_id, ingest_items)
        ingest_elapsed = time.perf_counter() - ingest_start

        consolidation_info = await wait_for_consolidation(config)

        async def run_question(index: int, qa: dict[str, Any]) -> dict[str, Any]:
            async with question_semaphore:
                question = qa["question"]
                gold = qa["answer"]
                category = int(qa.get("category", 0))
                query = build_reflect_query(question)
                started = time.perf_counter()
                from hindsight_client_api.models.reflect_include_options import ReflectIncludeOptions
                from hindsight_client_api.models.reflect_request import ReflectRequest

                request_obj = ReflectRequest(
                    query=query,
                    budget=config.budget,
                    context=None,
                    max_tokens=config.max_tokens,
                    response_schema=None,
                    tags=None,
                    tags_match="any",
                    include=ReflectIncludeOptions(facts={}),
                )
                response = await client._memory_api.reflect(
                    config.bank_id,
                    request_obj,
                    _request_timeout=client._timeout,
                )
                elapsed = time.perf_counter() - started

                response_dict = model_to_dict(response)
                answer = response_dict.get("text", "") if isinstance(response_dict, dict) else ""
                usage = response_dict.get("usage") or {}
                based_on = response_dict.get("based_on") or {}
                based_on_counts = {
                    key: len(value) if isinstance(value, list) else 0
                    for key, value in based_on.items()
                    if isinstance(value, list)
                }

                is_correct, judge_reasoning = await evaluator.judge_answer(
                    question,
                    gold,
                    answer,
                    judge_semaphore,
                    category=None,
                )

                return {
                    "question_index": index,
                    "question": question,
                    "category": category,
                    "category_name": CATEGORY_NAMES.get(category, str(category)),
                    "gold_answer": gold,
                    "predicted_answer": answer,
                    "is_correct": bool(is_correct),
                    "judge_reasoning": judge_reasoning,
                    "latency_s": elapsed,
                    "usage": usage,
                    "based_on_counts": based_on_counts,
                }

        qa_start = time.perf_counter()
        results = await asyncio.gather(*(run_question(i, qa) for i, qa in enumerate(qa_pairs)))
        qa_elapsed = time.perf_counter() - qa_start

        try:
            metrics_after_text = fetch_text(config.metrics_url, timeout=10.0)
            if metrics_before_text:
                metrics_summary = summarize_metric_delta(
                    diff_metrics(parse_prometheus(metrics_before_text), parse_prometheus(metrics_after_text))
                )
        except Exception as exc:  # noqa: BLE001
            metrics_error = (metrics_error + "; " if metrics_error else "") + f"after snapshot failed: {exc}"

        total_elapsed = time.perf_counter() - overall_start
        results.sort(key=lambda row: row["question_index"])

        correct = sum(1 for row in results if row["is_correct"])
        total = len(results)
        category_stats: dict[str, dict[str, Any]] = {}
        for category, name in CATEGORY_NAMES.items():
            rows = [row for row in results if row["category"] == category]
            if rows:
                category_stats[name] = {
                    "correct": sum(1 for row in rows if row["is_correct"]),
                    "total": len(rows),
                    "accuracy": sum(1 for row in rows if row["is_correct"]) / len(rows),
                }

        reflect_usage_total = {
            "input_tokens": sum(int(row["usage"].get("input_tokens", 0) or 0) for row in results),
            "output_tokens": sum(int(row["usage"].get("output_tokens", 0) or 0) for row in results),
        }
        reflect_usage_total["total_tokens"] = (
            reflect_usage_total["input_tokens"] + reflect_usage_total["output_tokens"]
        )

        artifact = {
            "benchmark": "hindsight-locomo-api",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "repo_root": str(config.repo_root),
                "api_url": config.api_url,
                "metrics_url": config.metrics_url,
                "dataset_path": str(config.dataset_path),
                "conversation": config.conversation,
                "bank_id": config.bank_id,
                "question_workers": config.question_workers,
                "judge_workers": config.judge_workers,
                "wait_consolidation": config.wait_consolidation,
                "max_tokens": config.max_tokens,
                "budget": config.budget,
            },
            "scope": {
                "sessions": sessions,
                "turns": turns,
                "questions": total,
            },
            "timing": {
                "total_s": total_elapsed,
                "ingest_s": ingest_elapsed,
                "consolidation_wait_s": consolidation_info["elapsed_s"],
                "qa_s": qa_elapsed,
                "qa_avg_s": (qa_elapsed / total) if total else 0.0,
            },
            "accuracy": {
                "correct": correct,
                "total": total,
                "accuracy": (correct / total) if total else 0.0,
                "by_category": category_stats,
            },
            "reflect_usage_from_responses": reflect_usage_total,
            "metrics": {
                "available": bool(metrics_summary),
                "error": metrics_error,
                "summary": metrics_summary,
            },
            "consolidation": consolidation_info,
            "questions": results,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2))
        if metrics_before_text:
            output_path.with_suffix(".before.prom").write_text(metrics_before_text)
        if metrics_after_text:
            output_path.with_suffix(".after.prom").write_text(metrics_after_text)

        print(f"Hindsight LoCoMo API benchmark complete")
        print(f"Conversation: {config.conversation}")
        print(f"Bank: {config.bank_id}")
        print(f"Accuracy: {correct}/{total} ({artifact['accuracy']['accuracy']*100:.1f}%)")
        print(f"Timing: total {fmt_s(total_elapsed)} | ingest {fmt_s(ingest_elapsed)} | consolidation {fmt_s(consolidation_info['elapsed_s'])} | qa {fmt_s(qa_elapsed)}")
        print(
            f"Reflect usage: {reflect_usage_total['input_tokens']} input + {reflect_usage_total['output_tokens']} output = {reflect_usage_total['total_tokens']} total"
        )
        if metrics_summary:
            llm_total = metrics_summary["llm_total"]
            print(
                f"Server metrics delta: {llm_total['input_tokens']} input + {llm_total['output_tokens']} output = {llm_total['total_tokens']} total across {llm_total['calls']} calls"
            )
        elif metrics_error:
            print(f"Server metrics unavailable: {metrics_error}")
        print(f"Saved: {output_path}")
        return 0
    finally:
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
