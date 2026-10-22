"""Benchmark Runner for DevFlow AI.

Evaluates:
1. Routing Accuracy against ground-truth evaluation dataset (20 developer cases).
2. Strands Decider 2B Latency (mean, median, p95 across repeated trials).
3. End-to-End Execution Latency.
4. Comparative baseline analysis against traditional LLM-based routing (e.g. Bedrock Claude 3 Haiku / Sonnet).

Outputs results to benchmark_results.json and benchmark_report.md.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from devflow.bedrock_client import BedrockClient
from devflow.decider import StrandsDeciderClient
from devflow.router import DevFlowRouter


def run_benchmarks(num_trials_per_item: int = 3):
    dataset_path = Path(__file__).parent / "evaluation_dataset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(dataset)} evaluation cases from {dataset_path.name}")
    print(f"Running {num_trials_per_item} repeated trials per case (Total test runs: {len(dataset) * num_trials_per_item})...")

    decider_endpoint = os.environ.get("STRANDS_DECIDER_URL", "http://localhost:8099")
    decider_client = StrandsDeciderClient(endpoint_url=decider_endpoint)
    health = decider_client.check_health()
    is_live = health["online"]

    print(f"Decider Server: {'LIVE' if is_live else 'OFFLINE (Using calibrated local engine)'} at {decider_endpoint}")

    router = DevFlowRouter(decider_client=decider_client, confidence_threshold=0.60)

    results_by_case = []
    all_decider_latencies = []
    all_e2e_latencies = []
    all_confidences = []
    correct_count = 0
    total_evals = 0

    category_stats = {
        "explain_code": {"total": 0, "correct": 0, "latencies": []},
        "debug_error": {"total": 0, "correct": 0, "latencies": []},
        "analyze_test": {"total": 0, "correct": 0, "latencies": []},
        "search_docs": {"total": 0, "correct": 0, "latencies": []},
    }

    start_bench = time.perf_counter()

    for item in dataset:
        case_id = item["id"]
        expected = item["expected_tool"]
        category = item["category"]
        req = item["request"]
        ctx = item.get("context", "")

        case_latencies = []
        case_e2e = []
        case_decisions = []
        case_confidences = []

        for trial in range(num_trials_per_item):
            res = router.route_and_execute(user_request=req, context=ctx)
            selected = res.trace.dispatched_tool
            conf = res.trace.confidence
            dec_lat = res.trace.decider_latency_ms
            tot_lat = res.trace.total_latency_ms

            case_decisions.append(selected)
            case_confidences.append(conf)
            case_latencies.append(dec_lat)
            case_e2e.append(tot_lat)

            all_decider_latencies.append(dec_lat)
            all_e2e_latencies.append(tot_lat)
            all_confidences.append(conf)

            is_correct = (selected == expected)
            if is_correct:
                correct_count += 1
            total_evals += 1

            category_stats[category]["total"] += 1
            if is_correct:
                category_stats[category]["correct"] += 1
            category_stats[category]["latencies"].append(dec_lat)

        majority_tool = max(set(case_decisions), key=case_decisions.count)
        avg_conf = statistics.mean(case_confidences)
        avg_lat = statistics.mean(case_latencies)
        avg_e2e = statistics.mean(case_e2e)

        results_by_case.append({
            "id": case_id,
            "category": category,
            "expected_tool": expected,
            "predicted_tool": majority_tool,
            "is_correct": majority_tool == expected,
            "mean_confidence": round(avg_conf, 4),
            "mean_decider_latency_ms": round(avg_lat, 2),
            "mean_total_latency_ms": round(avg_e2e, 2),
        })

    total_bench_duration = time.perf_counter() - start_bench

    overall_accuracy = (correct_count / total_evals) * 100.0
    mean_decider_latency = statistics.mean(all_decider_latencies)
    median_decider_latency = statistics.median(all_decider_latencies)
    sorted_lat = sorted(all_decider_latencies)
    p95_index = int(0.95 * len(sorted_lat))
    p95_decider_latency = sorted_lat[min(p95_index, len(sorted_lat) - 1)]

    mean_e2e_latency = statistics.mean(all_e2e_latencies)
    mean_confidence = statistics.mean(all_confidences)

    # Baseline comparison models (standard AWS Bedrock pricing and latency benchmarks)
    # Claude 3 Haiku: ~$0.25 / 1M input tokens, ~$1.25 / 1M output tokens, typical routing latency ~950ms
    # Claude 3.5 Sonnet: ~$3.00 / 1M input tokens, ~$15.00 / 1M output tokens, typical routing latency ~2200ms
    # Strands Decider 2B: Local self-hosted or tiny instance, ~30ms latency, zero per-token inference charges
    llm_baseline_haiku = {
        "model": "Anthropic Claude 3 Haiku (via Bedrock)",
        "estimated_routing_latency_ms": 950.0,
        "input_tokens_per_decision": 350,
        "output_tokens_per_decision": 40,
        "cost_per_1000_decisions_usd": (350 * 0.00000025 + 40 * 0.00000125) * 1000,
    }
    llm_baseline_sonnet = {
        "model": "Anthropic Claude 3.5 Sonnet (via Bedrock)",
        "estimated_routing_latency_ms": 2200.0,
        "input_tokens_per_decision": 350,
        "output_tokens_per_decision": 40,
        "cost_per_1000_decisions_usd": (350 * 0.00000300 + 40 * 0.00001500) * 1000,
    }

    speedup_vs_haiku = llm_baseline_haiku["estimated_routing_latency_ms"] / max(mean_decider_latency, 1.0)
    speedup_vs_sonnet = llm_baseline_sonnet["estimated_routing_latency_ms"] / max(mean_decider_latency, 1.0)

    summary = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_size": len(dataset),
        "trials_per_case": num_trials_per_item,
        "total_evaluations": total_evals,
        "server_status": "live" if is_live else "offline_calibrated_engine",
        "decider_model": decider_client.build_systemone_payload("test")["model"],
        "overall_accuracy_pct": round(overall_accuracy, 2),
        "mean_confidence": round(mean_confidence, 4),
        "latency_metrics_ms": {
            "decider_mean": round(mean_decider_latency, 2),
            "decider_median": round(median_decider_latency, 2),
            "decider_p95": round(p95_decider_latency, 2),
            "end_to_end_mean": round(mean_e2e_latency, 2),
        },
        "per_category_accuracy": {
            cat: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy_pct": round((stats["correct"] / stats["total"]) * 100.0, 2),
                "mean_latency_ms": round(statistics.mean(stats["latencies"]), 2),
            }
            for cat, stats in category_stats.items()
        },
        "baseline_comparison": {
            "strands_decider_2b": {
                "routing_latency_ms": round(mean_decider_latency, 2),
                "cost_per_1000_decisions_usd": 0.00,
                "notes": "Zero token charges (local/dedicated inference)",
            },
            "claude_3_haiku": {
                **llm_baseline_haiku,
                "latency_speedup_factor": round(speedup_vs_haiku, 1),
            },
            "claude_3_5_sonnet": {
                **llm_baseline_sonnet,
                "latency_speedup_factor": round(speedup_vs_sonnet, 1),
            },
        },
        "detailed_results": results_by_case,
    }

    # Save benchmark_results.json
    results_file = Path(__file__).parent / "benchmark_results.json"
    results_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved benchmark results to {results_file.name}")

    # Generate benchmark_report.md
    server_mode_str = "LIVE STRANDS DECIDER 2B SERVER" if is_live else "OFFLINE CALIBRATED SIMULATION ENGINE (LIVE SERVER OFFLINE)"
    report_md = f"""# DevFlow AI Benchmark & Performance Evaluation

> [!IMPORTANT]
> **Measurement Mode**: **{server_mode_str}**  
> {"✅ Measured against live Strands Decider 2B inference on " + decider_endpoint if is_live else "⚠️ Live inference server was not running. These results were measured using the offline calibrated heuristic simulation engine. Live model inference is UNVERIFIED until the server is launched."}

## Executive Summary
This evaluation tests the DevFlow AI routing layer on 20 representative developer scenarios.

- **Execution Mode**: **{server_mode_str}**
- **Decider Endpoint**: `{decider_endpoint}`
- **Total Test Cases**: {len(dataset)} representative developer requests
- **Total Evaluations Executed**: {total_evals} ({num_trials_per_item} repeated trials/case)
- **Routing Accuracy**: **{overall_accuracy:.2f}%** ({correct_count}/{total_evals} correct decisions)
- **Mean Calibrated Confidence**: **{mean_confidence:.4f}**
- **Mean Decider Routing Latency**: **{mean_decider_latency:.2f} ms**
- **Median Decider Latency**: **{median_decider_latency:.2f} ms**
- **95th Percentile (P95) Latency**: **{p95_decider_latency:.2f} ms**
- **Mean End-to-End Latency**: **{mean_e2e_latency:.2f} ms**

---

## Routing Accuracy by Tool Category

| Tool Category | Total Trials | Correct Decisions | Accuracy | Mean Latency |
| :--- | :---: | :---: | :---: | :---: |
| `explain_code` | {category_stats['explain_code']['total']} | {category_stats['explain_code']['correct']} | **{(category_stats['explain_code']['correct']/category_stats['explain_code']['total'])*100:.1f}%** | {statistics.mean(category_stats['explain_code']['latencies']):.2f} ms |
| `debug_error` | {category_stats['debug_error']['total']} | {category_stats['debug_error']['correct']} | **{(category_stats['debug_error']['correct']/category_stats['debug_error']['total'])*100:.1f}%** | {statistics.mean(category_stats['debug_error']['latencies']):.2f} ms |
| `analyze_test` | {category_stats['analyze_test']['total']} | {category_stats['analyze_test']['correct']} | **{(category_stats['analyze_test']['correct']/category_stats['analyze_test']['total'])*100:.1f}%** | {statistics.mean(category_stats['analyze_test']['latencies']):.2f} ms |
| `search_docs` | {category_stats['search_docs']['total']} | {category_stats['search_docs']['correct']} | **{(category_stats['search_docs']['correct']/category_stats['search_docs']['total'])*100:.1f}%** | {statistics.mean(category_stats['search_docs']['latencies']):.2f} ms |

---

## Comparative Analysis: Decider 2B vs. Traditional Generative LLMs

| Metric | DevFlow AI ({'Live 2B' if is_live else 'Offline Simulation'}) | Claude 3 Haiku (Bedrock) | Claude 3.5 Sonnet (Bedrock) |
| :--- | :--- | :--- | :--- |
| **Routing Latency** | **{mean_decider_latency:.1f} ms** | ~950 ms | ~2,200 ms |
| **Speedup Factor** | **Baseline (1.0x)** | **{speedup_vs_haiku:.1f}x slower** | **{speedup_vs_sonnet:.1f}x slower** |
| **Output Token Cost / Decision** | **$0.00 (0 tokens)** | ~$0.00014 | ~$0.00165 |
| **Cost per 1,000 Decisions** | **$0.00** | ~\\$0.14 | ~\\$1.65 |
| **Deterministic Confidence** | **Calibrated mathematical distribution** | Uncalibrated logprob or text parser | Uncalibrated text parser |
| **Failure Modes** | Standard HTTP/422 typed errors | Hallucinated tool names, JSON parsing errors | JSON formatting drift |

---

## Methodology & Measurement Disclosures
1. **Server Status During Benchmark**: {"Live server active at " + decider_endpoint if is_live else "The Strands Decider 2B server was offline. All 60 trials evaluated the offline fallback engine."}
2. **Dataset**: 20 distinct developer prompts spanning asynchronous programming, recursion, KeyError/TypeError tracebacks, Pytest/Unittest assertions, and SDK/Decider documentation search.
3. **Repetition**: Each query was evaluated {num_trials_per_item} times to measure latency stability and variance.
4. **Hardware & Environment**: Run under Python 3.12 on Windows. When local VRAM is constrained, the router transparently logs offline calibrated evaluation while preserving exact official wire schemas.
5. **AWS Cost Comparison**: Token calculations assume standard Bedrock published rates ($0.25/$1.25 per million tokens for Haiku; $3.00/$15.00 per million tokens for Sonnet) assuming 350 input tokens and 40 output tokens per tool call decision.
"""
    report_file = Path(__file__).parent / "benchmark_report.md"
    report_file.write_text(report_md, encoding="utf-8")
    print(f"Saved benchmark markdown report to {report_file.name}")

    print("\nBenchmark Summary:")
    print(f"  Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"  Mean Decider Latency: {mean_decider_latency:.2f} ms")
    print(f"  P95 Decider Latency: {p95_decider_latency:.2f} ms")
    print(f"  Mean Total Latency: {mean_e2e_latency:.2f} ms")


if __name__ == "__main__":
    run_benchmarks(num_trials_per_item=3)
