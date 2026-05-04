"""
RAG evaluation utilities for the classroom ChromaDB demo.

This module implements a lightweight version of the evaluation framework
described in `RAG_Evaluation_Framework.pdf`:
1. retrieval quality
2. response quality
3. system performance
4. business-oriented outcomes
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
import time
from typing import Any, Dict, List, Sequence

from demo_output_utils import print_banner, print_key_value, print_section
from llm_utils import ask_llm, load_huggingface_llm, load_llm
from rag_chromadb_demo import build_rag_prompt, build_vector_store, retrieve_top_k_records


EVAL_DATASET_PATH = Path("sample_data/rag_evaluation_dataset.json")
EVAL_REPORT_PATH = Path("rag_evaluation_report.json")
TOP_K_RETRIEVAL = 5
TOP_K_GENERATION = 3
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "to",
    "what",
    "when",
    "which",
    "with",
}


@dataclass
class EvaluationCase:
    """One golden evaluation record used for RAG regression checks."""

    case_id: str
    query: str
    expected_answer: str
    expected_keywords: List[str]
    gold_sources: List[str]
    difficulty: str
    query_type: str
    should_answer: bool


def load_evaluation_cases() -> List[EvaluationCase]:
    """
    Load the golden evaluation dataset from disk.

    Parameters:
    - None.

    Returns:
    - List[EvaluationCase]: The ordered golden test cases.
    """
    raw_cases = json.loads(EVAL_DATASET_PATH.read_text(encoding="utf-8"))
    return [
        EvaluationCase(
            case_id=case["case_id"],
            query=case["query"],
            expected_answer=case["expected_answer"],
            expected_keywords=case["expected_keywords"],
            gold_sources=case["gold_sources"],
            difficulty=case["difficulty"],
            query_type=case["query_type"],
            should_answer=case["should_answer"],
        )
        for case in raw_cases
    ]


def normalize_tokens(text: str) -> List[str]:
    """
    Convert free text into lowercase keyword tokens.

    Parameters:
    - text: The raw text to normalize.

    Returns:
    - List[str]: Filtered tokens used by heuristic evaluators.
    """
    cleaned = []
    for token in "".join(
        character.lower() if character.isalnum() else " "
        for character in text
    ).split():
        if token not in STOPWORDS and len(token) > 2:
            cleaned.append(token)
    return cleaned


def safe_ratio(numerator: float, denominator: float) -> float:
    """
    Divide safely and return zero when the denominator is missing.

    Parameters:
    - numerator: The top value.
    - denominator: The bottom value.

    Returns:
    - float: The ratio or `0.0`.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_retrieval_metrics(
    retrieved_sources: Sequence[str],
    gold_sources: Sequence[str],
    top_k: int,
) -> Dict[str, float]:
    """
    Compute retrieval metrics over one query result list.

    Parameters:
    - retrieved_sources: Ranked retrieved source labels.
    - gold_sources: Expected relevant source labels.
    - top_k: Retrieval depth used for this evaluation.

    Returns:
    - Dict[str, float]: Retrieval scores such as hit rate and MRR.
    """
    retrieved_top_k = list(retrieved_sources[:top_k])
    gold_set = set(gold_sources)
    relevant_positions: List[int] = []
    matched_sources = set()
    for index, source in enumerate(retrieved_top_k, start=1):
        if source in gold_set and source not in matched_sources:
            relevant_positions.append(index)
            matched_sources.add(source)

    relevant_count = len(matched_sources)

    dcg = 0.0
    for position in relevant_positions:
        dcg += 1.0 / log2(position + 1)

    ideal_hits = min(len(gold_set), len(retrieved_top_k))
    idcg = sum(1.0 / log2(position + 1) for position in range(1, ideal_hits + 1))

    return {
        "hit_at_3": 1.0 if any(source in gold_set for source in retrieved_sources[:3]) else 0.0,
        "precision_at_5": safe_ratio(relevant_count, len(retrieved_top_k)),
        "recall_at_5": safe_ratio(relevant_count, len(gold_set)),
        "mrr": 1.0 / relevant_positions[0] if relevant_positions else 0.0,
        "ndcg_at_5": safe_ratio(dcg, idcg),
    }


def log2(value: float) -> float:
    """
    Return base-2 logarithm without importing the full math namespace.

    Parameters:
    - value: The numeric input.

    Returns:
    - float: Base-2 logarithm value.
    """
    import math

    return math.log2(value)


def evaluate_format_compliance(answer: str, should_answer: bool) -> float:
    """
    Check whether the answer follows the classroom response style.

    Parameters:
    - answer: The generated answer.
    - should_answer: Whether the case expects a substantive answer.

    Returns:
    - float: A compliance score from 0.0 to 1.0.
    """
    lowered_answer = answer.lower()
    if not should_answer:
        return 1.0 if "could not find" in lowered_answer else 0.0

    if not answer.strip():
        return 0.0

    bullet_lines = [
        line
        for line in answer.splitlines()
        if line.strip().startswith(("-", "*")) or line.strip()[:2].isdigit()
    ]
    if bullet_lines:
        return 1.0

    return 0.75 if len(answer.split()) >= 20 else 0.5


def evaluate_answer_quality(
    case: EvaluationCase,
    answer: str,
    retrieved_context: str,
) -> Dict[str, float]:
    """
    Compute heuristic answer-quality metrics against the golden case.

    Parameters:
    - case: The evaluation case definition.
    - answer: The generated answer text.
    - retrieved_context: The concatenated retrieved evidence.

    Returns:
    - Dict[str, float]: Faithfulness, correctness, relevancy, completeness,
      and format-compliance scores.
    """
    lowered_answer = answer.lower()
    context_tokens = set(normalize_tokens(retrieved_context))
    answer_tokens = set(normalize_tokens(answer))
    query_tokens = set(normalize_tokens(case.query))
    expected_tokens = set(normalize_tokens(" ".join(case.expected_keywords)))

    if not case.should_answer:
        no_answer_score = 1.0 if "could not find it in the retrieved documents" in lowered_answer else 0.0
        return {
            "faithfulness": no_answer_score,
            "correctness": no_answer_score,
            "relevancy": 1.0 if "could not find" in lowered_answer else 0.0,
            "completeness": no_answer_score,
            "format_compliance": evaluate_format_compliance(answer, should_answer=False),
        }

    keyword_hits = sum(1 for keyword in case.expected_keywords if keyword.lower() in lowered_answer)
    faithfulness = safe_ratio(len(answer_tokens & context_tokens), len(answer_tokens))
    correctness = safe_ratio(keyword_hits, len(case.expected_keywords))
    relevancy = safe_ratio(len(answer_tokens & query_tokens), len(query_tokens))
    completeness = safe_ratio(len(answer_tokens & expected_tokens), len(expected_tokens))

    return {
        "faithfulness": faithfulness,
        "correctness": correctness,
        "relevancy": relevancy,
        "completeness": completeness,
        "format_compliance": evaluate_format_compliance(answer, should_answer=True),
    }


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """
    Compute a simple percentile over a numeric list.

    Parameters:
    - values: The input numeric sequence.
    - percentile_value: The percentile in the range 0 to 100.

    Returns:
    - float: The selected percentile value.
    """
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    index = round((percentile_value / 100) * (len(ordered) - 1))
    return ordered[index]


def average_metric(case_results: Sequence[Dict[str, Any]], metric_name: str) -> float:
    """
    Average a metric across all case results that contain it.

    Parameters:
    - case_results: Per-case evaluation payloads.
    - metric_name: The metric key to average.

    Returns:
    - float: The arithmetic mean.
    """
    values = [
        float(case_result[metric_name])
        for case_result in case_results
        if metric_name in case_result
    ]
    return safe_ratio(sum(values), len(values))


def summarize_results(case_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate retrieval, generation, ops, and business metrics.

    Parameters:
    - case_results: The completed per-case evaluation records.

    Returns:
    - Dict[str, Any]: The overall scorecard summary.
    """
    total_latency_values = [result["ops"]["total_latency_ms"] for result in case_results]
    failure_count = sum(1 for result in case_results if result["ops"]["failed"])
    zero_result_count = sum(1 for result in case_results if result["business"]["zero_result"])
    resolution_count = sum(1 for result in case_results if result["business"]["resolved"])
    escalation_count = sum(1 for result in case_results if result["business"]["escalated"])

    return {
        "retrieval": {
            "hit_at_3": average_metric(
                [result["retrieval"] for result in case_results],
                "hit_at_3",
            ),
            "precision_at_5": average_metric(
                [result["retrieval"] for result in case_results],
                "precision_at_5",
            ),
            "recall_at_5": average_metric(
                [result["retrieval"] for result in case_results],
                "recall_at_5",
            ),
            "mrr": average_metric([result["retrieval"] for result in case_results], "mrr"),
            "ndcg_at_5": average_metric(
                [result["retrieval"] for result in case_results],
                "ndcg_at_5",
            ),
        },
        "generation": {
            "faithfulness": average_metric(
                [result["generation"] for result in case_results],
                "faithfulness",
            ),
            "correctness": average_metric(
                [result["generation"] for result in case_results],
                "correctness",
            ),
            "relevancy": average_metric(
                [result["generation"] for result in case_results],
                "relevancy",
            ),
            "completeness": average_metric(
                [result["generation"] for result in case_results],
                "completeness",
            ),
            "format_compliance": average_metric(
                [result["generation"] for result in case_results],
                "format_compliance",
            ),
        },
        "ops": {
            "latency_p50_ms": median(total_latency_values) if total_latency_values else 0.0,
            "latency_p95_ms": percentile(total_latency_values, 95),
            "failure_rate": safe_ratio(failure_count, len(case_results)),
            "timeout_rate": 0.0,
            "cost_per_query": "not_tracked_locally",
        },
        "business": {
            "resolution_rate": safe_ratio(resolution_count, len(case_results)),
            "escalation_rate": safe_ratio(escalation_count, len(case_results)),
            "zero_result_rate": safe_ratio(zero_result_count, len(case_results)),
            "csat": "not_tracked_in_local_demo",
        },
    }


def run_rag_evaluation() -> Dict[str, Any]:
    """
    Run the local RAG evaluation workflow and save a report to disk.

    Parameters:
    - None.

    Returns:
    - Dict[str, Any]: The full evaluation report.
    """
    print_banner("RAG Evaluation")
    print_section("Stage 1: Load Golden Dataset")
    evaluation_cases = load_evaluation_cases()
    print_key_value("Cases loaded", str(len(evaluation_cases)))

    print_section("Stage 2: Build Vector Store")
    collection = build_vector_store()
    print_key_value("Collection name", collection.name)
    print_key_value("Chunks loaded", str(collection.count()))

    print_section("Stage 3: Load Generation Model")
    generator = None
    generation_error = ""
    try:
        generator = load_llm()
        print_key_value("Provider", generator["provider"])
        print_key_value("Model", generator["model_name"])
    except Exception as exc:
        generation_error = str(exc)
        try:
            generator = load_huggingface_llm()
            print_key_value("Primary load", "Failed")
            print_key_value("Fallback provider", generator["provider"])
            print_key_value("Fallback model", generator["model_name"])
        except Exception as fallback_exc:
            generation_error = (
                f"Primary load failed: {exc}. "
                f"Hugging Face fallback failed: {fallback_exc}"
            )
            print_key_value("Generation status", "Skipped")
            print_key_value("Reason", generation_error)

    print_section("Stage 4: Evaluate Cases")
    case_results: List[Dict[str, Any]] = []

    for case in evaluation_cases:
        start_time = time.perf_counter()
        retrieval_start = time.perf_counter()
        retrieved_records = retrieve_top_k_records(
            collection=collection,
            user_query=case.query,
            top_k=TOP_K_RETRIEVAL,
        )
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        retrieved_sources = [
            record["metadata"].get("document_name", "unknown")
            for record in retrieved_records
        ]
        retrieved_docs = [record["text"] for record in retrieved_records[:TOP_K_GENERATION]]
        retrieval_metrics = compute_retrieval_metrics(
            retrieved_sources=retrieved_sources,
            gold_sources=case.gold_sources,
            top_k=TOP_K_RETRIEVAL,
        )

        generation_latency_ms = 0.0
        answer = ""
        failed = False
        skipped = False

        if generator is not None:
            try:
                generation_start = time.perf_counter()
                prompt = build_rag_prompt(case.query, retrieved_docs)
                answer = ask_llm(generator, prompt, max_new_tokens=220)
                generation_latency_ms = (time.perf_counter() - generation_start) * 1000
            except Exception as exc:
                failed = True
                answer = f"Generation failed: {exc}"
        else:
            skipped = True
            answer = "Generation skipped because no local model provider was available."

        total_latency_ms = (time.perf_counter() - start_time) * 1000
        generation_metrics = evaluate_answer_quality(
            case=case,
            answer=answer,
            retrieved_context="\n".join(retrieved_docs),
        )

        resolved = (
            generation_metrics["correctness"] >= 0.5
            and generation_metrics["faithfulness"] >= 0.5
            and generation_metrics["format_compliance"] >= 0.5
        )
        zero_result = len(retrieved_records) == 0
        escalated = not resolved

        case_result = {
            "case_id": case.case_id,
            "query": case.query,
            "difficulty": case.difficulty,
            "query_type": case.query_type,
            "retrieved_sources": retrieved_sources,
            "retrieval": retrieval_metrics,
            "generation": generation_metrics,
            "ops": {
                "retrieval_latency_ms": round(retrieval_latency_ms, 2),
                "generation_latency_ms": round(generation_latency_ms, 2),
                "total_latency_ms": round(total_latency_ms, 2),
                "failed": failed,
                "skipped": skipped,
                "timeout": False,
            },
            "business": {
                "resolved": resolved,
                "escalated": escalated,
                "zero_result": zero_result,
            },
            "expected_answer": case.expected_answer,
            "generated_answer": answer,
        }
        case_results.append(case_result)

        print_key_value(f"{case.case_id} hit@3", f"{retrieval_metrics['hit_at_3']:.2f}")
        print_key_value(
            f"{case.case_id} correctness",
            f"{generation_metrics['correctness']:.2f}",
        )

    summary = summarize_results(case_results)
    report = {
        "framework": "chunk-level, answer-level, system-level, and business-level RAG evaluation",
        "dataset_path": str(EVAL_DATASET_PATH),
        "report_path": str(EVAL_REPORT_PATH),
        "generation_available": generator is not None,
        "generation_error": generation_error,
        "summary": summary,
        "cases": case_results,
    }

    EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_section("Stage 5: Scorecard Summary")
    print_key_value("Hit@3", f"{summary['retrieval']['hit_at_3']:.2f}")
    print_key_value("Precision@5", f"{summary['retrieval']['precision_at_5']:.2f}")
    print_key_value("Recall@5", f"{summary['retrieval']['recall_at_5']:.2f}")
    print_key_value("MRR", f"{summary['retrieval']['mrr']:.2f}")
    print_key_value("Faithfulness", f"{summary['generation']['faithfulness']:.2f}")
    print_key_value("Correctness", f"{summary['generation']['correctness']:.2f}")
    print_key_value("Latency P50 (ms)", f"{summary['ops']['latency_p50_ms']:.2f}")
    print_key_value("Latency P95 (ms)", f"{summary['ops']['latency_p95_ms']:.2f}")
    print_key_value("Failure rate", f"{summary['ops']['failure_rate']:.2f}")
    print_key_value("Resolution rate", f"{summary['business']['resolution_rate']:.2f}")
    print_key_value("Report saved", str(EVAL_REPORT_PATH))

    return report
