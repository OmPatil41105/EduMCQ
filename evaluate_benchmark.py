"""
Empirical Benchmark Evaluation Script for EduMCQ Pipeline.
Executes systematic quantitative evaluation across multi-domain corpora:
- Computer Engineering (Operating Systems)
- Life Sciences (Cellular Respiration)
- Artificial Intelligence (Natural Language Processing)

Computes statistical aggregates for DPS, DDI, Readability, and NLI rejection efficiency.
"""

import os
import time
import pandas as pd
from edumcq.mcq_pipeline import EduMCQPipeline

DOMAINS = [
    ("Computer Engineering", "data/samples/operating_systems.txt"),
    ("Life Sciences", "data/samples/cellular_respiration.txt"),
    ("Artificial Intelligence", "data/samples/natural_language_processing.txt")
]


def run_benchmark():
    print("=" * 76)
    print("EduMCQ Empirical Benchmark Evaluation")
    print("Executing quantitative assessment across 3 academic corpora...")
    print("=" * 76)

    pipeline = EduMCQPipeline(use_transformer_qg=False, use_nli_validation=True, difficulty="medium")
    results = []

    for domain_name, file_rel_path in DOMAINS:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file_rel_path)

        if not os.path.exists(file_path):
            print(f"Skipping {domain_name}: {file_path} not found.")
            continue

        print(f"\nEvaluating Domain: {domain_name} ({file_rel_path})...")
        start_time = time.time()

        res = pipeline.generate_from_file(
            file_path=file_path,
            num_questions=4,
            num_options=4,
            difficulty="medium"
        )
        elapsed = time.time() - start_time

        if res.get("success"):
            mcqs = res.get("mcqs", [])
            metrics = res.get("metrics", {})

            # Count NLI classifications
            nli_contradictions = 0
            nli_neutrals = 0
            for q in mcqs:
                for d in q.get("distractors", []):
                    verdict = d.get("nli_verdict", "")
                    if "Contradiction" in verdict:
                        nli_contradictions += 1
                    elif "Neutral" in verdict:
                        nli_neutrals += 1

            results.append({
                "Domain": domain_name,
                "Questions": len(mcqs),
                "Mean DPS": metrics.get("mean_distractor_plausibility", 0.0),
                "Mean DDI": metrics.get("mean_distractor_diversity", 0.0),
                "Flesch Ease": metrics.get("mean_readability_score", 0.0),
                "Quality Index": metrics.get("quiz_quality_index", 0.0),
                "NLI Contradictions": nli_contradictions,
                "NLI Neutrals": nli_neutrals,
                "Latency (s)": round(elapsed, 2)
            })
            print(f"  -> Generated {len(mcqs)} items in {elapsed:.2f}s | DPS: {metrics.get('mean_distractor_plausibility', 0):.3f} | DDI: {metrics.get('mean_distractor_diversity', 0):.3f}")
        else:
            print(f"  -> Evaluation failure: {res.get('error')}")

    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 76)
        print("SUMMARY BENCHMARK REPORT (Empirical Validation)")
        print("=" * 76)
        print(df.to_string(index=False))
        print("=" * 76)

        # Compute averages
        avg_dps = df["Mean DPS"].mean()
        avg_ddi = df["Mean DDI"].mean()
        avg_qi = df["Quality Index"].mean()
        avg_lat = df["Latency (s)"].mean()

        print(f"\nSystem Performance Aggregates:")
        print(f"  - Overall Mean DPS:       {avg_dps:.3f} (Ideal: 0.50 - 0.75)")
        print(f"  - Overall Mean DDI:       {avg_ddi:.3f} (Ideal: > 0.45)")
        print(f"  - Overall Quality Index:  {avg_qi:.1f} / 100")
        print(f"  - Mean Batch Latency:     {avg_lat:.2f}s per 4-question corpus")
        print("=" * 76)


if __name__ == "__main__":
    run_benchmark()
