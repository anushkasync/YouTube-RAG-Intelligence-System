from datetime import datetime

from metrics.retrieval_eval import retrieval_metrics
from metrics.system_metrics import system_metrics
from config.config import CONFIG


def run_benchmark(evaluated_results):

    print("BENCHMARK STARTED")

    if not isinstance(evaluated_results, list):

        raise ValueError(
            "Expected list of evaluated results"
        )

    results = {

        "timestamp": datetime.utcnow().isoformat() + "Z",

        "model": CONFIG["LLM_MODEL"],

        "cases": [],

        "avg_score": 0.0
    }

    total_score = 0.0

    for item in evaluated_results:

        output = item.get(
            "output",
            ""
        )

        processed_chunks = item.get(
            "processed_chunks",
            []
        )

        latency = item.get(
            "latency",
            0.0
        )

        retrieval = retrieval_metrics(

            item.get("task", ""),

            processed_chunks
        )

        system = system_metrics(
            latency,
            output
        )

        retrieval_score = retrieval.get(
            "average_top_k_cosine_similarity"
        )

        if retrieval_score is None:

            retrieval_score = 0.0

        system_score = round(
            1 / (1 + latency),
            4
        )

        final_score = (
            0.6 * retrieval_score
            + 0.4 * system_score
        )

        total_score += final_score

        results["cases"].append({

            "test_id": item.get(
                "test_id"
            ),

            "task": item.get(
                "task"
            ),

            "mode": item.get(
                "mode"
            ),

            "retrieval": {

                "embedding_similarity_score":
                    retrieval.get(
                        "embedding_similarity_score"
                    ),

                "average_top_k_cosine_similarity":
                    retrieval.get(
                        "average_top_k_cosine_similarity"
                    ),

                "fallback_used":
                    retrieval.get(
                        "fallback_used",
                        False
                    ),

                "error":
                    retrieval.get(
                        "error"
                    )
            },

            "system": {

                "latency": round(
                    latency,
                    3
                ),

                "system_score":
                    system_score
            },

            "final_score": round(
                final_score,
                4
            )
        })

    if results["cases"]:

        results["avg_score"] = round(

            total_score / len(
                results["cases"]
            ),

            4
        )

    print("BENCHMARK COMPLETE")

    return results