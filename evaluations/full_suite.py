from evaluations.test_loader import load_test_cases
from experiments.run_rag import run_rag
from evaluations.run_evaluations import run_evaluation
from benchmarks.run_benchmark import run_benchmark


def run_full_suite(
    cache,
    llm,
    config,
    mode
):
    """
    mode:
        - benchmark → benchmark only
        - eval → evaluation only
    """

    if mode not in ["benchmark", "eval"]:

        raise ValueError(
            "Invalid mode. Use 'benchmark' or 'eval'."
        )

    test_cases = load_test_cases()

    pipeline_results = []

    eval_results = []

    for test in test_cases:

        run_result = run_rag(
            test,
            config,
            cache,
            llm
        )

        output = run_result.get(
            "output",
            ""
        )

        processed_chunks = (
            run_result.get(
                "processed_chunks",
                {}
            ) or {}
        )

        latency = run_result.get(
            "latency",
            0.0
        )

        metadata = run_result.get(
            "metadata",
            {}
        )

        mode_used = metadata.get(
            "mode"
        )

        pipeline_results.append({

            "test_id": test.get("id"),

            "task": test.get("task"),

            "output": output,

            "processed_chunks": processed_chunks,

            "latency": latency,

            "mode": mode_used,

            "metadata": metadata
        })

    if mode == "eval":

        for item in pipeline_results:

            eval_result = run_evaluation(

                query=item["task"],

                output=item["output"],

                processed_chunks=item[
                    "processed_chunks"
                ],

                mode=item["mode"],

                latency=item["latency"],

                config=config
            )

            eval_results.append({

                "test_id": item["test_id"],

                "task": item["task"],

                "mode": item["mode"],

                **eval_result
            })

        return {

            "mode": mode,

            "total_test_cases": len(
                test_cases
            ),

            "evaluations": eval_results
        }
    
    benchmark_results = run_benchmark(
        pipeline_results
    )

    return {

        "mode": mode,

        "total_test_cases": len(
            test_cases
        ),

        "benchmark": benchmark_results
    }