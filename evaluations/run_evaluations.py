from evaluations.llm_as_judge import llm_judge_ux
from evaluations.ragas_metrics import ragas_metrics
from metrics.retrieval_eval import retrieval_metrics


def get_eval_context(processed_chunks, mode):

    if not isinstance(processed_chunks, dict):
        return []

    if mode in ("raw", "small"):

        return processed_chunks.get(
            "raw",
            []
        )[:3]

    elif mode == "medium":

        return processed_chunks.get(
            "medium",
            []
        )[:3]

    elif mode == "long":

        return processed_chunks.get(
            "long",
            []
        )[:3]

    return []


def run_evaluation(
    query,
    output,
    processed_chunks,
    mode,
    latency,
    config
):

    context = get_eval_context(
        processed_chunks,
        mode
    )

    retrieval = retrieval_metrics(
        query,
        context
    )

    ragas = ragas_metrics(
        query,
        context,
        output
    )

    ux = llm_judge_ux(
        output,
        context,
        query
    )

    ragas_score = (
        ragas.get("faithfulness", 0.5) +
        ragas.get("answer_relevancy", 0.5)
    ) / 2

    ux_score = ux.get(
        "overall_score",
        0.5
    )

    generation_quality = (
        0.6 * ragas_score +
        0.4 * ux_score
    )

    return {
        "ragas_score": ragas_score,
        "ux_score": ux_score,
        "generation_quality": generation_quality,
        "latency": latency
    }