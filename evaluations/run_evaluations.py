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

    if ragas.get("fallback_used"):

        ragas_score = None

    else:

        ragas_score = round(
            (
                ragas["faithfulness"] +
                ragas["answer_relevancy"]
            ) / 2,
            3
        )

    if ux.get("fallback_used"):

        ux_score = None

    else:

        ux_score = ux["overall_score"]


    if ragas_score is not None and ux_score is not None:

        generation_quality = round(
            (
                0.6 * ragas_score +
                0.4 * ux_score
            ),
            3
        )

    elif ragas_score is not None:

        generation_quality = ragas_score

    elif ux_score is not None:

        generation_quality = ux_score

    else:

        generation_quality = None

    return {

        "ragas": {
            "faithfulness": ragas.get("faithfulness"),
            "answer_relevancy": ragas.get("answer_relevancy"),
            "overall_score": ragas_score,
            "fallback_used": ragas.get("fallback_used", False),
            "error": ragas.get("error")
        },

        "ux": {
            "clarity": ux.get("clarity"),
            "completeness": ux.get("completeness"),
            "usefulness": ux.get("usefulness"),
            "overall_score": ux_score,
            "fallback_used": ux.get("fallback_used", False),
            "error": ux.get("error")
        },

        "generation_quality": generation_quality,

        "latency": latency
    }