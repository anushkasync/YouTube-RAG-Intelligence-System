import os
import math

from dotenv import load_dotenv

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models.llms import LLM

from utils.llm import GROQLLM
from utils.cache_manager import CacheManager

from config.config import (
    JUDGE_MODEL,
    EMBEDDING_MODEL_NAME
)

load_dotenv()

cache_manager = CacheManager()


class RagasGroqLLM(LLM):

    model_name: str
    api_key: str

    @property
    def _llm_type(self):
        return "groq"

    def _call(
        self,
        prompt,
        stop=None,
        run_manager=None,
        **kwargs
    ):

        llm = GROQLLM(
            api_key=self.api_key,
            model=self.model_name,
            cache_manager=cache_manager
        )

        response = llm.invoke(prompt)

        if hasattr(response, "content"):
            return response.content

        return str(response)


def safe_float(value):

    try:

        value = float(value)

        if math.isnan(value):
            return None

        if math.isinf(value):
            return None

        return round(value, 3)

    except Exception:
        return None


def ragas_metrics(
    query,
    context,
    response,
    ground_truth=None
):
    """
    Evaluate generation quality using RAGAS v0.4.3
    """

    try:

        if context is None:
            context = []

        if isinstance(context, str):
            context = [context]

        dataset = EvaluationDataset.from_list([
            {
                "user_input": query,
                "response": response,
                "retrieved_contexts": context,
                "reference": ground_truth or ""
            }
        ])

        groq_key = os.getenv("GROQ_API_KEY")

        base_llm = RagasGroqLLM(
            model_name=JUDGE_MODEL,
            api_key=groq_key
        )

        ragas_llm = LangchainLLMWrapper(base_llm)

        hf_embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )

        ragas_embeddings = LangchainEmbeddingsWrapper(
            hf_embeddings
        )

        result = evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(),
                AnswerRelevancy()
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )

        df = result.to_pandas()

        faithfulness_score = safe_float(
            df["faithfulness"][0]
        )

        answer_relevancy_score = safe_float(
            df["answer_relevancy"][0]
        )

        overall_score = None

        valid_scores = [
            s for s in [
                faithfulness_score,
                answer_relevancy_score
            ]
            if s is not None
        ]

        if valid_scores:

            overall_score = round(
                sum(valid_scores) / len(valid_scores),
                3
            )

        return {

            "faithfulness": faithfulness_score,

            "answer_relevancy": answer_relevancy_score,

            "overall_score": overall_score,

            "fallback_used": False,

            "error": None
        }

    except Exception as e:

        print(f"RAGAS ERROR: {str(e)}")

        return {

            "faithfulness": None,

            "answer_relevancy": None,

            "overall_score": None,

            "fallback_used": True,

            "error": str(e)
        }