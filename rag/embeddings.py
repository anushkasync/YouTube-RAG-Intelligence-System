from langchain_huggingface import HuggingFaceEmbeddings
from config.config import EMBEDDING_MODEL_NAME

embedding_model = HuggingFaceEmbeddings(
    model_name = EMBEDDING_MODEL_NAME
)
