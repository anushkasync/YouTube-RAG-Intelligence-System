from langchain_core.documents import Document


def build_documents(chunks, video_id):
    docs = []

    for i, chunk in enumerate(chunks):
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "video_id": video_id,
                    "chunk_id": i
                }
            )
        )

    return docs