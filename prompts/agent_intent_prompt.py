INTENT_PROMPT = """
Classify the query into ONE label:

summary:
User wants video summary

keypoints:
User wants main points

qa_gen:
User wants the system to CREATE questions from the video

rag_qa:
User is ASKING a question about the video content

unknown:
Unclear intent

Return ONE label only.
No explanation.

Query: {query}
Output:
"""