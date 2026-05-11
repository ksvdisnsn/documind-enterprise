from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

from app.services.retriever import retriever

SYSTEM_PROMPT = """
You are DocuMind Enterprise AI.

STRICT RULES:

1. Answer ONLY from the provided context.

2. NEVER use external knowledge.

3. If information is not found in documents, say:
   "This information is not available in the provided documents."

4. Always provide citations.
"""

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


def ask_question(question):

    result = qa_chain.invoke({
        "query": question
    })

    answer = result["result"]

    sources = []

    for doc in result["source_documents"]:

        sources.append({
            "document": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "content": doc.page_content[:300]
        })

    return {
        "answer": answer,
        "sources": sources
    }