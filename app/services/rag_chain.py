from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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


def ask_question(question):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = ChatPromptTemplate.from_template(
        """
        {system_prompt}

        Context:
        {context}

        Question:
        {question}
        """
    )

    chain = prompt | llm

    result = chain.invoke({
        "system_prompt": SYSTEM_PROMPT,
        "context": context,
        "question": question
    })

    sources = []

    for doc in docs:

        sources.append({
            "document": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "content": doc.page_content[:300]
        })

    return {
        "answer": result.content,
        "sources": sources
    }