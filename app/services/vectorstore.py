from pinecone import Pinecone

from langchain_openai import OpenAIEmbeddings

from langchain_pinecone import PineconeVectorStore

from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX
)

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

embedding_model = OpenAIEmbeddings()

vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX,
    embedding=embedding_model
)