from dotenv import load_dotenv
from pinecone import ServerlessSpec
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from torch import embedding
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, embedding
import os

load_dotenv()

index_name = "final-project"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key = pinecone_api_key )

extracted_data = load_pdf_files("data")
print(len(extracted_data))
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunks = text_split(minimal_docs)
print(f"number of chunks: {len(texts_chunks)}")




index_name = "final-project"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud = "aws", region ="us-east-1")
    )
    
    
index = pc.Index(index_name)


docSearch = PineconeVectorStore.from_documents(
    documents = minimal_docs,
    embedding = embedding,
    index_name = index_name
)


docSearch = PineconeVectorStore.from_documents(
    documents = texts_chunks,
    embedding = embedding,
    index_name = index_name
)