from dotenv import load_dotenv
import os

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, embedding

load_dotenv()

# API KEYS
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# INIT PINECONE
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "final-project"

# CREATE INDEX (only if not exists)
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# LOAD INDEX
index = pc.Index(index_name)

# LOAD + PROCESS DOCUMENTS (ONLY ONCE)
extracted_data = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunks = text_split(minimal_docs)

print(f"Loaded docs: {len(extracted_data)}")
print(f"Chunks: {len(texts_chunks)}")

# STORE IN VECTOR DB
docSearch = PineconeVectorStore.from_documents(
    documents=texts_chunks,
    embedding=embedding,
    index_name=index_name
)

# LLM (OpenRouter)
chatModel = ChatOpenAI(
    model="openai/gpt-4o-mini"
)

# PROMPT
system_prompt = (
    "You are a helpful assistant for answering questions about Aryan College. "
    "Use the provided context to answer the question. "
    "If you don't know, say you don't know.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# RETRIEVER
retriever = docSearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# CHAINS
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)