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
if pc.has_index(index_name):
    pc.delete_index(index_name)
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


index = pc.Index(index_name)

# LOAD + PROCESS DOCUMENTS (ONLY ONCE)
extracted_data = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunks = text_split(minimal_docs)
print(minimal_docs[0].page_content)
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
system_prompt = """
You are a helpful assistant for answering questions about Aryan College.

Use the provided context to answer the question naturally.

If the user greets you, greet them back.

If the user asks question related to college, say then to contact the college and provide them the phone number and email.

If the user ask many question , reply in point or number format.

instead of using the ** sign use points , bullets or numbers write all the points in new line like 1. new line then in another \n

If the question is unrelated to Aryan College,
politely ask the user to ask questions related to Aryan College.

If the answer is partially available, try to provide the best possible answer from the context.

Only say "I do not have sufficient information"
when the context truly does not contain the answer.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

retriever = docSearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

docs = retriever.get_relevant_documents(
    "Where is Aryan College located?"
)

for i, doc in enumerate(docs):
    print(f"\n--- DOC {i+1} ---\n")
    print(doc.page_content)