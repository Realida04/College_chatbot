from flask import Flask, render_template, request
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceEmbeddings
from torch import embedding
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, embedding
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain 
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os
from src.prompt import *


app = Flask(__name__)
@app.route("/")
def index():
    return render_template("chat.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000, debug=True)