import os
from git import Repo
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LangauageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings 

# Clone any github repositories
def repo_ingestion(repo_url):
    os.makedirs("repo",exist_ok=True)
    repo_path = "repo/"
    Repo.Clone_from(repo_url,to_path=repo_path)

# Loading repositories as documents
def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(repo_path,
                                           glob = "**/*",
                                           suffixes = [".py"],
                                           parser = LangauageParser(language = Language.PYTHON,parser_threshold=500))
    documents = loader.load()
    return documents

# Creating text chunks
def text_splitter(documents):
    documents_splitter = RecursiveCharacterTextSplitter.from_language(language = Language.PYTHON,
                                                                      chunk_size = 2000,
                                                                      chunk_overlap = 200)
    text_chunks = documents_splitter.split_documents(documents)
    return text_chunks

# loading embeddings model
def load_embedding():
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    return gemini_embeddings