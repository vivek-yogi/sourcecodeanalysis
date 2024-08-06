from src.helper import repo_ingestion,load_repo,text_splitter,load_embedding
from dotenv import load_dotenv
from langchain.vectorstores import Chroma

load_dotenv()

#url = "https://github.com/entbappy/End-to-End-Chest-Cancer-Classification-using-MLflow-DVC"
url = "https://github.com/vivek-yogi/sourcecodeanalysis"

repo_ingestion(url)

documents  = load_repo("repo/")
text_chunk = text_splitter(documents)
embedding  = load_embedding()

# storing vector in chromadb
vectordb = Chroma.from_documents(text_chunk,embedding=embedding,persist_directory='./db')
vectordb.persist()