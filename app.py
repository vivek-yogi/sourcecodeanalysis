import warnings
warnings.filterwarnings('ignore')
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os 
from src.helper import repo_ingestion,load_embedding
from flask import Flask, render_template,jsonify,request
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)
load_dotenv()

embeddings        = load_embedding()
persist_directory = "db"

# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory = persist_directory, embedding_function = embeddings)

llm    = ChatGoogleGenerativeAI(model = "gemini-1.5-pro")
memory = ConversationSummaryMemory(llm = llm, memory_key = "chat_history", return_messages=True)
qa     = ConversationalRetrievalChain.from_llm(llm, retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k":8}), memory=memory)


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')



@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():
    
    if request.method == 'POST':
        user_input = request.form['question']
        repo_ingestion(user_input)
        os.system("python store_index.py")

    return jsonify({"response": str(user_input) })


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg   = request.form["msg"]
    input = msg
    print(input)

    if input == "clear":
        os.system("rm -rf repo")

    result = qa(input)
    print(result['answer'])
    return (str(result["answer"]))



if __name__ == '__main__':
    app.run(debug=True)

