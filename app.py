#!pip3 install chromadb==0.5.0
import os
import nest_asyncio
import threading
from flask import Flask, request, jsonify, render_template
import argparse
#from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pymongo.mongo_client import MongoClient
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

openapi_key = os.getenv('OPENAI_API_KEY')
mongo_uri = os.getenv('MONGO_URI')

nest_asyncio.apply()

app = Flask(__name__)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a AI chatbot. Your role is to build conversation with student and give good responses.
If you are not confident on any question. Let student know that you don't have that information.
Use the below context to complete the conversation. If the context is empty string or the answer to question is not present in
context then let the student know that you don't have enough sources to answer his question and also remind him you are an 
AI chatbot whose role is to answer the questions only related to University of Massachusetts, Lowell. Strictly look for answer only in context
Context is the text between ``` below
context : ```{context}```

"""

messages = []

q=[]
c=[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    
    data = request.json
    query_text = data.get('query')
    
    uri = "mongodb+srv://umlbot:Boston123@cluster0.mlwhq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    
    # Create a new client and connect to the server
    client = MongoClient(uri)
    
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        query_text = "Pinged your deployment. You successfully connected to MongoDB!"
    except Exception as e:
        query_text = e

    # # Prepare the DB.
    # embedding_function = OpenAIEmbeddings()
    # #db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # model = ChatOpenAI()
    # q.append(query_text)
    # c[0]+=1
    # if c[0]==5:
    #     q.pop(0)
    #     c[0]-=1
    
    # q_text = ' '.join(q)

    # # # Search the DB.
    # #results = db.similarity_search_with_relevance_scores(q_text, k=10)
    # vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    #     mongo_uri,
    #     namespace="UML_ChatBot.demo-db",
    #     embedding= OpenAIEmbeddings(),
    #     index_name="vector_index"
    # )
    # results = vector_search.similarity_search_with_score(query=q_text, k=10)
    
    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # if len(results) == 0 or results[0][1] < 0.65:
    #     context_text = ""
    #     #print('hello')
    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format(context=context_text)

    # if messages == []:
    #     messages.append(SystemMessage(content=prompt))

    # messages[0]=SystemMessage(content=prompt)
    # messages.append(HumanMessage(content=query_text))

    # res = model.invoke(messages)
    # messages.append(AIMessage(content=res.content))
    # return jsonify({'response': res.content})
    return jsonify({'response': query_text})


if __name__ == '__main__':
    app.run(debug=True)
# def run_app():
#     app.run(debug=True, port=5010, use_reloader=False)

# thread = threading.Thread(target=run_app)
# thread.start()
