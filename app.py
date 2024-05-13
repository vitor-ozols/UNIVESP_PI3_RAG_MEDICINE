from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
# teste deploy 2

load_dotenv('.env')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))

app = Flask(__name__)


######### Funções de oeração ##########
def ada_embbed(chunk):
    """Vetoriza a sentença no modelo ADA para comparar com vetor do banco"""
    embbed_result = (client.
                     embeddings.
                     create(model="text-embedding-ada-002",
                            input=chunk))
    return embbed_result.data[0].embedding


def pinecone_question(vector, medicine):
    """Realiza query de termo equivalente no banco de dados
     de vetor e retorna os 5 melhores resultados concatenados"""
    query_result = index.query(vector=vector,
                               top_k=5,
                               include_metadata=True,
                               filter={"medicine": {"$eq": f"{medicine}"}})['matches']

    return (query_result[0].metadata['texto'] + " | \n\n" +
            query_result[1].metadata['texto'] + " | \n\n" +
            query_result[2].metadata['texto'] + " | \n\n" +
            query_result[3].metadata['texto'] + " | \n\n" +
            query_result[4].metadata['texto'])


######### Funções do FLASK ##########
@app.route("/")
def index_html():
    return render_template("index.html")


@app.route("/chat-options")
def chat_options():
    return render_template("chat-options.html")


@app.route('/chat', methods=['POST'])
def chat():
    medicine = request.form['medicine']
    return render_template("chat.html", medicine=medicine)

@app.route("/chat-logic", methods=["POST"])
def chat_logic():

    medicine = request.form["medicine"]
    question = request.form["message"]
    vector = ada_embbed(question)
    rag = pinecone_question(vector=vector,
                            medicine=medicine)

    prompt = open('resource/prompts/question_1.txt', encoding='utf-8').read()

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": question},
                  {"role": "system", "content": rag}])

    return jsonify({"message": completion.choices[0].message.content})




if __name__ == "__main__":
    app.run(debug=True)