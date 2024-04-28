import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv('../.env')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def ada_embbed(chunk):
    embbed_result = (client.
                     embeddings.
                     create(model="text-embedding-ada-002",
                            input=chunk))

    return embbed_result.data[0].embedding

def pinecone_conf():
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    return pc.Index(os.getenv('PINECONE_INDEX_NAME'))

def pinecone_question(vector, medicine):
    query_result = index.query(vector=vector,
                               top_k=3,
                               include_metadata=True,
                               filter={"medicine": {"$eq": f"{medicine}"}})['matches']

    return (query_result[0].metadata['texto'] +
            query_result[1].metadata['texto'] +
            query_result[2].metadata['texto'])

index = pinecone_conf()

while True:

    question = input("Digite sua Mensagem: ")
    vector = ada_embbed(question)
    rag = pinecone_question(vector=vector,
                            medicine='losartana')

    prompt = open('prompts/question_1.txt', encoding='utf-8').read()

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": question},
                  {"role": "system", "content": rag}])

    answer = completion.choices[0].message.content
    print(answer)