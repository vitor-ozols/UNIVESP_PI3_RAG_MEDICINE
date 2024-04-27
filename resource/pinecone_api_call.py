import os

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv('../.env')
def ada_embbed(chunk):
    embbed_result = (OpenAI().
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

#question = input("Digite sua Mensagem: ")
question = "qual o efeito colateral?"

vector = ada_embbed(question)

pinecone_question(vector=vector,
                  medicine='losartana')

prompt = """
Seu nome é o Dr.Gebara, você é uma inteligência artificial que responde dúvidas sobre bulas de remédios.
- Você é simpático e Educado;
- Você responde de acordo com o texto que você encontra nas bulas dos remédios;
"""

completion = OpenAI.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)



