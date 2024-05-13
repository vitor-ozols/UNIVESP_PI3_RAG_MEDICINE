from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import uuid
import os
from dotenv import load_dotenv

load_dotenv('../.env')

def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def pdf_to_text(path):
    with open(path, 'rb') as arquivo:
        leitor_pdf = PdfReader(arquivo)
        texto_completo = ''
        for pagina in leitor_pdf.pages:
            texto_pagina = pagina.extract_text()
            if texto_pagina:  # Verifica se a extração de texto foi bem-sucedida
                texto_completo += texto_pagina
    return texto_completo

def ada_embbed(chunk):
    embbed_result = (OpenAI().
                     embeddings.
                     create(model="text-embedding-ada-002",
                            input=chunk))

    return embbed_result.data[0].embedding

def pinecone_conf(api_key, index_name):
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='gcp',
                region='us-west1'
            )
        )
    return pc.Index(index_name)



# dados pinecone
pinecone_api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('PINECONE_INDEX_NAME')
index = pinecone_conf(pinecone_api_key, index_name)

# Caminho para o arquivo PDF
path = "pdf/127728---neo-fidipina-10mg-neo-quimica-30-capsulas.pdf"
text = pdf_to_text(path)
chunks = process_text(text)
for chunk in chunks:
    vector = ada_embbed(chunk)
    metadata = {"texto": chunk,
                "medicine":'neo-fidipina'}
    index.upsert(vectors=[(str(uuid.uuid4()), vector, metadata)])
    print(chunk)