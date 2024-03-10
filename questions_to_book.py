# PDF Loaders. If unstructured gives you a hard time, try PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from langchain.vectorstores import Chroma
from langchain_community.embeddings.openai import OpenAIEmbeddings


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
file_path=os.getenv('FILE_PATH')

def load_and_create_embeddings():
    #loading the data
    loader = TextLoader(file_path=file_path)
    data = loader.load()

    #Chunk data up into smaller documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(data)

    #Create embeddings of documents 
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return texts,embeddings


def load_into_chroma(texts,embeddings,query):
    vectorstore = Chroma.from_documents(texts, embeddings)
    docs = vectorstore.similarity_search(query)
    return docs


if __name__=='__main__':
    query = "What is great about having kids?"
    texts,embeddings=load_and_create_embeddings()
    docs = load_into_chroma(texts,embeddings,query)
    for doc in docs:
        print (f"{doc.page_content}\n")