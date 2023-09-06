import os
import openai
import pinecone
import sys
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

loader = PyPDFDirectoryLoader("data/")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], 
                environment=os.environ["PINECONE_ENVIRONMENT"])
embedding = OpenAIEmbeddings()
# Initialize Pinecone index
indexName = os.environ["PINECONE_INDEX_NAME"]
index = pinecone.Index(index_name=indexName)
vectordb = Pinecone(index=index, embedding=embedding, text_key="text")

# vectordb.add_documents(splits) # only need to run this once!

question = input("\nPrompt: ")
docs = vectordb.similarity_search(question,k=3)

llm_name = "gpt-3.5-turbo-0301"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

result = qa_chain({"query": question})

print(result["result"])

source_document = result["source_documents"][0]
print(source_document.metadata)