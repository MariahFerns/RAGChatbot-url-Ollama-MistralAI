# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Install required libraries
# # !pip install -r requirements.txt
# -

# Import required libraries
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community import embeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# +
# Define a function to take input URLS and process them

def process_input(urls, question):
    model_local = Ollama(model = 'mistral')
    
    # Convert URls to list using WebBaseLoader
    url_list = urls.split('\n')
    docs = [ WebBaseLoader(url).load() for url in url_list]
    docs_list = [ item for sublist in docs for item in sublist]
    
    # Split text documents into chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_split = text_splitter.split_documents(docs_list)
    
    # Convert text chunks into embeddings and store in the vector database
    vector_store = Chroma.from_documents(
        documents=doc_split,
        collection_name='rag_chroma',
        embedding = embeddings.ollama.OllamaEmbeddings(model = 'nomic-embed-text')
    ) 
    retriever = vector_store.as_retriever()
    
    # Perform RAG
    after_rag_template = ''' Answer the question based only on the following context:
    {context}
    Question: {question}
    '''
    
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    
    after_rag_chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    
    return after_rag_chain.invoke(question)


# +
# Streamlit UI

# Title & Desc
st.title('🔖Document Query with Ollama')
st.write('Enter URLs (one per line) and your question pertaining to the URLs ')

# Enter URLs
urls = st.text_area('Enter URLs separated by new line', height=150)
question = st.text_input('Question')

# Submit Button
if st.button('Query URLs'):
    with st.spinner('Processing...'):
        answer = process_input(urls, question)
        st.text_area('Answer:', value=answer, height=300, disabled=True)

