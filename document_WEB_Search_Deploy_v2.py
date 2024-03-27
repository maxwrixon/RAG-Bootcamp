#!/usr/bin/env python
# coding: utf-8

# # Cohere Document Search with Langchain

# This example shows how to use the Python [langchain](https://python.langchain.com/docs/get_started/introduction) library to run a text-generation request against [Cohere's](https://cohere.com/) API, then augment that request using the text stored in a collection of local PDF documents.
# 
# **Requirements:**
# - You will need an access key to Cohere's API key, which you can sign up for at (https://dashboard.cohere.com/welcome/login). A free trial account will suffice, but will be limited to a small number of requests.
# - After obtaining this key, store it in plain text in your home in directory in the `~/.cohere.key` file.
# - (Optional) Upload some pdf files into the `source_documents` subfolder under this notebook. We have already provided some sample pdfs, but feel free to replace these with your own.

# ## Set up the RAG workflow environment

# In[1]:


from getpass import getpass
import os
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatCohere
from langchain.document_loaders import TextLoader
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import Cohere
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


# Set up some helper functions:

# In[2]:


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# Make sure other necessary items are in place:

# In[3]:


try:
    os.environ["COHERE_API_KEY"] =  'ht9CB7BKp1gXtE9RSGbwCplCOfTnwysPj2misv2y'
except Exception:
    print(f"ERROR: You must have a Cohere API key available in your home directory at ~/.cohere.key")

# Look for the source_documents folder and make sure there is at least 1 pdf file here
contains_pdf = False
directory_path = "./source_documents"
if not os.path.exists(directory_path):
    print(f"ERROR: The {directory_path} subfolder must exist under this notebook")
for filename in os.listdir(directory_path):
    contains_pdf = True if ".pdf" in filename else contains_pdf
if not contains_pdf:
    print(f"ERROR: The {directory_path} subfolder must contain at least one .pdf file")


# ## Start with a basic generation request without RAG augmentation
# 
# Let's start by asking the Cohere LLM a difficult, domain-specific question we don't expect it to have an answer to. A simple question like "*What is the capital of France?*" is not a good question here, because that's basic knowledge that we expect the LLM to know.
# 
# Instead, we want to ask it a question that is very domain-specific that it won't know the answer to. A good example would an obscure detail buried deep within a company's annual report. For example:
# 
# "*How many Vector scholarships in AI were awarded in 2022?*"

# In[4]:


query = "What is the December-2023 snapshot for Mastercards?"


# ## Now send the query to Cohere

# In[5]:


llm = Cohere()
result = llm(query)
print(f"Result: \n\n{result}")


# In[7]:


# Load the pdfs
loader = PyPDFDirectoryLoader(directory_path)
docs = loader.load()
print(f"Number of source materials: {len(docs)}")

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
print(f"Number of text chunks: {len(chunks)}")

# Define the embeddings model
model_name = "BAAI/bge-small-en-v1.5"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

print(f"Setting up the embeddings model...")
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
)

print(f"Done")


# # Retrieval: Make the document chunks available via a retriever

# The retriever will identify the document chunks that most closely match our original query. (This takes about 1-2 minutes)

# In[8]:


vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Retrieve the most relevant context from the vector store based on the query(No Reranking Applied)
docs = retriever.get_relevant_documents(query)


# Let's see what results it found. Important to note, these results are in the order the retriever thought were the best matches.

# In[9]:


pretty_print_docs(docs)


# These results seem to somewhat match our original query, but we still can't seem to find the information we're looking for. Let's try sending our LLM query again including these results, and see what it comes up with.

# In[10]:


print(f"Sending the RAG generation with query: {query}")
qa = RetrievalQA.from_chain_type(llm=llm,
        chain_type="stuff",
        retriever=retriever)
print(f"Result:\n\n{qa.run(query=query)}") 


# # Adding Web search capability

# !pip install google
# 

# !pip install googlesearch-python
# 

# In[13]:


from bs4 import BeautifulSoup
from getpass import getpass
from googlesearch import search
import os
from pathlib import Path
import requests

from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Cohere
from langchain_community.vectorstores import FAISS


# In[14]:


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


def main():

    # Setup the environment
    os.environ["COHERE_API_KEY"] = 'ht9CB7BKp1gXtE9RSGbwCplCOfTnwysPj2misv2y'

    # Start with making a generation request without RAG augmentation
    query = "What is the current unemployment rate in Canada?"
    llm = Cohere()
    print(f"*** Sending non-RAG augmented generation request for query: {query}\n")
    result = llm(query)
    print(f"Result: {result}\n")

    print(f"*** Now retrieve results from a Google web search for the same query\n")
    result_text = ""
    for result_url in search(query, num_results=10):
        response = requests.get(result_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        result_text = result_text + soup.get_text()

    # Split the result text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_text(result_text)
    print(f"*** Number of text chunks: {len(texts)}\n")

    # Define Embeddings Model
    model_name = "BAAI/bge-small-en-v1.5"
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    print(f"*** Setting up the embeddings model...\n")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs=encode_kwargs
    )

    # Set up the base vector store retriever
    print(f"*** Setting up the base vector store retriever\n")
    vectorstore = FAISS.from_texts(texts, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    # Retrieve the most relevant context from the vector store based on the query(No Reranking Applied)
    docs = retriever.get_relevant_documents(query)
    #pretty_print_docs(docs)

    
    # The Generation part of RAG Pipeline
    print(f"*** Now do the reranked RAG generation with query: {query}\n")
    qa = RetrievalQA.from_chain_type(llm=llm,
            chain_type="stuff",
            retriever=compression_retriever)
    print(f"*** Running generation...\n")
    print(f"Result: {qa.run(query=query)}")


if __name__ == "__main__":
    main()



# In[ ]:




