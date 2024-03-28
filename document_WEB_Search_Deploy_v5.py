#!/usr/bin/env python
# coding: utf-8

# # Import Streamlit

# import streamlit as st
# 
# # Use a text input field for the user to enter a query
# query = st.text_input("Enter your query:", "", key="unique_query_input_key")
# 
# 

# import streamlit as st
# import os
# from langchain.llms import Cohere
# from langchain.document_loaders.pdf import PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceBgeEmbeddings
# # Include any other necessary imports
# 
# # Set the API key for Cohere
# os.environ["COHERE_API_KEY"] = "ht9CB7BKp1gXtE9RSGbwCplCOfTnwysPj2misv2y"
# 
# # Initialize the language model
# llm = Cohere()
# 
# # Define the directory path for document search
# directory_path = r"C:\Users\yulia\Desktop\RAG BOOTCAMP\RAG-Bootcamp\source_documents"
# 
# # Use a text input field for the user to enter a query
# query = st.text_input("Enter your query:", "", key="unique_query_input_key")
# 
# # Check if the query is not empty and has a reasonable length
# if query and len(query.strip()) > 0:  # Adjust the length check as necessary
#     # Document search logic
#     # Assuming you have a function or logic to handle document search
#     # For example:
#     loader = PyPDFDirectoryLoader(directory_path)
#     docs = loader.load()
#     # Add logic to process documents and display results
# 
#     # Web search logic
#     # Utilize the 'llm' function to send the query to a language model and get the results
#     try:
#         web_search_results = llm(query)  # Make the API call
#         st.write(f"Web Search Results: \n\n{web_search_results}")
#     except Exception as e:
#         st.error(f"An error occurred: {e}")
# 
#     # If you want to show results from the document search
#     # Assuming 'process_documents' is a function that returns search results from your documents
#     document_search_results = "Document search results after processing"
#     st.write(f"Document Search Results: \n\n{document_search_results}")
# else:
#     # Inform the user that the input is too short
#     st.error("Please enter a more substantial query.")
# 
# 

# In[ ]:


import streamlit as st
import os
from langchain.llms import Cohere
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
# Include any other necessary imports

# Set the API key for Cohere
os.environ["COHERE_API_KEY"] = "KJefdUzquaCHsNJ7SXgJFHZH3Ti132hyxmKMySGU"

# Initialize the language model
llm = Cohere()

# Define the directory path for document search
directory_path = r"C:\Users\yulia\Desktop\RAG BOOTCAMP\RAG-Bootcamp\source_documents"

# Use a text input field for the user to enter a query
query = st.text_input("Enter your query:", "", key="unique_query_input_key")

# Initialize variable to hold web search results
web_search_results = ""

# Check if the query is not empty and has a reasonable length
if query and len(query.strip()) > 0:  # Adjust the length check as necessary
    # Document search logic
    loader = PyPDFDirectoryLoader(directory_path)
    docs = loader.load()
    # Logic to process documents and display results would go here

    # Web search logic
    try:
        web_search_results = llm(query)  # Make the API call
    except Exception as e:
        st.error(f"An error occurred during web search: {e}")

    # Display the web search results if available
    if web_search_results:
        st.write(f"Web Search Results: \n\n{web_search_results}")

    # Assuming 'process_documents' returns search results from your documents
    document_search_results = "Document search results after processing"
    if document_search_results:
        st.write(f"Document Search Results: \n\n{document_search_results}")
else:
    # Inform the user that the input is too short
    st.error("Please enter a more substantial query.")


# import streamlit as st
# # Include other necessary imports
# 
# # Define the directory path
# directory_path = r"C:\Users\yulia\Desktop\RAG BOOTCAMP\RAG-Bootcamp\source_documents"
# 
# # Use a text input field for the user to enter a query
# query = st.text_input("Enter your query:", "", key="unique_query_input_key")
# 
# # Check if the query is not empty and has a reasonable length
# if query and len(query.strip()) > 0:  # Adjust the length check as necessary
#     # Document search logic
#     # ...
# 
#     # Web search logic
#     # Assuming 'llm' is a function that sends the query to a language model
#     if query.strip():  # Ensures query is not just whitespace
#         try:
#             web_search_results = llm(query)  # Make the API call
#             st.write(f"Web Search Results: \n\n{web_search_results}")
#         except Exception as e:
#             st.error(f"An error occurred: {e}")
# else:
#     # Inform the user that the input is too short
#     st.error("Please enter a more substantial query.")
# 

# import streamlit as st
# from langchain.document_loaders.pdf import PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceBgeEmbeddings
# 
# # Initialize these variables to avoid reference before assignment errors
# docs = []
# document_search_results = ""
# web_search_results = ""
# 
# # Define the directory path
# directory_path = r"C:\Users\yulia\Desktop\RAG BOOTCAMP\RAG-Bootcamp\source_documents"
# 
# # Use a text input field for the user to enter a query
# query = st.text_input("Enter your query:", "", key="unique_query_input_key")
# 
# 
# 
# # Document search logic
# if query:  # This checks if the query is not empty
#     loader = PyPDFDirectoryLoader(directory_path)
#     docs = loader.load()
#     st.write(f"Number of source materials: {len(docs)}")
#    
#     # Split the documents into smaller chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
#     chunks = text_splitter.split_documents(docs)
#     st.write(f"Number of text chunks: {len(chunks)}")
# 
#     
#     # Initialize and set up the embeddings model
#     model_name = "BAAI/bge-small-en-v1.5"
#     encode_kwargs = {'normalize_embeddings': True}
#     embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'}, encode_kwargs=encode_kwargs)
#     st.write("Embeddings model is set up.")
# 
#     # Here, include logic to process the chunks and generate search results
#     document_search_results = "Processed document search results"
#     st.write(f"Document Search Results: \n\n{document_search_results}")
# 
#     # Display the document search results
#     st.write(f"Document Search Results: \n\n{document_search_results}")
#     
# 
# # Web search logic (assuming you have specific logic for web search)
# if query.strip():  # Reuse the same query
#     # Include logic to process the query for web search and generate results
#     web_search_results = "Processed web search results"
#     st.write(f"Web Search Results: \n\n{web_search_results}")
# 

# # For Document Search:

# if query:  # Checks if the query is not empty
#     # Proceed with document search logic
#     # For example:
#     loader = PyPDFDirectoryLoader(directory_path)
#     docs = loader.load()
#     # Continue with splitting documents, creating embeddings, etc.
# 

# # For Web Search:

# if query:  # Checks if the query is not empty
#     # Proceed with web search logic
#     # For example:
#     result_text = ""
#     for result_url in search(query, num_results=10):
#         response = requests.get(result_url)
#         soup = BeautifulSoup(response.content, 'html.parser')
#         result_text += soup.get_text()
#     # Continue with processing the web search results
# 

# if query:
#     # Your logic to process the query goes here
#     # For example:
#     result = llm(query)
#     st.write(f"Result: \n\n{result}")

# if query:
# # Document search logic
#     loader = PyPDFDirectoryLoader(directory_path)
#     docs = loader.load()
# 
# # Web search logic
# if query:
#     # Your logic to process the query goes here
#     # For example:
#     result = llm(query)
#     st.write(f"Result: \n\n{result}")
# 
# # Display the results
#     st.write(f"Document Search Results: \n\n{document_search_results}")
#     st.write(f"Web Search Results: \n\n{web_search_results}")
# 

# if query:
#     # Your logic to process the query goes here
#     # For example:
#     result = llm(Cohere)
#     st.write(f"Result: \n\n{result}")
# 

# # Cohere Document Search with Langchain

# In[ ]:





# This example shows how to use the Python [langchain](https://python.langchain.com/docs/get_started/introduction) library to run a text-generation request against [Cohere's](https://cohere.com/) API, then augment that request using the text stored in a collection of local PDF documents.
# 
# **Requirements:**
# - You will need an access key to Cohere's API key, which you can sign up for at (https://dashboard.cohere.com/welcome/login). A free trial account will suffice, but will be limited to a small number of requests.
# - After obtaining this key, store it in plain text in your home in directory in the `~/.cohere.key` file.
# - (Optional) Upload some pdf files into the `source_documents` subfolder under this notebook. We have already provided some sample pdfs, but feel free to replace these with your own.

# ## Set up the RAG workflow environment

# In[36]:


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

# In[37]:


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# Make sure other necessary items are in place:

# In[ ]:


try:
    os.environ["COHERE_API_KEY"] =  'KJefdUzquaCHsNJ7SXgJFHZH3Ti132hyxmKMySGU'
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


# print(f"Checking for PDFs in: {os.path.abspath(directory_path)}")
# 

# ## Now send the query to Cohere

# In[ ]:


llm = Cohere()
result = llm(query)
print(f"Result: \n\n{result}")


# In[ ]:


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

# In[30]:


vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Retrieve the most relevant context from the vector store based on the query(No Reranking Applied)
docs = retriever.get_relevant_documents(query)


# Let's see what results it found. Important to note, these results are in the order the retriever thought were the best matches.

# In[31]:


pretty_print_docs(docs)


# These results seem to somewhat match our original query, but we still can't seem to find the information we're looking for. Let's try sending our LLM query again including these results, and see what it comes up with.

# In[32]:


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

# In[33]:


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


# In[34]:


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


def main():

    # Setup the environment
    os.environ["COHERE_API_KEY"] = 'KJefdUzquaCHsNJ7SXgJFHZH3Ti132hyxmKMySGU'

    # Start with making a generation request without RAG augmentation
    query = st.text_input("Enter your query:", "")
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


if __name__ == "__main__":
    main()



# import streamlit as st
# 
# # Use a text input field for the user to enter a query
# query = st.text_input("Enter your query:", "", key="unique_query_input_key")
# 

# if query:
#     # Your logic to process the query goes here
#     # For example:
#     result = llm(query)
#     st.write(f"Result: \n\n{result}")

# if query.strip():  # This checks if the query is not empty or just whitespace
#     result = llm(query)
#     st.write(f"Result: \n\n{result}")
# 

# if query.strip():  # This checks if the query is not empty or just whitespace
#     result = llm(query)
#     st.write(f"Result: \n\n{result}")
# else:
#     st.write("Please enter a valid query.")
# 

# 
