import pandas as pd
import numpy as np
import openai
import os
from dotenv import load_dotenv
import chromadb
#from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
#pip install psutil
import psutil
#import pymysql.cursors
import citizenphil as cp
from datetime import datetime, timedelta
import time

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Validate that the API key was loaded
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

class OpenAIEmbeddingFunction:
    def __init__(self, model="text-embedding-3-large"):
        self.model = model

    def __call__(self, input):
        """Generate embeddings for a list of texts using OpenAI's embedding model."""
        response = openai.embeddings.create(
            input=input, # Ensure parameter name matches ChromaDB's expectations
            model=self.model
        )
        # Convert to numpy arrays for ChromaDB compatibility
        embeddings = [item.embedding for item in response.data]
        return [np.array(embedding) for embedding in embeddings]
    
    def name(self):
        """Return the name of the embedding function for ChromaDB compatibility."""
        return f"openai_{self.model.replace('-', '_')}"

# Initialize ChromaDB with persistent storage
chroma_client = chromadb.HttpClient(host="localhost", port=8100)

# Initialize ChromaDB with OpenAI's embedding function
embedding_function = OpenAIEmbeddingFunction(model="text-embedding-3-large")

print("ChromaDB initialized with a text-embedding-3-large model.")

# Get the virtual memory details
memory_info = psutil.virtual_memory()
# Print the available memory
print("Démarrage de l'API")
print(f"Total Memory: {memory_info.total / (1024 ** 3):.2f} GB")
print(f"Available Memory: {memory_info.available / (1024 ** 3):.2f} GB")
print(f"Used Memory: {memory_info.used / (1024 ** 3):.2f} GB")
print(f"Free Memory: {memory_info.free / (1024 ** 3):.2f} GB")
print(f"Memory Usage: {memory_info.percent}%")

# Create or load a collection with the custom embedding function
strentitycollection = "topics"
topics = chroma_client.get_or_create_collection(
    name=strentitycollection,
    embedding_function=embedding_function  # Custom embedding model
)
strentitycollection = "movies"
movies = chroma_client.get_or_create_collection(
    name=strentitycollection,
    embedding_function=embedding_function  # Custom embedding model
)
strentitycollection = "series"
series = chroma_client.get_or_create_collection(
    name=strentitycollection,
    embedding_function=embedding_function  # Custom embedding model
)
strentitycollection = "persons"
persons = chroma_client.get_or_create_collection(
    name=strentitycollection,
    embedding_function=embedding_function  # Custom embedding model
)
strentitycollection = "companies"
companies = chroma_client.get_or_create_collection(
    name=strentitycollection,
    embedding_function=embedding_function  # Custom embedding model
)
strentitycollection = "networks"
networks = chroma_client.get_or_create_collection(
    name=strentitycollection,
    embedding_function=embedding_function  # Custom embedding model
)
#Anonymized queries collection
anonymizedqueries = chroma_client.get_or_create_collection(
    name="anonymizedqueries",
    embedding_function=embedding_function  # Custom embedding model
)

def f_searchembeddings(collection, strquery):
    # Start timing the search operation
    start_time = time.time()
    
    # Query ChromaDB using a text-based search
    results = collection.query(
        query_texts=[strquery],  # Query is converted into a vector
        n_results=1
    )
    
    # End timing and calculate duration
    end_time = time.time()
    search_duration = end_time - start_time
    
    # Display detailed information about the first result
    if results["documents"][0]:
        print(f"Query on {collection}: {strquery}")
        print(f"Search time: {search_duration:.4f} seconds")
        print(f"First result ID: {results['ids'][0][0]}")
        print(f"{collection}: {results['documents'][0][0]}")
        print(f"Distance: {results['distances'][0][0]:.4f}")
        #print("---------------")
        print("\n")
    else:
        print(f"No results found for {collection}: {strquery}")

def f_listembeddings(collection):
    # Start timing the search operation
    start_time = time.time()
    
    # List the first 50 documents in the collection and display id and document
    results = collection.get(limit=50)
    
    # End timing and calculate duration
    end_time = time.time()
    search_duration = end_time - start_time
    
    # Display the first 50 documents with their IDs
    if results["documents"]:
        print(f"Listing first 50 documents from {collection}")
        print(f"Retrieval time: {search_duration:.4f} seconds")
        print(f"Total documents retrieved: {len(results['documents'])}")
        print("=" * 50)
        
        for i, (doc_id, document) in enumerate(zip(results['ids'], results['documents'])):
            print(f"ID: {doc_id}")
            print(f"Document: {document}")
            
            # Display metadata if available
            if 'metadatas' in results and results['metadatas'] and i < len(results['metadatas']):
                metadata = results['metadatas'][i]
                if metadata:
                    print(f"Metadata: {metadata}")
            
            print("-" * 30)
        print("\n")
    else:
        print(f"No documents found in {collection}")

"""
for i in range(1, 12):
    strdocid = "companyid_" + str(i) + "_en"
    result = companies.get(ids=[strdocid])
    if result:
        if "documents" in result:
            if result["documents"][0]:
                print(strdocid + " : " + result["documents"][0])
            else:
                print(strdocid + " : Not found")
        else:
            print(strdocid + " : Not found")
    else:
        print(strdocid + " : Not found")
for i in range(136, 140):
    strdocid = "companyid_" + str(i) + "_en"
    result = companies.get(ids=[strdocid])
    if result:
        if "documents" in result:
            if result["documents"][0]:
                print(strdocid + " : " + result["documents"][0])
            else:
                print(strdocid + " : Not found")
        else:
            print(strdocid + " : Not found")
    else:
        print(strdocid + " : Not found")
"""

current_search_type = "topic"  # Default to query search
current_collection = topics
while True:
    strquery = input("\nEnter your " + current_search_type + " search query (or 'quit' to exit): ").strip()
    
    if strquery.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if strquery:
        # Check if the first word is "topic" or "movie"
        words = strquery.split()
        if words and words[0].lower() == "topic":
            current_search_type = "topic"
            current_collection = topics
            # Remove "topic" from the query and call topic search
            strquery = " ".join(words[1:]).strip()
        elif words and words[0].lower() == "movie":
            current_search_type = "movie"
            current_collection = movies
            strquery = " ".join(words[1:]).strip()
        elif words and words[0].lower() == "serie":
            current_search_type = "serie"
            current_collection = series
            strquery = " ".join(words[1:]).strip()
        elif words and words[0].lower() == "person":
            current_search_type = "person"
            current_collection = persons
            strquery = " ".join(words[1:]).strip()
        elif words and words[0].lower() == "company":
            current_search_type = "company"
            current_collection = companies
            strquery = " ".join(words[1:]).strip()
        elif words and words[0].lower() == "network":
            current_search_type = "network"
            current_collection = networks
            strquery = " ".join(words[1:]).strip()
        elif words and words[0].lower() == "query":
            current_search_type = "query"
            current_collection = anonymizedqueries
            strquery = " ".join(words[1:]).strip()
        if strquery:
            if strquery.lower() == "list":
                f_listembeddings(current_collection)
            else:
                f_searchembeddings(current_collection, strquery)
        else:
            print("Please provide a search term after the collection name.")
    else:
        print("Please enter a valid search query.")
