import pandas as pd
import numpy as np
import openai
import os
from dotenv import load_dotenv
import chromadb
#from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
#pip install psutil
import psutil
from datetime import datetime, timedelta
import time
import configparser

# Load environment variables from .env file
load_dotenv()

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Read search settings
SEARCH_N_RESULTS = config.getint('search', 'n_results', fallback=1)
SEARCH_SIMILARITY_THRESHOLD = config.getfloat('search', 'similarity_threshold', fallback=-1)

# Read list settings
LIST_DOCUMENT_LIMIT = config.getint('list', 'document_limit', fallback=50)

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
    
    def embed_query(self, input):
        """Generate embedding for a single query text - required by ChromaDB."""
        # Handle both single string and list inputs
        if isinstance(input, str):
            query_input = [input]
        else:
            query_input = input
            
        response = openai.embeddings.create(
            input=query_input,
            model=self.model
        )
        # Return as a list of numpy arrays (same format as __call__ method)
        embeddings = [item.embedding for item in response.data]
        return [np.array(embedding) for embedding in embeddings]
    
    def name(self):
        """Return the name of the embedding function for ChromaDB compatibility."""
        return f"openai_{self.model.replace('-', '_')}"

# Initialize ChromaDB with persistent storage
chroma_client = chromadb.HttpClient(
    host=os.getenv("CHROMADB_HOST", "localhost"), 
    port=int(os.getenv("CHROMADB_PORT", "8000"))
)

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
        n_results=SEARCH_N_RESULTS
    )

    # End timing and calculate duration
    end_time = time.time()
    search_duration = end_time - start_time

    # Apply similarity threshold filtering if enabled
    filtered_results = {"ids": [[]], "documents": [[]], "distances": [[]]}
    if results["documents"][0]:
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            # Filter by threshold if enabled (threshold >= 0)
            if SEARCH_SIMILARITY_THRESHOLD < 0 or distance <= SEARCH_SIMILARITY_THRESHOLD:
                filtered_results["ids"][0].append(results["ids"][0][i])
                filtered_results["documents"][0].append(results["documents"][0][i])
                filtered_results["distances"][0].append(distance)

    # Display detailed information about the results
    if filtered_results["documents"][0]:
        print(f"Query on {collection}: {strquery}")
        print(f"Search time: {search_duration:.4f} seconds")
        if SEARCH_SIMILARITY_THRESHOLD >= 0:
            print(f"Similarity threshold: {SEARCH_SIMILARITY_THRESHOLD:.4f}")
        print(f"Results found: {len(filtered_results['ids'][0])}")
        print("=" * 50)

        for i in range(len(filtered_results["ids"][0])):
            print(f"Result {i+1}:")
            print(f"ID: {filtered_results['ids'][0][i]}")
            print(f"{collection}: {filtered_results['documents'][0][i]}")
            print(f"Distance: {filtered_results['distances'][0][i]:.4f}")
            print("-" * 30)
        print("\n")
    else:
        print(f"No results found for {collection}: {strquery}")
        if SEARCH_SIMILARITY_THRESHOLD >= 0:
            print(f"(Similarity threshold: {SEARCH_SIMILARITY_THRESHOLD:.4f})")
        print("\n")

def f_listembeddings(collection):
    # Start timing the search operation
    start_time = time.time()

    # List the first N documents in the collection and display id and document
    results = collection.get(limit=LIST_DOCUMENT_LIMIT)

    # End timing and calculate duration
    end_time = time.time()
    search_duration = end_time - start_time

    # Display the documents with their IDs
    if results["documents"]:
        print(f"Listing first {LIST_DOCUMENT_LIMIT} documents from {collection}")
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

def print_available_commands():
    print("\nAvailable commands:")
    print("  topic <search terms>   - search in topics collection")
    print("  movie <search terms>   - search in movies collection")
    print("  serie <search terms>   - search in series collection")
    print("  person <search terms>  - search in persons collection")
    print("  company <search terms> - search in companies collection")
    print("  network <search terms> - search in networks collection")
    print("  query <search terms>   - search in anonymized queries collection")
    print("  list                   - list documents for current collection")
    print("  setting list limit <value>")
    print("  setting search n_result <value>")
    print("  setting search threshold <value>")
    print("  setting display        - show current settings")
    print("  quit / exit / q        - exit the program")
    print("  help                   - show this help message")

def print_current_settings():
    print("\nCurrent settings:")
    print(f"  LIST_DOCUMENT_LIMIT        = {LIST_DOCUMENT_LIMIT}")
    print(f"  SEARCH_N_RESULTS           = {SEARCH_N_RESULTS}")
    print(f"  SEARCH_SIMILARITY_THRESHOLD = {SEARCH_SIMILARITY_THRESHOLD}")

print_available_commands()

while True:
    strquery = input("\nEnter your " + current_search_type + " search query (or 'quit' to exit): ").strip()

    if strquery.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break

    if strquery:
        # Check if the first word is "topic" or "movie"
        words = strquery.split()
        if words and words[0].lower() == "setting":
            # Handle runtime settings changes
            if len(words) >= 2 and words[1].lower() == "display":
                print_current_settings()
            elif len(words) >= 4:
                setting_category = words[1].lower()
                setting_name = words[2].lower()

                if setting_category == "list" and setting_name == "limit":
                    try:
                        new_value = int(words[3])
                        if new_value > 0:
                            LIST_DOCUMENT_LIMIT = new_value
                            print(f"LIST_DOCUMENT_LIMIT updated to {LIST_DOCUMENT_LIMIT}")
                        else:
                            print("Error: limit value must be positive")
                    except ValueError:
                        print("Error: invalid limit value, must be an integer")
                elif setting_category == "search" and setting_name == "n_result":
                    try:
                        new_value = int(words[3])
                        if new_value > 0:
                            SEARCH_N_RESULTS = new_value
                            print(f"SEARCH_N_RESULTS updated to {SEARCH_N_RESULTS}")
                        else:
                            print("Error: n_result value must be positive")
                    except ValueError:
                        print("Error: invalid n_result value, must be an integer")
                elif setting_category == "search" and setting_name == "threshold":
                    try:
                        new_value = float(words[3])
                        SEARCH_SIMILARITY_THRESHOLD = new_value
                        print(f"SEARCH_SIMILARITY_THRESHOLD updated to {SEARCH_SIMILARITY_THRESHOLD}")
                    except ValueError:
                        print("Error: invalid threshold value, must be a number")
                else:
                    print(f"Error: unknown setting '{setting_category} {setting_name}'")
                    print("Valid settings:")
                    print("  - setting list limit <value>")
                    print("  - setting search n_result <value>")
                    print("  - setting search threshold <value>")
            else:
                print("Error: invalid setting command format")
                print("Usage:")
                print("  - setting list limit <value>")
                print("  - setting search n_result <value>")
                print("  - setting search threshold <value>")
            continue
        elif words and words[0].lower() == "help":
            print_available_commands()
            continue
        elif words and words[0].lower() == "topic":
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
