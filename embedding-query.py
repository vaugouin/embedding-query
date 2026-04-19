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
from rapidfuzz.distance import Levenshtein

def levenshtein_distance(a: str, b: str) -> int:
    return int(Levenshtein.distance(a or "", b or ""))
    
# Load environment variables from .env file
load_dotenv()

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Read search settings
SEARCH_N_RESULTS = config.getint('search', 'n_results', fallback=1)
SEARCH_SIMILARITY_THRESHOLD = config.getfloat('search', 'similarity_threshold', fallback=-1)

# Read ls settings
LIST_DOCUMENT_LIMIT = config.getint('ls', 'document_limit', fallback=50)

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

# Create or load entity collections with the custom embedding function
CHROMADB_COLLECTIONS_BY_NAME = {
    name: chroma_client.get_or_create_collection(name=name, embedding_function=embedding_function)
    for name in [
        "persons",
        "movies",
        "series",
        "companies",
        "networks",
        "topics",
        "locations",
        "groups",
        "characters",
        "lists",
        "collections",
        "deaths",
        "awards",
        "nominations",
        "movements",
    ]
}

topics = CHROMADB_COLLECTIONS_BY_NAME["topics"]
movies = CHROMADB_COLLECTIONS_BY_NAME["movies"]
series = CHROMADB_COLLECTIONS_BY_NAME["series"]
persons = CHROMADB_COLLECTIONS_BY_NAME["persons"]
companies = CHROMADB_COLLECTIONS_BY_NAME["companies"]
networks = CHROMADB_COLLECTIONS_BY_NAME["networks"]
characters = CHROMADB_COLLECTIONS_BY_NAME["characters"]
groups = CHROMADB_COLLECTIONS_BY_NAME["groups"]
locations = CHROMADB_COLLECTIONS_BY_NAME["locations"]
lists = CHROMADB_COLLECTIONS_BY_NAME["lists"]
collections = CHROMADB_COLLECTIONS_BY_NAME["collections"]
deaths = CHROMADB_COLLECTIONS_BY_NAME["deaths"]
awards = CHROMADB_COLLECTIONS_BY_NAME["awards"]
nominations = CHROMADB_COLLECTIONS_BY_NAME["nominations"]
movements = CHROMADB_COLLECTIONS_BY_NAME["movements"]

#Anonymized queries collection
strentitycollection = "anonymizedqueries"
anonymizedqueries = chroma_client.get_or_create_collection(
    name=strentitycollection,
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
            document = filtered_results["documents"][0][i]
            lev = levenshtein_distance(strquery, document)
            print(f"{collection}: {document} [Levenshtein={lev}]")
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
    print("  location <search terms> - search in locations collection")
    print("  character <search terms> - search in characters collection")
    print("  group <search terms>   - search in groups collection")
    print("  list <search terms>    - search in lists collection")
    print("  collection <search terms> - search in collections collection")
    print("  death <search terms>   - search in deaths collection")
    print("  award <search terms>   - search in awards collection")
    print("  nomination <search terms> - search in nominations collection")
    print("  movement <search terms> - search in movements collection")
    print("  query <search terms>   - search in anonymized queries collection")
    print("  ls                     - list documents for current collection")
    print("  setting collections    - list available collections")
    print("  setting ls limit <value>")
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

def print_available_collections():
    def _safe_str(value):
        try:
            return str(value)
        except Exception:
            return repr(value)

    def _safe_get_collection(name):
        try:
            return chroma_client.get_collection(name=name, embedding_function=embedding_function)
        except Exception:
            try:
                return chroma_client.get_collection(name=name)
            except Exception:
                return None

    def _extract_metadata(collection_obj):
        metadata = None
        for attr in ("metadata", "metadatas"):
            if hasattr(collection_obj, attr):
                try:
                    metadata = getattr(collection_obj, attr)
                    break
                except Exception:
                    pass
        if callable(metadata):
            try:
                metadata = metadata()
            except Exception:
                metadata = None
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            try:
                metadata = dict(metadata)
            except Exception:
                metadata = {"raw": _safe_str(metadata)}
        return metadata

    def _extract_other_info(collection_obj):
        info = {}
        for attr in ("id", "name", "tenant", "database"):
            if hasattr(collection_obj, attr):
                try:
                    info[attr] = getattr(collection_obj, attr)
                except Exception:
                    pass
        try:
            d = getattr(collection_obj, "__dict__", {})
            for k, v in d.items():
                if k in ("_embedding_function", "embedding_function"):
                    continue
                if k in info:
                    continue
                if isinstance(v, (str, int, float, bool)) or v is None:
                    info[k] = v
        except Exception:
            pass
        return info

    try:
        collections = chroma_client.list_collections()
    except Exception as e:
        print(f"Error listing collections: {_safe_str(e)}")
        return

    if not collections:
        print("No collections found in ChromaDB.")
        return

    print("\nAvailable ChromaDB collections:")
    print("=" * 50)

    for entry in collections:
        if isinstance(entry, str):
            name = entry
            collection_obj = _safe_get_collection(name)
        else:
            name = getattr(entry, "name", None) or _safe_str(entry)
            collection_obj = entry

        if collection_obj is None:
            print(f"Collection: {name}")
            print("  Error: unable to load collection details")
            print("-" * 30)
            continue

        metadata = _extract_metadata(collection_obj)

        try:
            count = collection_obj.count()
        except Exception:
            count = "unknown"

        index_used = metadata.get("index") or metadata.get("index_type") or metadata.get("chroma:index")
        if index_used is None:
            index_used = "hnsw" if any(k.startswith("hnsw:") for k in metadata.keys()) else "unknown"

        distance_function = (
            metadata.get("hnsw:space")
            or metadata.get("distance")
            or metadata.get("distance_function")
            or metadata.get("metric")
            or "unknown"
        )

        print(f"Collection: {name}")
        print(f"  Index: {index_used}")
        print(f"  Distance: {distance_function}")
        print(f"  Document count: {count}")
        print(f"  Metadata: {metadata}")

        other_info = _extract_other_info(collection_obj)
        if other_info:
            print(f"  Other info: {other_info}")

        print("-" * 30)

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
            elif len(words) >= 2 and words[1].lower() == "collections":
                print_available_collections()
            elif len(words) >= 4:
                setting_category = words[1].lower()
                setting_name = words[2].lower()

                if setting_category == "ls" and setting_name == "limit":
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
                    print("  - setting collections")
                    print("  - setting ls limit <value>")
                    print("  - setting search n_result <value>")
                    print("  - setting search threshold <value>")
            else:
                print("Error: invalid setting command format")
                print("Usage:")
                print("  - setting collections")
                print("  - setting ls limit <value>")
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
        elif words and words[0].lower() == "location":
            current_search_type = "location"
            current_collection = locations
            strquery = " ".join(words[1:]).strip()
        elif words and words[0].lower() == "character":
            current_search_type = "character"
            current_collection = characters
            strquery = " ".join(words[1:]).strip()
        elif words and words[0].lower() == "group":
            current_search_type = "group"
            current_collection = groups
            strquery = " ".join(words[1:]).strip()
        elif words and words[0].lower() == "list":
            current_search_type = "list"
            current_collection = lists
            strquery = " ".join(words[1:]).strip()
        elif words and words[0].lower() == "collection":
            current_search_type = "collection"
            current_collection = collections
            strquery = " ".join(words[1:]).strip()
        elif words and words[0].lower() == "death":
            current_search_type = "death"
            current_collection = deaths
            strquery = " ".join(words[1:]).strip()
        elif words and words[0].lower() == "award":
            current_search_type = "award"
            current_collection = awards
            strquery = " ".join(words[1:]).strip()
        elif words and words[0].lower() == "nomination":
            current_search_type = "nomination"
            current_collection = nominations
            strquery = " ".join(words[1:]).strip()
        elif words and words[0].lower() == "movement":
            current_search_type = "movement"
            current_collection = movements
            strquery = " ".join(words[1:]).strip()
        elif words and words[0].lower() == "query":
            current_search_type = "query"
            current_collection = anonymizedqueries
            strquery = " ".join(words[1:]).strip()
        if strquery:
            if strquery.lower() == "ls":
                f_listembeddings(current_collection)
            else:
                f_searchembeddings(current_collection, strquery)
        else:
            print("Please provide a search term after the collection name.")
    else:
        print("Please enter a valid search query.")
