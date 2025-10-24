# Embedding Query

A powerful command-line interface for searching through various embedding collections using ChromaDB and OpenAI's text-embedding-3-large model. This tool enables semantic search across multiple categories including topics, movies, series, persons, companies, networks, and user queries.

## Features

- **Semantic Search**: Leverage OpenAI's text-embedding-3-large model for advanced similarity-based searches
- **Multiple Collections**: Search across different entity types:
  - Topics
  - Movies
  - TV Series
  - Persons
  - Companies
  - Networks
  - Anonymized Queries
- **ChromaDB Integration**: Persistent vector storage with HTTP client support
- **Configurable Settings**: Customize search behavior via configuration file or runtime commands
- **Performance Monitoring**: Track search times and memory usage
- **Docker Support**: Easy deployment with containerization
- **Interactive CLI**: User-friendly command-line interface with collection switching

## Prerequisites

- Python 3.12 or higher
- ChromaDB server (configured via .env file)
- OpenAI API key

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vaugouin/embedding-query.git
cd embedding-query
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root based on `.env.example`:

```bash
# Copy the example file
cp .env.example .env
```

Then edit `.env` with your configuration:

```bash
# OpenAI API Key for embeddings creation
OPENAI_API_KEY=your_openai_api_key_here

# ChromaDB server configuration
CHROMADB_HOST=localhost
CHROMADB_PORT=8000
```

### 4. Configure Settings

Copy the example configuration file:

```bash
cp config.example.ini config.ini
```

Edit `config.ini` to customize your settings:

```ini
[search]
# Number of results to return (default: 1)
n_results = 1

# Similarity threshold for filtering (default: -1, disabled)
# Range: -1 (disabled) or 0.0 to 2.0
# Lower values = stricter matching
similarity_threshold = -1

[list]
# Number of documents to display when listing (default: 50)
document_limit = 50
```

### 5. Configure Database (Optional)

If using database features, copy and configure the secrets file:

```bash
cp citizenphilsecrets.example.py citizenphilsecrets.py
```

Edit `citizenphilsecrets.py` with your database credentials and API keys.

## Usage

### Starting the Application

```bash
python embedding-query.py
```

### Search Commands

The application supports context-aware searches. You can switch between collections and perform searches:

#### Switch Collection and Search

```bash
# Search topics
Enter your topic search query: topic artificial intelligence

# Search movies
Enter your topic search query: movie inception

# Search series
Enter your topic search query: serie breaking bad

# Search persons
Enter your topic search query: person christopher nolan

# Search companies
Enter your topic search query: company warner bros

# Search networks
Enter your topic search query: network netflix

# Search anonymized queries
Enter your topic search query: query what is the best movie
```

#### Continue Searching in Current Collection

Once you've set a collection, subsequent searches will use that collection:

```bash
# After typing "movie inception", the collection is now movies
Enter your movie search query: the matrix
Enter your movie search query: avatar
```

#### List Collection Contents

View the first N documents (configured by `document_limit`) in the current collection:

```bash
Enter your topic search query: topic list
Enter your movie search query: movie list
```

### Runtime Settings

You can modify settings during runtime without restarting the application:

```bash
# Set number of search results
setting search n_result 5

# Set similarity threshold
setting search threshold 0.8

# Set document list limit
setting list limit 100
```

### Exit the Application

```bash
quit
# or
exit
# or
q
```

## Docker Deployment

### Build the Docker Image

```bash
docker build -t embedding-query .
```

### Run the Container

```bash
docker run -it \
  --env-file .env \
  -v $(pwd)/config.ini:/app/config.ini \
  embedding-query
```

The Dockerfile includes:
- Python 3.12 slim base image
- SQLite 3.40.1+ for ChromaDB compatibility
- All required dependencies

## Configuration Reference

### Search Settings

| Setting | Description | Default | Range |
|---------|-------------|---------|-------|
| `n_results` | Number of search results to return | 1 | 1 to ∞ |
| `similarity_threshold` | Maximum distance for filtering results | -1 (disabled) | -1 or 0.0 to 2.0 |

**Similarity Threshold Guide:**
- `-1`: Disabled, show all results
- `0.5`: Very strict, only highly similar results
- `1.0`: Moderate, reasonably similar results
- `1.5`: Lenient, allows less similar results

### List Settings

| Setting | Description | Default | Range |
|---------|-------------|---------|-------|
| `document_limit` | Number of documents to display when listing | 50 | 1 to ∞ |

## Architecture

### Components

- **OpenAI Embedding Function**: Custom implementation of ChromaDB's embedding interface using OpenAI's API
- **ChromaDB Client**: HTTP client connecting to ChromaDB server (configured via CHROMADB_HOST and CHROMADB_PORT environment variables)
- **Collections**: Pre-defined vector collections for different entity types
- **Search Function**: Configurable semantic search with performance tracking
- **List Function**: Collection content viewer with metadata support

### Memory Monitoring

The application displays memory usage on startup:
- Total Memory
- Available Memory
- Used Memory
- Free Memory
- Memory Usage Percentage

## Performance

Search operations include timing information:
- Query execution time
- Number of results found
- Distance scores for each result
- Similarity threshold application (if enabled)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Philippe Vaugouin

## Requirements

See [requirements.txt](requirements.txt) for the complete list of Python dependencies:
- openai >= 1.0.0
- chromadb >= 0.4.0
- pandas >= 1.5.0
- numpy >= 1.24.0
- python-dotenv >= 1.0.0
- psutil >= 5.9.0
- requests
- pytz

## Troubleshooting

### ChromaDB Connection Issues

Ensure ChromaDB server is running and accessible:
```bash
# Check if ChromaDB is accessible (using your configured host/port)
curl http://${CHROMADB_HOST}:${CHROMADB_PORT}/api/v1/heartbeat
```

Verify your ChromaDB configuration in `.env`:
```bash
cat .env | grep CHROMADB
```

### OpenAI API Key Not Found

Verify your `.env` file contains the correct API key:
```bash
cat .env | grep OPENAI_API_KEY
```

### SQLite Version Issues

If running in Docker, the Dockerfile installs SQLite 3.40.1. For local installations, ensure SQLite >= 3.35.0:
```bash
python -c "import sqlite3; print(sqlite3.sqlite_version)"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/vaugouin/embedding-query).
