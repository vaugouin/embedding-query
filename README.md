# Embedding Query

A powerful command-line interface for searching through various embedding collections using ChromaDB and OpenAI's text-embedding-3-large model. This tool enables semantic search across multiple categories including topics, movies, series, persons, companies, networks, and user queries.

## Features

- **Semantic Search**: Leverage OpenAI's text-embedding-3-large model for advanced similarity-based searches
- **Multiple Collections**: Search across different entity types:
  - Topics
  - Locations
  - Movies
  - TV Series
  - Persons
  - Companies
  - Networks
  - Characters
  - Groups
  - Lists
  - Collections
  - Deaths
  - Awards
  - Nominations
  - Movements
  - Anonymized Queries
- **ChromaDB Integration**: Persistent vector storage with HTTP client support
- **Configurable Settings**: Customize search behavior via configuration file or runtime commands
- **Performance Monitoring**: Track search times and memory usage
- **Compact Results Display**: Search results are displayed as a single aligned table (one row per result)
- **Docker Support**: Easy deployment with containerization
- **Interactive CLI**: User-friendly command-line interface with collection switching

## Prerequisites

- Python 3.12 or higher
- ChromaDB server
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

When running locally, you can use a `.env` file for convenience (do not commit it):

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

[ls]
# Number of documents to display when listing (default: 50)
document_limit = 50
```

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

# Search locations
Enter your topic search query: location paris

# Search characters
Enter your topic search query: character walter white

# Search groups
Enter your topic search query: group the avengers

# Search lists
Enter your topic search query: list best sci fi

# Search collections
Enter your topic search query: collection imdb_top_250

# Search deaths
Enter your topic search query: death ned stark

# Search awards
Enter your topic search query: award academy awards

# Search nominations
Enter your topic search query: nomination best picture

# Search movements
Enter your topic search query: movement french new wave

# Search anonymized queries
Enter your topic search query: query what is the best movie
```

### Search Results Output

Search results are displayed in a compact table:

- A **header row** is printed with aligned columns
- Each result is printed as a **single row**
- The long overview/summary text is not printed (only the title part is shown)

Columns:

- `#` (result index)
- `ID`
- `Title`
- `Distance` (ChromaDB distance)
- `Levenshtein` (Levenshtein distance between your query and the extracted title)

#### Continue Searching in Current Collection

Once you've set a collection, subsequent searches will use that collection:

Use `ls` to list documents in the current collection:

```bash
Enter your topic search query: ls
```

Modify runtime settings without restarting:

```bash
setting search n_result 5
setting search threshold 0.8
setting ls limit 100
setting collections
setting display
```

### Collections Listing Output

The `setting collections` command prints a compact table (one row per collection) with aligned columns:

- `Name`
- `Count`
- `Index`
- `Distance`

## Docker Deployment

Never bake secrets into the Docker image. Do not copy `.env` into the image.

The repository includes a `.dockerignore` that excludes `.env` files from the Docker build context.

At runtime, pass secrets from a host-managed env file using Docker's `--env-file` option. The env file should live outside the app source tree, for example:

`/home/debian/docker/embedding-query/.env`

```bash
docker build -t embedding-query .
docker run -it --rm --network="host" \
  --env-file /home/debian/docker/embedding-query/.env \
  -v $(pwd)/config.ini:/app/config.ini \
  embedding-query
```

## Configuration Reference

Search settings:

| Setting | Description | Default |
|---------|-------------|---------|
| `n_results` | Number of search results to return | 1 |
| `similarity_threshold` | Maximum distance for filtering results (`-1` disables) | -1 |

List settings:

| Setting | Description | Default |
|---------|-------------|---------|
| `document_limit` | Number of documents to display when listing | 50 |

## Architecture

- **OpenAI Embedding Function**: Integrates OpenAI embeddings into ChromaDB-compatible interface.
- **ChromaDB Client**: HTTP client connecting to a ChromaDB server.
- **Collections & Search**: Pre-defined collections and a configurable search pipeline.

## Troubleshooting

Check ChromaDB connectivity:

```bash
curl http://${CHROMADB_HOST}:${CHROMADB_PORT}/api/v1/heartbeat
```

Check that `OPENAI_API_KEY` is present in your environment.

## Requirements

See [requirements.txt](requirements.txt) for dependencies.

## Contributing

Contributions welcome — open a Pull Request.

## Support

For issues, visit the [GitHub repository](https://github.com/vaugouin/embedding-query).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Philippe Vaugouin
