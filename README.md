# TransitMate: Translink Service Update RAG Chatbot

An intelligent RAG (Retrieval-Augmented Generation) chatbot designed to provide real-time updates and answer questions regarding Translink service disruptions, track closures, and stop changes.

## Motivation

Translink frequently updates its service status through website notices via https://translink.com.au/service-updates and an RSS feed at https://translink.com.au/service-updates/rss. However, users often find it challenging to quickly locate relevant information about specific routes or disruptions. This project aims to streamline access to this information by leveraging advanced retrieval techniques and local LLMs to deliver accurate, context-aware responses.

## Features

- **Real-time Ingestion**: Automatically scrapes the Translink RSS feed for the latest service updates.
- **RAG Capability**: Uses advanced retrieval techniques (BM25 + Semantic Search) to find relevant disruption info.
- **LLM Integration**: Powered by local LLMs via Ollama (defaulting to `qwen3:8b`) for professional, human-like responses.
- **Structured Data Extraction**: Automatically extracts routes, locations, and dates from raw transport notices using LLM-based parsing.
- **Responsive UI**: A modern frontend built with HTML/CSS/JS.

## Tech Stack

- **Backend**: Python, Flask, Flask-CORS
- **RAG Architecture**:
  - `sentence-transformers` for embeddings.
  - `rank_bm25` for keyword retrieval.
  - `ollama` for LLM orchestration.
- **Scraping**: `BeautifulSoup4`, `feedparser`.
- **Frontend**: Pure HTML5, CSS3 (Vanilla), JavaScript (ES6+).

## Example
<style>
  figcaption {
    font-size: 18px;
    color: #eee;
    margin-top: 8px;
  }
</style>

<figure style="text-align: center;">
    <img src="./assets/frontend.png" style="width: 70%; height: auto;">
    <figcaption>
        Chatbot UI
    </figcaption>
</figure>
<br>
<div style="display: flex; justify-content: space-between; align-items: flex-end; gap: 10px;">

  <!-- LEFT COLUMN -->
  <figure style="text-align: center; width: 50%; margin: 0;">
    <img src="./assets/Example1.png" style="width: 100%; height: auto;">
    <img src="./assets/Example1TSA.png" style="width: 100%; height: auto; margin-top: 6px;">
    <figcaption style="font-size: 14px; color: #eee; margin-top: 8px;">
      Location Name
    </figcaption>
  </figure>

  <!-- RIGHT COLUMN -->
  <figure style="text-align: center; width: 50%; margin: 0;">
    <img src="./assets/Example2.png" style="width: 100%; height: auto;">
    <img src="./assets/Example2TSA.png" style="width: 100%; height: auto; margin-top: 6px;">
    <figcaption style="font-size: 14px; color: #eee; margin-top: 8px;">
      Route Name
    </figcaption>
  </figure>

</div>
<br>
<div style="display: flex; justify-content: space-between; align-items: flex-end; gap: 10px;">

  <!-- LEFT COLUMN -->
  <figure style="text-align: center; width: 50%; margin: 0;">
    <img src="./assets/Example3.png" style="width: 100%; height: auto;">
    <img src="./assets/Example3TSA.png" style="width: 100%; height: auto; margin-top: 6px;">
    <figcaption style="font-size: 14px; color: #eee; margin-top: 8px;">
      Stop Example
    </figcaption>
  </figure>

  <!-- RIGHT COLUMN -->
  <figure style="text-align: center; width: 50%; margin: 0;">
    <img src="./assets/Example4.png" style="width: 100%; height: auto;">
    <img src="./assets/Example4TSA.png" style="width: 100%; height: auto; margin-top: 6px;">
    <figcaption style="font-size: 14px; color: #eee; margin-top: 8px;">
      Event Impact
    </figcaption>
  </figure>

</div>




## Algorithm Overview
```mermaid
flowchart LR
    %% Ingestion Pipeline
    subgraph INGESTION_PIPELINE["Ingestion Pipeline"]
        RSS["RSS Feed"]
        SCRAPE["Web Scraping"]
        PARSE["LLM Parsing"]
        EMBED["Embed & Index"]
        DB[(JSON Database)]

        RSS --> SCRAPE
        SCRAPE --> PARSE
        PARSE --> EMBED
        EMBED --> DB
    end

    %% Query Pipeline
    subgraph QUERY_PIPELINE["Query Pipeline"]
        QUERY["User Query"]
        TEMP["Temporal Extraction"]
        HYBRID["Hybrid Retrieval"]
        VECTOR["Vector Search"]
        BM25["BM25 Search"]
        FILTER["Temporal Filter"]
        LLM["Context + LLM"]
        RESPONSE["Response"]

        QUERY --> TEMP
        TEMP --> HYBRID
        HYBRID --> VECTOR
        HYBRID --> BM25
        VECTOR --> FILTER
        BM25 --> FILTER
        FILTER --> LLM
        LLM --> RESPONSE
    end

    %% Cross links
    DB --> VECTOR
    DB --> BM25
```

## Getting Started

### Prerequisites

1.  **Python 3.10+**
2.  **Ollama**: Install [Ollama](https://ollama.ai/) and pull the required model:
    ```bash
    ollama pull qwen3:8b
    ```

### Installation

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/sanjeevbhurtyal/Translink_Service_Update_RAG_Chatbot.git
    cd Translink_Service_Update_RAG_Chatbot
    ```

2.  **Set up a virtual environment**:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Ingest Service Data**:
    Run the ingestion script to fetch and process current Translink service updates.

    ```bash
    python backend/ingest.py
    ```

    _Use `--fresh-start` to clear the database and start fresh._

2.  **Run the Server**:
    Start the Flask backend server.

    ```bash
    python backend/app.py
    ```

3.  **Launch the Frontend**:
    Open `frontend/index.html` in your favorite web browser.

## Project Structure

```text
├── backend/
│   ├── app.py              # Flask API server
│   ├── ingest.py           # RSS scraper and data processing
│   ├── rag.py              # RAG logic (retrieval + LLM)
│   ├── documents.json      # Processed textual data
│   └── vector_store.json   # Local vector database
├── frontend/
│   └── index.html          # Chat interface
├── requirements.txt        # Python dependencies
└── README.md
```

## Limitations

- **Hardware Dependency**: Local LLM performance (Ollama) is highly dependent on system CPU/GPU capabilities.
- **Scraping Sensitivity**: The ingestion pipeline relies on the current structure of the Translink RSS feed and website; changes to their HTML may require scraper updates.
- **Local Storage**: Currently uses JSON-based storage for the vector database and documents, which is suitable for moderate data but not optimized for massive scales.
- **Development Server**: The Flask backend is configured as a development server and should be wrapped in a production-ready WSGI (like Gunicorn) for deployment.

## Future Enhancements

- Add support for Multi-modal retrieval (Map images).

## License

Distributed under the MIT License. See `LICENSE` for more information.
