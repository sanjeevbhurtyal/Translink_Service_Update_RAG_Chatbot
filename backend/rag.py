import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import json
import os
from datetime import datetime

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "vector_store.json")

# Initialize Embedding Model
# all-MiniLM-L6-v2 is fast, easy to install, and good for this use case
embedding_model = SentenceTransformer('backend/models/all-MiniLM-L6-v2')

# In-memory store
# Structure: {'ids': [], 'documents': [], 'metadatas': [], 'embeddings': []}
vector_store = {
    'ids': [],
    'documents': [],
    'metadatas': [],
    'embeddings': []
}

def load_store():
    global vector_store
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            data = json.load(f)
            vector_store['ids'] = data.get('ids', [])
            vector_store['documents'] = data.get('documents', [])
            vector_store['metadatas'] = data.get('metadatas', [])
            # Embeddings are stored as list of lists in JSON, convert back to numpy
            if data.get('embeddings'):
                vector_store['embeddings'] = np.array(data['embeddings'], dtype='float32')
            else:
                vector_store['embeddings'] = []
        print(f"Loaded {len(vector_store['ids'])} documents from store.")

def save_store():
    data = {
        'ids': vector_store['ids'],
        'documents': vector_store['documents'],
        'metadatas': vector_store['metadatas'],
        'embeddings': vector_store['embeddings'].tolist() if len(vector_store['embeddings']) > 0 else []
    }
    with open(DB_FILE, 'w') as f:
        json.dump(data, f)

def add_documents(documents):
    global vector_store
    
    new_texts = []
    new_ids = []
    new_metas = []
    
    for doc in documents:
        if doc['id'] not in vector_store['ids']:
            new_texts.append(doc['text'])
            new_ids.append(doc['id'])
            new_metas.append(doc['metadata'])
    
    if not new_texts:
        print("No new documents to add.")
        return

    print(f"Embedding {len(new_texts)} new documents...")
    embeddings = embedding_model.encode(new_texts)
    
    # Update store
    vector_store['ids'].extend(new_ids)
    vector_store['documents'].extend(new_texts)
    vector_store['metadatas'].extend(new_metas)
    
    if len(vector_store['embeddings']) == 0:
        vector_store['embeddings'] = embeddings
    else:
        vector_store['embeddings'] = np.vstack([vector_store['embeddings'], embeddings])
        
    save_store()
    print("Store updated and saved.")

def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

def format_dates_for_bm25(date_str):
    """
    Expand a canonical ISO date (YYYY-MM-DD) into
    multiple lexical variants for BM25 indexing.
    """

    if not date_str:
        return []

    try:
        # STRICT: only accept ISO
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return []

    day = dt.day
    day_0 = f"{day:02d}"
    month = dt.month
    month_0 = f"{month:02d}"
    year = dt.year

    day_ord = ordinal(day)           # 2nd
    month_name = dt.strftime("%B")   # February
    month_abbr = dt.strftime("%b")   # Feb

    variants = set()

    # --- ISO ---
    variants.add(f"{year}-{month_0}-{day_0}")

    # --- Numeric formats ---
    variants.update({
        f"{day}/{month}/{year}",
        f"{day_0}/{month_0}/{year}",
        f"{day}-{month}-{year}",
        f"{day_0}-{month_0}-{year}",
        f"{month}/{day}/{year}",
        f"{month_0}/{day_0}/{year}",
        f"{month}-{day}-{year}",
        f"{month_0}-{day_0}-{year}",
    })

    # --- Text month formats ---
    variants.update({
        f"{day} {month_name} {year}",
        f"{day_ord} {month_name} {year}",
        f"{day} {month_abbr} {year}",
        f"{day_ord} {month_abbr} {year}",
        f"{month_name} {day} {year}",
        f"{month_name} {day_ord} {year}",
        f"{month_abbr} {day} {year}",
        f"{month_abbr} {day_ord} {year}",
    })

    return sorted(variants)


# Hybrid Search global variables
bm25 = None
bm25_corpus = []


def init_bm25():
    """
    Initialize BM25 index from current vector_store documents.
    Creates a searchable corpus that includes title, text, and key metadata.
    """
    global bm25, bm25_corpus, bm25_doc_ids
    from rank_bm25 import BM25Okapi
    
    if not vector_store or 'documents' not in vector_store:
        print("Vector store not initialized")
        return
    
    documents = vector_store['documents'] 
    metadatas = vector_store.get('metadatas', [])
    ids = vector_store.get('ids', [])
    
    if not documents:
        print("No documents to index")
        return
    
    # Create corpus for BM25 - combine text with metadata
    bm25_corpus = []
    bm25_doc_ids = []
    
    for i, doc_text in enumerate(documents):
        # Get metadata for this document
        metadata = metadatas[i] if i < len(metadatas) else {}
        doc_id = ids[i] if i < len(ids) else i
        
        routes = metadata.get('routes', [])
        # Ensure routes is always a list
        if isinstance(routes, list):
            routes_text = ' '.join(routes)
        else:
            routes_text = str(routes)
        # Build searchable text combining multiple fields
        searchable_parts = [
            doc_text,
            metadata.get('title', ''),
            metadata.get('category', ''),
            metadata.get('services', ''),
            metadata.get('start_date', ''),
            metadata.get('end_date', ''),
            metadata.get('locations', ''),
            routes_text
        ]

        # Add date information
        if metadata.get('start_date'):
            searchable_parts.extend(format_dates_for_bm25(metadata['start_date_iso']))
        if metadata.get('end_date'):
            searchable_parts.extend(format_dates_for_bm25(metadata['end_date_iso']))

        # Combine all parts and tokenize
        full_searchable_text = ' '.join(str(p) for p in searchable_parts if p)
        tokenized = full_searchable_text.lower().split()
        
        bm25_corpus.append(tokenized)
        bm25_doc_ids.append(doc_id)
    
    # Initialize BM25
    bm25 = BM25Okapi(bm25_corpus)
    print(f"BM25 Index initialized with {len(bm25_corpus)} documents.")

def extract_query_date(query_text):
    import re
    from datetime import datetime, timedelta

    query_lower = query_text.lower().strip()
    today = datetime.now().date()

    # Temporal keywords
    if any(word in query_lower for word in ['today', 'right now', 'currently', 'current']):
        return today
    if 'tomorrow' in query_lower:
        return today + timedelta(days=1)
    if 'yesterday' in query_lower:
        return today - timedelta(days=1)
    if 'this week' in query_lower or 'this weekend' in query_lower:
        return today

    # Month mapping
    month_map = {
        'jan':1,'january':1,'feb':2,'february':2,'mar':3,'march':3,
        'apr':4,'april':4,'may':5,'jun':6,'june':6,'jul':7,'july':7,
        'aug':8,'august':8,'sep':9,'sept':9,'september':9,
        'oct':10,'october':10,'nov':11,'november':11,'dec':12,'december':12
    }
    month_pattern = '|'.join(month_map.keys())

    # 1. YYYY-MM-DD
    match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', query_lower)
    if match:
        y,m,d = match.groups()
        try: return datetime(int(y),int(m),int(d))
        except: pass

    # 2. DD/MM/YYYY or DD-MM-YYYY
    match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', query_lower)
    if match:
        d,m,y = match.groups()
        try: return datetime(int(y),int(m),int(d))
        except: pass

    # 3. DD Month YYYY (26 May 2026, 26th May 2026)
    match = re.search(rf'\b(\d{{1,2}})(?:st|nd|rd|th)?\b\s+({month_pattern})\b\s+(\d{{4}})', query_lower)
    if match:
        d,month_str,y = match.groups()
        m = month_map.get(month_str.lower())
        try: return datetime(int(y), m, int(d))
        except: pass

    # 4. Month DD, YYYY (May 26, 2026)
    match = re.search(rf'\b({month_pattern})\b\s+(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(\d{{4}})', query_lower)
    if match:
        month_str,d,y = match.groups()
        m = month_map.get(month_str.lower())
        try: return datetime(int(y), m, int(d))
        except: pass

    # 5. DD Month (no year)
    match = re.search(rf'\b(\d{{1,2}})(?:st|nd|rd|th)?\b\s+({month_pattern})\b', query_lower)
    if match:
        d,month_str = match.groups()
        m = month_map.get(month_str.lower())
        try:
            dt = datetime(today.year, m, int(d))
            if dt < today:
                dt = datetime(today.year+1, m, int(d))
            return dt
        except: pass

    # 6. Month DD (no year)
    match = re.search(rf'\b({month_pattern})\b\s+(\d{{1,2}})(?:st|nd|rd|th)?\b', query_lower)
    if match:
        month_str,d = match.groups()
        m = month_map.get(month_str.lower())
        try:
            dt = datetime(today.year, m, int(d))
            if dt < today:
                dt = datetime(today.year+1, m, int(d))
            return dt
        except: pass

    # 7. Month (no day, no year)
    match = re.search(rf'\b({month_pattern})\b', query_lower)
    if match:
        month_str = match.group()
        m = month_map.get(month_str.lower())
        try:
            dt = datetime(today.year, m, 1)
            if dt < today:
                dt = datetime(today.year+1, m, 1)
            return dt
        except: pass

    # fallback
    return None


def is_document_active_on_date(metadata, target_date):
    try:
        start = datetime.strptime(metadata["start_date_iso"], "%Y-%m-%d").date()

        end = datetime.strptime(
            metadata.get("end_date_iso") or metadata["start_date_iso"],
            "%Y-%m-%d").date()
        
        if start <= target_date.date() <= end:
            status = "ACTIVE"
        elif start > target_date.date():
            status = "UPCOMING"
        elif end < target_date.date():
            status = "ENDED"
        else:
            status = "UNKNOWN"
        return status

    except Exception as e:
        print(f"   ERROR checking document '{metadata.get('title', 'Unknown')}': {e}")
        return "UNKNOWN"



def query_rag(query_text, n_results=10, model="qwen3:8b"):
    # If store is empty in memory, try reloading from disk
    if len(vector_store['ids']) == 0:
        load_store()
        
    if len(vector_store['ids']) == 0:
        return "No information available (Database empty).", []

    # Extract temporal information from query
    target_date = extract_query_date(query_text)
    query_target_date = target_date
    
    # If target date is not found, default to today for temporal filtering
    if not target_date:
        target_date = datetime.now().date()
    
    # Dense Search (Vector)
    query_embedding = embedding_model.encode([query_text])[0]
    docs_embeddings = vector_store['embeddings']
    
    norm_query = np.linalg.norm(query_embedding)
    norm_docs = np.linalg.norm(docs_embeddings, axis=1)
    norm_docs[norm_docs == 0] = 1e-10
    
    dense_scores = np.dot(docs_embeddings, query_embedding) / (norm_docs * norm_query)
    
    # Sparse Search (BM25)
    global bm25
    if bm25 is None:
        init_bm25()
        
    sparse_scores = np.zeros(len(vector_store['ids']))
    if bm25:
        tokenized_query = query_text.lower().split()
        sparse_scores = np.array(bm25.get_scores(tokenized_query))
            
    # Normalize Dense
    dense_scores = np.clip(dense_scores, 0, 1)
    
    # Normalize Sparse
    if sparse_scores.max() > 0:
        sparse_scores = sparse_scores / sparse_scores.max()
        
    # Weighting
    if len(query_text.split()) < 4:
        alpha = 0.3  # Short queries, rely more on sparse
    else:
        alpha = 0.7  # Longer queries, dense dominates
    final_scores = (alpha * dense_scores) + ((1 - alpha) * sparse_scores)
    
    # Apply Temporal Filtering and Boosting BEFORE selecting top results
    if query_target_date:        
        # Create mask for temporally relevant documents
        temporal_relevant = np.zeros(len(vector_store['ids']), dtype=bool)
        temporal_boost = np.ones(len(vector_store['ids']))
        
        for i, metadata in enumerate(vector_store['metadatas']):
            status = is_document_active_on_date(metadata, target_date)

            if status == "ACTIVE":
                temporal_relevant[i] = True
                temporal_boost[i] = 5   # strong boost
            elif status == "UPCOMING":
                temporal_boost[i] = 3    # mild boost
            else:
                temporal_boost[i] = 1   # past disruptions get no boost
        
        # Apply temporal boost to scores
        final_scores = final_scores * temporal_boost
        
        # If we have temporally relevant documents, prioritize them
        if np.any(temporal_relevant):
            # Set non-relevant documents to very low scores if temporal query detected
            # This ensures temporally relevant docs always come first
            final_scores[~temporal_relevant] = final_scores[~temporal_relevant] * 0.1
    
    # Get top N indices AFTER temporal boosting
    k = min(n_results, len(vector_store['ids']))
    top_k_indices = np.argsort(final_scores)[-k:][::-1]
    
    # Gather results
    retrieved_docs = [vector_store['documents'][i] for i in top_k_indices]
    retrieved_metas = [vector_store['metadatas'][i] for i in top_k_indices]

    # Context Construction
    context_blocks = []

    for doc, meta in zip(retrieved_docs, retrieved_metas):
        status = is_document_active_on_date(meta, target_date)
        block = (
            f"Title: {meta.get('title', 'Unknown')}\n"
            f"Dates: {meta.get('start_date_iso')} to {meta.get('end_date_iso') or meta.get('start_date_iso')}\n"
            f"URL: {meta.get('url', '')}\n"
            f"Details:\n{doc}\n"
            f"Status: {status}\n"
            f"Category: {meta.get('category', '')}\n"
        )
        context_blocks.append(block)

    context_str = "\n\n---\n\n".join(context_blocks)
    
    # System Prompt
    system_prompt = (
        "You are a friendly and helpful Translink Assistant. Your job is to explain public transport service updates to customers in a clear, calm, and human way. You are only capable of providing Translink Serivice Updates\n"
        "Answer the user's question using only the information you know.\n"
        "You must not mention, imply, or reference any external documents, sources, or context.\n"
        "Treat all provided information as if it is part of your own knowledge.\n"
        "You must evaluate each service update before mentioning it and report all relevant active or upcoming impacts."
        "If the user does not specify a date, always include today’s and upcoming service updates only.\n"
        "Metro is not a train line. It is a separate route similar to Bus, Train, Tram and Ferry."
        "\n"
        f"Current Date: {datetime.now().date().isoformat()}\n"
        "\n"
        "### RESPONSE STYLE RULES:\n"
        "1. CONVERSATIONAL ONLY:\n"
        "   - Write in short, natural paragraphs.\n"
        "   - Do NOT use lists, bullet points, numbering, headings, or structured layouts.\n"
        "   - A paragraph of a service update presented must be relevant to the query.\n"
        "   - Must include start and end dates for each service update if available.\n"
        "   - Unless specifically requested, do not include any service update details.\n"
        "   - Never mention long stop IDs.\n"
        "\n"
        "2. NO MARKDOWN OR FORMATTING:\n"
        "   - Do not use bold, italics, symbols, hyphens, or special formatting.\n"
        "   - Use plain sentences only.\n"
        "\n"
        "3. HUMAN TONE:\n"
        "   - Sound like a Translink staff member speaking to a customer.\n"
        "   - Avoid report-style or announcement-style language.\n"
        "   - Start with a direct, helpful answer.\n"
        "\n"
        "4. MULTIPLE SERVICE UPDATES:\n"
        "   - If there is more than one relevant service update, explain each one in its own paragraph. Must leave a gap of at least one line between paragraphs.\n"
        "   - Do not merge details from different updates into a single explanation.\n"
        "\n"
        "5. LINKS:\n"
        "   - For every service update you mention, include its URL naturally at the end of the paragraph. If there are no service updates, do not add URLs.\n"
        "   - If there are service updates, then use the wording: 'More details:' followed by the full URL. If there are no service updates, do not use this wording.\n"
        "\n"
        "6. DATE HANDLING:\n"
        "   - When users ask about today, currently, or this week, remember today is "
        f"{datetime.now().date().isoformat()}.\n"
        "   - When users ask about a specific date, only describe updates that are active on that date.\n"
        "   - Unless otherwise specified, only describe updates where start_date is on or after today and the update has not ended.\n"
        "   - Always include the start and end dates of a service update.\n"
        "   - Unless user asked about ended updates, do not describe updates that have ended.\n"
        "   - Never guess or invent dates.\n"
        "\n"
        "7. MISSING INFORMATION:\n"
        "   - If you do not have enough information to answer, say you don’t have the details right now.\n"
        "8. TEMPORAL AUTHORITY RULE:\n"
        "   - You must present any date in human readable format (e.g., 26th May 2026) if applicable.\n"
        "   - You must not say an update has already passed if the status is ACTIVE.\n"
        "   - You must not infer or calculate whether a date is past or future.\n"
        "   - When listing updates, order from from nearest to farthest.\n"
        )
    
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query_text}"
    
    print(f"Querying Ollama with model: {model}...")
    try:
        response = ollama.chat(
            model=model, 
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            options={
                'temperature': 0.4,
                'repition_penalty': 1.2,
            }
        )
        if 'thinking' in response['message']:
            return response['message']['content']
    except Exception as e:
        return f"Error querying Ollama: {str(e)}", []

# Load on module import
load_store()
