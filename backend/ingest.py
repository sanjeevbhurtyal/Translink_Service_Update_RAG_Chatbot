import feedparser
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import json
import ollama
import os
import argparse
import rag

RSS_URL = "https://translink.com.au/service-updates/rss"
CHECKPOINT_FILE = "backend/ingestion_checkpoint.json"
DOCUMENTS_FILE = "backend/documents.json"

def load_checkpoint():
    """Load checkpoint to track processed URLs."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"processed_urls": [], "last_updated": None}

def save_checkpoint(checkpoint):
    """Save checkpoint state."""
    checkpoint["last_updated"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def load_documents(filename=DOCUMENTS_FILE):
    """Load previously saved documents"""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_documents(documents, filename=DOCUMENTS_FILE):
    """Save documents to file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

def clean_scarped_text(text):
    """Aggressive cleaning of scraped text."""
    # Block Multi-line JSON/Map data
    map_patterns = [
        r'"markers"\s*:.*?\]',
        r'markers\s*[:=].*?\]',
        r'jQuery\.extend\(.*?\);'
    ]
    for p in map_patterns:
        text = re.sub(p, '', text, flags=re.IGNORECASE | re.DOTALL)

    # Block Single line noise
    patterns_to_remove = [
        r'"lat":".*?",',
        r'"lng":".*?",',
        r'"icon":".*?",',
        r'https?://\S+?\.png', 
        r'https?://\S+?\.jpg',
        r"Last reviewed:.*",
        r"View map of closure",
        r"Plan your journey.*",
        r"Need assistance\?.*",
        r"call 13 16 17.*",
        r"SMS 0428.*",
        r"\w+ \d{1,2} \w+ \d{4} at \d{1,2}\.\d{2}[ap]m" 
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    text = re.sub(r'[\[\]\{\}"]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_dates_and_services(description: str):
    """Extract start_date, end_date, and services from RSS description."""
    pattern = re.compile(
        r"Start date:\s*(?P<start>[^,]+(?:AM|PM))"
        r"(?:,\s*End date:\s*(?P<end>[^,]+(?:AM|PM)))?"
        r",\s*Services:\s*(?P<services>.+)$"
    )
    match = pattern.search(description)
    if not match:
        return None, None, None
    return match.group("start"), match.group("end"), match.group("services")

def parse_datetime_flex(date_str):
    if not date_str:
        return None

    formats = [
        "%d/%m/%Y %I:%M %p", 
        "%d/%m/%Y %H:%M",     
        "%d/%m/%Y",          
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except:
            continue

    return None

def scrape_translink_page(url):
    """Scrape detailed content from Translink page."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove noise
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'map']):
            tag.decompose()
        
        noise_classes = ['map-container', 'location-map', 'contact-info', 'social-share']
        for cls in noise_classes:
            for div in soup.find_all(class_=cls):
                div.decompose()

        # Find content
        content_div = None
        selectors = ['main', 'div[role="main"]', '#content', '#main', '.region-content', 'article']
        for selector in selectors:
            content_div = soup.select_one(selector)
            if content_div:
                break
        
        if not content_div:
            content_div = soup.body
        if not content_div:
            return ""

        text = content_div.get_text(separator='\n', strip=True)
        return clean_scarped_text(text)

    except Exception as e:
        print(f" Scraping failed: {e}")
        return ""

def summarize_with_llm(detailed_text: str, metadata: dict, model: str = "qwen3:8b"):
    """Summarize and extract structured information using LLM."""
    system_prompt = """You are a transport disruption extraction engine.

    You will receive:
    1. Authoritative metadata that has already been extracted.
    2. Raw notice text.

    Rules:
    - Do NOT modify metadata values.
    - Use metadata to improve accuracy and completeness.
    - Use start_date and end_date from metadata if they are not null.
    - Only infer missing fields if the metadata value is null.
    - Output ONLY a single valid JSON object matching the schema.
    - embedding_text must be 1-3 sentence, search-optimized, and include routes, locations, start/end dates, and impact.
    """

    user_prompt = f"""Using the authoritative metadata and the notice text below, extract structured information.

    Authoritative metadata (do not modify these values):
    {json.dumps(metadata, indent=2)}

    Return a JSON object EXACTLY in this format.  
    **MUST include all keys** exactly as listed, even if the value is empty or null. 
    If a value cannot be determined, use null (for strings) or empty array (for lists).


    {{
    "embedding_text": "1-4 sentence summary that must include all relevant information from the notice",
    "start_date": "YYYY-MM-DD or YYYY-MM-DD HH:MM AM/PM or null",
    "end_date": "YYYY-MM-DD or YYYY-MM-DD HH:MM AM/PM or null",
    "routes": ["route numbers or null"],
    "locations": ["location names or null"],
    "stops": ["stop IDs or null"],
    "keywords": ["relevant search keywords or null"]
    }}


    Rules:
    1. Always use start_date from metadata.
    2. Use end_date from metadata if present. If null, infer it from the notice text if possible.
    3. Format embedding_text as follows:
    - If both start_date and end_date exist: "From start_date to end_date, [summary of impact including routes and locations]."
    - If only start_date exists: "On start_date, [summary of impact including routes and locations]."
    - If start_date is null: omit dates from embedding_text.
    - embedding_text: 1-4 sentence summary that must include all relevant information from the notice, including:
        - routes, locations, start and end dates, impact,
        - operational details, access instructions, safety or accessibility notes,
        - and all instructions for users exactly as described in the notice text, including specific paths, access points, or construction details.
    4. Infer routes, locations, stops, and keywords from the notice text.
    5. Do not invent information.
    6. If a value cannot be determined, use null or an empty array.
    7. **Double check the output to match with the JSON object schema**. 
    8. There can be more routes or stops or services in notice text than in metadata. The metadata is just for reference. The embedding_text can add more details than the metadata. 

    Examples:

    1. Multi-day closure from metadata
    Metadata:
    {{"start_date": "2023-10-09", "end_date": "2023-10-11", "services": ["700","753"]}}
    Notice Text:
    Temporary stop closures on Gold Coast Highway.
    embedding_text:
    "From 2023-10-09 to 2023-10-11, temporary stop closures affected routes 700 and 753 on Gold Coast Highway."

    2. End date in text only
    Metadata:
    {{"start_date": "2023-10-09", "end_date": null, "services": ["700","753"]}}
    Notice Text:
    Temporary stop closures on Gold Coast Highway until 2023-10-11 8:30 PM.
    embedding_text:
    "From 2023-10-09 to 2023-10-11 8:30 PM, temporary stop closures affected routes 700 and 753 on Gold Coast Highway."

    3. Single-day closure
    Metadata:
    {{"start_date": "2023-10-09", "end_date": null, "services": ["600","611"]}}
    Notice Text:
    Temporary stop closure at Mooloolaba Bowls Club.
    embedding_text:
    "On 2023-10-09, temporary stop closures affected routes 600 and 611 at Mooloolaba Bowls Club."

    4. Estimated end date
    Metadata:
    {{"start_date": "2023-10-09", "end_date": null, "services": ["600","611"]}}
    Notice Text:
    Temporary stop closure at Mooloolaba Bowls Club until early 2026.
    embedding_text:
    "From 2023-10-09 to early 2026, temporary stop closures affected routes 600 and 611 at Mooloolaba Bowls Club."

    Notice text:
    <<<
    {detailed_text}
    >>>
    """

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.4,
        "repetition_penalty": 1.1
        }
    )

    raw_output = response["message"]["content"]

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        raise ValueError(f"LLM returned invalid JSON.\n\nRaw output:\n{raw_output}")

def fetch_translink_rss(resume=True):
    """
    Fetch and process RSS feed with checkpoint support.
    
    Args:
        resume: If True, skip already-processed URLs. If False, start fresh.
    """
    feed = feedparser.parse(RSS_URL)
    documents = []
    
    # Load checkpoint
    checkpoint = load_checkpoint() if resume else {"processed_urls": [], "last_updated": None}
    documents = load_documents() if resume else []
    processed_urls = set(checkpoint["processed_urls"])
    
    print(f"\n Fetched {len(feed.entries)} entries from Translink RSS")
    if resume and processed_urls:
        print(f" Resuming - {len(processed_urls)} already processed")
    else:
        print(" Starting fresh scrape")
    
    count = 0
    processed_count = 0
    skipped_count = 0
    errors = []
    
    for entry in feed.entries:
        count += 1
        
        title = clean_text(entry.get('title', 'Unknown Title'))
        link = entry.get('link', '')
        categories = [t.term for t in entry.get('tags', [])] if 'tags' in entry else []
        category = current_category(categories)
        start_date, end_date, services = extract_dates_and_services(entry.get('description', ''))

        # Skip if already processed
        if resume and link in processed_urls:
            skipped_count += 1
            continue
        
        print(f"  [{count}/{len(feed.entries)}] {title[:60]}...")
        
        try:
            # Scrape
            detailed_text = scrape_translink_page(link)
            final_text = f"FULL DETAILS:\n{detailed_text}"

            # Prepare metadata
            metadata = {
                "title": title,
                "url": link,
                "category": category,
                "start_date": start_date,
                "end_date": end_date,
                "services": services,
            }
            
            # Summarize with LLM
            summary = summarize_with_llm(final_text, metadata)

            # Retry if keys missing
            keys = ["embedding_text", "start_date", "end_date", "routes", "locations", "stops", "keywords"]
            while any(key not in summary for key in keys):
                summary = summarize_with_llm(final_text, metadata)

            # Expand metadata

            start_dt = parse_datetime_flex(metadata.get("start_date"))
            end_dt   = parse_datetime_flex(metadata.get("end_date"))
            
            metadata["start_date_raw"] = metadata.get("start_date")
            metadata["end_date_raw"]   = metadata.get("end_date")
            metadata["start_date_iso"] = start_dt.strftime("%Y-%m-%d") if start_dt else None
            metadata["end_date_iso"]   = end_dt.strftime("%Y-%m-%d") if end_dt else None
            metadata["start_date_epoch"] = int(start_dt.timestamp()) if start_dt else None
            metadata["end_date_epoch"]   = int(end_dt.timestamp()) if end_dt else None
            metadata["routes"] = summary["routes"]
            metadata["locations"] = summary["locations"]
            metadata["stops"] = summary["stops"]
            metadata["keywords"] = summary["keywords"]

            doc = {
                "id": link,
                "text": summary["embedding_text"],
                "metadata": metadata
            }

            # Add immediately to vector store
            rag.add_documents([doc])
    
            # Save checkpoint
            processed_urls.add(link)
            checkpoint["processed_urls"] = list(processed_urls)
            save_checkpoint(checkpoint)

            # Save documents 
            documents.append(doc)
            save_documents(documents)

            processed_count += 1
            
            time.sleep(0.2)
            
        except KeyboardInterrupt:
            print("\n\n Interrupted - checkpoint saved")
            print(f" Progress: {processed_count} new entries processed")
            raise
        except Exception as e:
            error_msg = f"{title}: {str(e)}"
            errors.append(error_msg)
            print(f" ERROR: {error_msg}")
            continue
    
    print(f"\n Completed: {processed_count} new, {skipped_count} skipped")
    if errors:
        print(f" Errors: {len(errors)}")
        for err in errors[:5]:
            print(f"  - {err}")
    
    return documents

def clean_text(text):
    """Remove unicode characters and whitespace."""
    text = text.encode('ascii', 'ignore').decode('ascii') 
    return text.strip()

def current_category(cats):
    """Determine category priority."""
    cats = {c.strip().lower() for c in cats}
    if {"major"}.issubset(cats):
        return "Major"
    if "minor" in cats:
        return "Minor"
    return "Informative"

if __name__ == "__main__":
    import rag
    
    parser = argparse.ArgumentParser(description='Ingest Translink RSS with checkpoint support')
    parser.add_argument('--fresh-start', action='store_true', help='Clear documents and vector store and start fresh (ignore checkpoint)')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(" TRANSLINK RSS INGESTION")
    print("="*60)
    
    if args.fresh_start:
        print("\n Clearing vector store and checkpoint...")
        rag.vector_store = {'ids': [], 'documents': [], 'metadatas': [], 'embeddings': []}
        rag.save_store()
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        if os.path.exists(DOCUMENTS_FILE):
            os.remove(DOCUMENTS_FILE)
        print(" Cleared")
    
    try:
        docs = fetch_translink_rss(resume=not args.fresh_start)
        
        if docs:
            print(f"\n Processed {len(docs)} new documents.")
        else:
            print("\n No new documents")
    
    except KeyboardInterrupt:
        print("\n\n Exiting - resume with 'python ingest.py'")
    except Exception as e:
        print(f"\n Error: {e}")
        raise