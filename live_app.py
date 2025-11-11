import streamlit as st
import chromadb
import os
import google.genai as genai # <-- NEW IMPORT
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# New imports for the Live Agent
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs, urljoin
import time

# --- 1. CONFIGURATION & MODEL LOADING ---

# Set your Google API key from Streamlit's secrets
try:
    # This securely reads the key from your .streamlit/secrets.toml file
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("Could not find GEMINI_API_KEY in your .streamlit/secrets.toml file.")
    st.stop()

st.set_page_config(page_title="MAIT Multimodal Search", layout="wide")
st.title("🤖 MAIT Multimodal RAG Assistant")

# --- Directories ---
DB_DIR = "database_unified"
IMG_DIR = "data/images"

@st.cache_resource
def load_models():
    """Loads the CLIP model and the database client."""
    print("Loading CLIP model and DB...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Connect to the existing UNIFIED database
    db_client = chromadb.PersistentClient(path=DB_DIR) # Renamed to avoid confusion
    collection = db_client.get_collection(name="mait_multimodal")
    
    print("Models and DB loaded.")
    return model, processor, collection

clip_model, clip_processor, multimodal_collection = load_models()

# --- 2. CORE SEARCH FUNCTIONS (No changes) ---

def get_query_embedding(text_query=None, image_query=None):
    if text_query:
        inputs = clip_processor(text=[text_query], return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            embedding = clip_model.get_text_features(**inputs)
        return embedding.tolist()
    elif image_query:
        img = Image.open(image_query).convert("RGB")
        inputs = clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs)
        return embedding.tolist()
    return None

def search_unified_db(collection, embedding, k=5):
    if embedding is None: return None
    results = collection.query(
        query_embeddings=embedding, n_results=k, include=["metadatas", "documents"]
    )
    return results

# --- *** Updated generate_summary function *** ---
def generate_summary(query_str, search_results, is_live_agent=False):
    st.subheader("🤖 AI Summary")
    
    text_context = ""
    for i, metadata in enumerate(search_results['metadatas'][0]):
        doc = search_results['documents'][0][i]
        source = metadata.get('source', 'unknown source')
        if is_live_agent:
            source = metadata.get('url', 'unknown url') # Live agent uses 'url'
            
        if doc and doc.strip():
            text_context += f"CONTEXT (from {source}):\n{doc}\n---\n"

    if not text_context:
        text_context = "No relevant text was found in the database. I only found images."

    prompt = f"""
    You are a helpful assistant. A user submitted a search: "{query_str}"
    I have found the following pieces of text from the website:
    {text_context}
    
    Please answer the user's question using *only* this provided text.
    First, provide a direct, concise answer.
    Then, if the context is detailed, you can add a "Key Findings" section with bullet points.
    If no relevant text was found, just say "I could not find any specific information about that topic in the provided text."
    """
    
    try:
        # --- *** THIS IS THE FIX *** ---
        # We must call client.MODELS.generate_content(...)
        response = client.models.generate_content(
            model='gemini-2.5-flash', # Using the most modern, available model
            contents=prompt
        )
        # --- *** END FIX *** ---

        with st.chat_message("assistant"):
            st.write(response.text)
    except Exception as e:
        st.error(f"Error generating answer from Gemini: {e}")

# --- 3. LIVE AGENT FUNCTIONS (No changes) ---

@st.cache_data(ttl=600) # Cache scrapes for 10 minutes
def scrape_and_find_links(url, base_domain):
    """
    Scrapes a single page, extracts text, and finds all crawlable links.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # 1. Handle YouTube
    if "youtube.com" in url or "youtu.be" in url:
        try:
            video_id = parse_qs(urlparse(url).query)['v'][0]
            transcript = " ".join([d['text'] for d in YouTubeTranscriptApi.get_transcript(video_id)])
            return transcript, set() # No links to crawl from a video
        except Exception:
            return None, set()

    # 2. Handle regular Webpages
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text(separator='\n', strip=True)
        
        # 3. Find links
        links = set()
        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']
            full_link = urljoin(url, link) # Convert relative links to absolute
            
            # Only crawl links that are part of the original site
            if full_link.startswith(base_domain):
                links.add(full_link.split('#')[0]) # Add link and remove fragments
                
        return text, links
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None, set()

def create_temp_index_from_crawl(scraped_data):
    """Creates a temporary, in-memory ChromaDB from scraped text."""
    print("Creating temporary in-memory index...")
    temp_client = chromadb.Client()
    
    try:
        temp_client.delete_collection(name="live_agent_collection")
    except Exception:
        pass # Collection didn't exist, which is fine
        
    temp_collection = temp_client.create_collection(name="live_agent_collection")
    
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    # Chunk the text from each scraped page
    for url, text_content in scraped_data.items():
        if not text_content:
            continue
            
        # Split by paragraph
        chunks = [chunk for chunk in text_content.split("\n") if len(chunk.strip()) > 50]
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({"data_type": "text", "url": url})
            all_ids.append(f"{url}::{i}")
    
    if not all_chunks:
        return None
    
    # Embed all chunks in batches (using CLIP)
    batch_size = 32
    for i in range(0, len(all_chunks), batch_size):
        batch_texts = all_chunks[i:i+batch_size]
        batch_metadatas = all_metadatas[i:i+batch_size]
        batch_ids = all_ids[i:i+batch_size]
        
        inputs = clip_processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            embeddings = clip_model.get_text_features(**inputs)
        
        temp_collection.add(
            embeddings=embeddings.tolist(),
            documents=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
    
    print(f"Temporary index created with {len(all_chunks)} text chunks.")
    return temp_collection

# --- 4. STREAMLIT UI (NOW WITH TABS) ---

tab1, tab2 = st.tabs(["🏛️ MAIT Search (Pre-Indexed)", "🕷️ Live Agent (Crawl any URL)"])

# --- TAB 1: Your original app ---
with tab1:
    st.header("Search the Pre-Indexed MAIT Database")
    st.caption("This is FAST. It searches the database we built with `indexer.py`.")
    
    input_col1, input_col2 = st.columns(2)
    with input_col1:
        text_query_1 = st.text_input("Search with TEXT:", placeholder="e.g., 'workshop on AI'", key="text_1")
    with input_col2:
        image_query_1 = st.file_uploader("OR Search with an IMAGE:", type=['png', 'jpg', 'jpeg'], key="image_1")

    search_button_1 = st.button("Search MAIT DB", type="primary")

    if search_button_1:
        query_embedding_1 = None
        query_display_1 = ""
        if image_query_1:
            query_embedding_1 = get_query_embedding(image_query=image_query_1)
            query_display_1 = "your uploaded image"
            st.image(image_query_1, caption="Your Query Image", width=200)
        elif text_query_1:
            query_embedding_1 = get_query_embedding(text_query=text_query_1)
            query_display_1 = f"'{text_query_1}'"
        else:
            st.warning("Please enter a text query or upload an image.")
            st.stop()

        if query_embedding_1:
            st.divider()
            results_1 = search_unified_db(multimodal_collection, query_embedding_1)
            
            text_results_docs_1 = []
            image_results_files_1 = []
            
            if results_1:
                for i, metadata in enumerate(results_1['metadatas'][0]):
                    if metadata.get('data_type') == 'text':
                        text_results_docs_1.append({
                            "content": results_1['documents'][0][i],
                            "filename": metadata['filename']
                        })
                    elif metadata.get('data_type') == 'image':
                        image_results_files_1.append(metadata['filename'])

            col1, col2 = st.columns([2, 1])
            with col1:
                generate_summary(query_display_1, results_1)
                with st.expander("Show Retrieved Text Snippets"):
                    if not text_results_docs_1: st.write("No relevant text documents found.")
                    for res in text_results_docs_1:
                        st.caption(f"Source: {res['filename']}")
                        st.text(res['content'][:500] + "...")
                        st.divider()
            with col2:
                st.subheader("Retrieved Images")
                if not image_results_files_1: st.write("No relevant images found.")
                for filename in image_results_files_1:
                    image_path = os.path.join(IMG_DIR, filename)
                    if os.path.exists(image_path):
                        st.image(image_path, caption=filename, use_container_width=True)
                    else:
                        st.warning(f"Image not found: {filename}")
            with st.expander("Show Full Raw Database Output"):
                st.json(results_1)

# --- TAB 2: Your new "Live Agent" feature ---
with tab2:
    st.header("Crawl a Live URL (or YouTube video)")
    st.caption("This is SLOW but always has 100% fresh data.")
    
    live_url = st.text_input("Enter a URL to crawl (webpage or YouTube):", placeholder="https://mait.ac.in/ or https://www.youtube.com/watch?v=...")
    live_query = st.text_input("What do you want to know about this URL?", placeholder="e.g., 'Summarize this' or 'What events are happening?'")
    
    # New Slider for crawl depth
    crawl_depth = st.slider("Crawl Depth (How many links deep to search)", 1, 3, 1)
    
    search_button_2 = st.button("Crawl and Analyze", type="primary")
    
    if search_button_2:
        if not live_url or not live_query:
            st.warning("Please enter both a URL and a query.")
            st.stop()
            
        with st.spinner(f"Starting crawl at {live_url} (depth {crawl_depth})... This will take time..."):
            
            # --- This is the new CRAWLING logic ---
            base_domain = f"{urlparse(live_url).scheme}://{urlparse(live_url).netloc}"
            links_to_crawl = {live_url}
            scraped_data = {} # Will store {url: text_content}
            visited_links = set()
            
            for depth in range(crawl_depth):
                if not links_to_crawl:
                    break
                    
                current_links_to_crawl = list(links_to_crawl)
                links_to_crawl.clear() # Prepare for next level
                
                st.write(f"Depth {depth+1}: Crawling {len(current_links_to_crawl)} links...")
                
                for link in current_links_to_crawl:
                    if link in visited_links:
                        continue
                        
                    visited_links.add(link)
                    text, new_links = scrape_and_find_links(link, base_domain)
                    
                    if text:
                        scraped_data[link] = text
                    
                    if depth < crawl_depth - 1: # Don't find new links on the last level
                        links_to_crawl.update(new_links - visited_links)
                    
                    time.sleep(0.1) # Be nice to their server

            if not scraped_data:
                st.error("Could not retrieve any text. The site may be blocked or empty.")
                st.stop()
            
            st.write(f"Crawl complete. Scraped {len(scraped_data)} pages. Now building index...")
            
            # 2. Create a temporary database from ALL scraped text
            temp_collection = create_temp_index_from_crawl(scraped_data)
            if not temp_collection:
                st.error("Could not create a temporary index from the URL's text.")
                st.stop()
        
        st.success("Page(s) processed! Now searching...")
        
        # 3. Get the embedding for the user's query
        query_embedding_2 = get_query_embedding(text_query=live_query)
        
        # 4. Search the temporary database
        results_2 = search_unified_db(temp_collection, query_embedding_2, k=10)
        
        # 5. Display the results
        generate_summary(live_query, results_2, is_live_agent=True)
        
        with st.expander("Show Retrieved Text Snippets from Crawl"):
            st.json(results_2)
