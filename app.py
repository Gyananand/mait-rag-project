import streamlit as st
import chromadb
import os
import google.generativeai as genai
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# --- 1. CONFIGURATION & MODEL LOADING ---

# Set your Google API key
# -----------------------------------------------------------------
# !!! IMPORTANT !!!
# PASTE YOUR GOOGLE (GEMINI) API KEY HERE
# -----------------------------------------------------------------
try:
    genai.configure(api_key="AIzaSyB-RMPKHJ-Qslp8hd-bq_HbpVNNLXFy9nA") # <--- MAKE SURE YOUR KEY IS HERE
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.error("Please make sure your API key is correct in the app.py file.")
    st.stop()
# -----------------------------------------------------------------


# Set page config (do this first)
st.set_page_config(page_title="MAIT Multimodal Search", layout="wide")
st.title("🤖 MAIT True Multimodal Search (Text/Image-In)")
st.caption("Search with text OR upload an image to find related info.")

# --- Directories ---
DB_DIR = "database_unified" # <-- NEW: Using the unified DB
IMG_DIR = "data/images"

@st.cache_resource  # Magic command to load models only once
def load_models():
    """Loads the CLIP model and the database client."""
    print("Loading CLIP model and DB...")
    # We only need the CLIP model now
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Connect to the existing UNIFIED database
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(name="mait_multimodal")
    
    print("Models and DB loaded.")
    return model, processor, collection

# Load all models and the DB
clip_model, clip_processor, multimodal_collection = load_models()

# --- 2. CORE SEARCH FUNCTIONS ---

def get_query_embedding(text_query=None, image_query=None):
    """
    Creates a CLIP embedding from either text or an image.
    """
    if text_query:
        print("Creating embedding from TEXT query...")
        inputs = clip_processor(text=[text_query], return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            embedding = clip_model.get_text_features(**inputs)
        return embedding.tolist()
        
    elif image_query:
        print("Creating embedding from IMAGE query...")
        img = Image.open(image_query).convert("RGB")
        inputs = clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs)
        return embedding.tolist()
        
    return None

def search_unified_db(embedding, k=5):
    """
    Searches the unified multimodal database.
    """
    if embedding is None:
        return None
        
    results = multimodal_collection.query(
        query_embeddings=embedding,
        n_results=k,
        include=["metadatas", "documents"] # Get both metadata and text content
    )
    return results

def generate_summary(query_str, search_results):
    """Generates a final answer using the Gemini LLM based on text results."""
    
    st.subheader("🤖 AI Summary")
    
    # Filter for text results
    text_context = ""
    for i, metadata in enumerate(search_results['metadatas'][0]):
        if metadata.get('data_type') == 'text':
            doc = search_results['documents'][0][i]
            if doc and doc.strip(): # Check if doc is not None and not empty
                text_context += f"CONTEXT (from {metadata['filename']}):\n{doc}\n---\n"

    if not text_context:
        text_context = "No relevant text was found in the database. I only found images."

    # Create the prompt for the LLM
    prompt = f"""
    You are a helpful assistant for Maharaja Agrasen Institute of Technology (MAIT).
    A user submitted a search: "{query_str}"

    I have found the following pieces of text from the college website:
    {text_context}
    
    Please write a summary based *only* on the provided text.
    If the text says "No relevant text was found", please inform the user that
    you could not find specific text information related to their query, 
    but they should check the retrieved images.
    """
    
    try:
        # Option 1: Faster, cheaper, good for most tasks
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Option 2: More powerful, better reasoning (slower, higher cost)
        model = genai.GenerativeModel('gemini-2.5-pro')
        

        response = model.generate_content(prompt)
        with st.chat_message("assistant"):
            st.write(response.text)
    
    except Exception as e:
        st.error(f"Error generating answer from Gemini: {e}")

# --- 3. STREAMLIT UI ---

# We use two columns for the inputs
input_col1, input_col2 = st.columns(2)

with input_col1:
    text_query = st.text_input("Search with TEXT:", placeholder="e.g., 'workshop on AI'")

with input_col2:
    image_query = st.file_uploader("OR Search with an IMAGE:", type=['png', 'jpg', 'jpeg'])

# Add a search button
search_button = st.button("Search", type="primary")

if search_button:
    query_embedding = None
    query_display = ""

    # --- Logic to decide which query to use ---
    if image_query:
        query_embedding = get_query_embedding(image_query=image_query)
        query_display = "your uploaded image"
        st.image(image_query, caption="Your Query Image", width=200)
    elif text_query:
        query_embedding = get_query_embedding(text_query=text_query)
        query_display = f"'{text_query}'"
    else:
        st.warning("Please enter a text query or upload an image.")
        st.stop()

    # --- Perform the search ---
    if query_embedding:
        st.divider()
        print(f"Searching database for: {query_display}")
        results = search_unified_db(query_embedding)
        
        # --- Separate results into text and images ---
        text_results_docs = []
        image_results_files = []
        
        if results:
            for i, metadata in enumerate(results['metadatas'][0]):
                if metadata.get('data_type') == 'text':
                    text_results_docs.append({
                        "content": results['documents'][0][i],
                        "filename": metadata['filename']
                    })
                elif metadata.get('data_type') == 'image':
                    image_results_files.append(metadata['filename'])

        # --- Display the results ---
        col1, col2 = st.columns([2, 1])

        with col1:
            # Generate Gemini summary from the TEXT results
            generate_summary(query_display, results)
            
            # Display raw text results in an expander
            with st.expander("Show Retrieved Text Snippets"):
                if not text_results_docs:
                    st.write("No relevant text documents found.")
                for res in text_results_docs:
                    st.caption(f"Source: {res['filename']}")
                    st.text(res['content'][:500] + "...") # Show a snippet
                    st.divider()

        with col2:
            st.subheader("Retrieved Images")
            if not image_results_files:
                st.write("No relevant images found.")
            for filename in image_results_files:
                image_path = os.path.join(IMG_DIR, filename)
                if os.path.exists(image_path):
                    st.image(image_path, caption=filename, use_container_width=True)
                else:
                    st.warning(f"Image not found: {filename}")
        
        with st.expander("Show Full Raw Database Output"):
            st.json(results)