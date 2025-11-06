import fitz  # pymupdf
import os
import io
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# --- Our Data Directories ---
PDF_DIR = "data/pdf"
TEXT_DIR = "data/text"
IMG_DIR = "data/images"
DB_DIR = "database_unified" # We are rebuilding this

# Create directories
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
# We don't delete the DB_DIR, we just re-initialize the client

print("Script started. Loading CLIP model... This may take a minute.")

# --- 1. LOAD AI MODEL (CLIP ONLY) ---
print("Loading CLIP Model (openai/clip-vit-base-patch32)...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP Model loaded.")


# --- 2. PRE-PROCESSING (Your Unpacker) ---
def unpack_pdfs():
    print("\n--- Phase 1: Unpacking PDFs ---")
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDFs to process in '{PDF_DIR}'...")
    
    processed_files = 0
    # Since we deleted the /text folder, this will re-process EVERYTHING
    for pdf_filename in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_filename)
        text_filename = f"pdf_{pdf_filename.replace('.pdf', '.txt')}"
        text_filepath = os.path.join(TEXT_DIR, text_filename)
        
        # We don't check for existence, we just re-build
        try:
            doc = fitz.open(pdf_path)
            print(f"Processing PDF: {pdf_filename}...")
            # Extract Text
            full_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text += page.get_text("text") + "\n"
            
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write(f"SOURCE_PDF: {pdf_filename}\n\n")
                f.write(full_text)
            print(f"  [+] Saved extracted text.")
            processed_files += 1
        except Exception as e:
            print(f"  [!] FAILED to process {pdf_filename}: {e}")
    
    print(f"PDF Unpacking complete. {processed_files} PDFs processed.")


# --- 3. DATABASE INDEXING (MODIFIED) ---

def initialize_database():
    """Initializes a persistent ChromaDB client with a SINGLE collection."""
    print("\n--- Phase 2: Initializing Unified Vector Database ---")
    client = chromadb.PersistentClient(path=DB_DIR)

    # Delete the old collection if it exists, to start fresh
    try:
        client.delete_collection(name="mait_multimodal")
        print("  [+] Deleted old 'mait_multimodal' collection.")
    except Exception:
        pass # It didn't exist, which is fine

    multimodal_collection = client.get_or_create_collection(
        name="mait_multimodal",
        metadata={"hnsw:space": "cosine"}
    )
    
    print("Database and new 'mait_multimodal' collection are ready.")
    return multimodal_collection

def embed_text_files(collection):
    """
    Embeds all .txt files using CLIP's TEXT model and stores them.
    """
    print("\n--- Phase 3: Indexing Text Files (with CLIP) ---")
    text_files = [f for f in os.listdir(TEXT_DIR) if f.endswith('.txt')]
    print(f"Found {len(text_files)} text files to index...")

    batch_size = 32
    files_to_add = text_files # We re-add all files
    
    print(f"Attempting to add {len(files_to_add)} text files.")

    for i in range(0, len(files_to_add), batch_size):
        batch_files = files_to_add[i:i+batch_size]
        batch_documents = []
        batch_metadatas = []
        batch_ids = []
        
        for filename in batch_files:
            filepath = os.path.join(TEXT_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.readline().strip()
                content = f.read()

            # --- *** THIS IS THE CRITICAL FIX *** ---
            # We set a "quality filter"
            # Only add the file if it has more than 100 characters of real content.
            if len(content.strip()) < 100:
                print(f"  [!] REJECTING {filename}, no meaningful content.")
                continue # Skip this file
            # --- *** END FIX *** ---
            
            batch_documents.append(content)
            batch_metadatas.append({"source": source, "filename": filename, "data_type": "text"})
            batch_ids.append(f"text_{filename}")
        
        # Only add to DB if the batch is not empty
        if batch_documents:
            # Create embeddings using CLIP's text processor
            inputs = clip_processor(text=batch_documents, return_tensors="pt", padding=True, truncation=True, max_length=77)
            with torch.no_grad(): # Disable gradient calculation for efficiency
                embeddings = clip_model.get_text_features(**inputs)
            
            collection.add(
                embeddings=embeddings.tolist(),
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            print(f"  Indexed text batch {i//batch_size + 1}/{(len(files_to_add)-1)//batch_size + 1}")

    print("Text indexing complete.")

def embed_image_files(collection):
    """
    Embeds all image files using CLIP's IMAGE model and stores them.
    """
    print("\n--- Phase 4: Indexing Image Files (with CLIP) ---")
    image_files = [f for f in os.listdir(IMG_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} image files to index...")

    images_to_add = image_files # We re-add all images
    
    print(f"Attempting to add {len(images_to_add)} new image files.")

    processed_count = 0
    for filename in images_to_add:
        filepath = os.path.join(IMG_DIR, filename)
        file_id = f"image_{filename}"
        try:
            img = Image.open(filepath).convert("RGB")
            
            inputs = clip_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                embedding = clip_model.get_image_features(**inputs)
            
            collection.add(
                embeddings=embedding.tolist(),
                metadatas=[{"source": filepath, "filename": filename, "data_type": "image"}],
                ids=[file_id]
            )
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"  ...indexed {processed_count} images...")
                
        except Exception as e:
            print(f"  [!] FAILED to index image {filename}: {e}")
            
    print(f"Image indexing complete. Added {processed_count} new images.")


# --- Main script execution ---
if __name__ == "__main__":
    # We don't need to run unpack_pdfs() if we already have the /data/text folder.
    # But since we deleted it, we run it.
    unpack_pdfs() 
    
    # Run the scraper again to get the text from HTML pages
    # NOTE: You must have your scraper.py file in the same folder.
    try:
        import scraper 
        print("\n--- Running scraper.py to fetch HTML text ---")
        scraper.main()
        print("--- Scraper run complete ---")
    except ImportError:
        print("\n[!] 'scraper.py' not found. Skipping HTML text scraping.")
        print("    Your database will only contain text from PDFs.")
    
    # Initialize the single, unified database
    multimodal_collection = initialize_database()
    
    # Embed all text files into the collection
    embed_text_files(multimodal_collection)
    
    # Embed all image files into the *same* collection
    embed_image_files(multimodal_collection)
    
    print("\n\n--- All Unified Indexing Complete! ---")
    print("Your new, CLEAN multimodal vector database is ready.")
    print(f"Database is saved in the '{DB_DIR}' folder.")