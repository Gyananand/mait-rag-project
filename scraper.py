import requests
import os
import re
import shutil
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time # Import time for polite scraping

# 1. Define our data directories
BASE_DATA_DIR = "data"
TEXT_DIR = os.path.join(BASE_DATA_DIR, "text")
PDF_DIR = os.path.join(BASE_DATA_DIR, "pdf")
IMG_DIR = os.path.join(BASE_DATA_DIR, "images")

# 2. Create these directories
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# 3. Set a browser-like User-Agent
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def get_target_urls():
    """
    Returns the curated list of 21 key MAIT pages to scrape.
    """
    urls = [
        "https://www.mait.ac.in/index.php/component/content/category/latest-news.html?Itemid=",
        "https://it.mait.ac.in/index.php/it/events",
        "https://cse.mait.ac.in/index.php/academics-2/student-events-activities",
        "https://eee.mait.ac.in/index.php/academics-2/sch-events",
        "https://ece.mait.ac.in/index.php/ece/events-main/event-calendar",
        "https://mait.ac.in/index.php/abouts/about-us.html",
        "https://cse.mait.ac.in/index.php/academics-2/student-events-activities/seminars-workshops-bootcamps",
        "https://cse.mait.ac.in/index.php/academics-2/student-events-activities/internship-fair",
        "https://cse.mait.ac.in/index.php/academics-2/student-events-activities/innovation-mela",
        "https://cse.mait.ac.in/index.php/academics-2/student-events-activities/pitch-your-idea",
        "https://cse.mait.ac.in/index.php/academics-2/student-events-activities/industry-interaction",
        "https://cse.mait.ac.in/index.php/hackathon/sih",
        "https://cse.mait.ac.in/index.php/hackathon/hackwithmait",
        "https://ece.mait.ac.in/",
        "https://ece.mait.ac.in/index.php/student-info/events-activities/internship-fair",
        "https://ece.mait.ac.in/index.php/student-info/events-activities/external",
        "https://ece.mait.ac.in/index.php/ece/events-main/industrial-visits",
        "https://eee.mait.ac.in/",
        "http://gallery.mait.ac.in/gallery3/",
        "https://eee.mait.ac.in/index.php/quick-links/popular-links",
        "https://it.mait.ac.in/"
    ]
    
    # Add the management department URLs from 2017 to 2025
    mgmt_base = "https://mgmt.mait.ac.in/index.php/events-menu/events"
    for year in range(2017, 2026): # 2026 is exclusive, so it stops at 2025
        urls.append(f"{mgmt_base}{year}")
        
    # Fix for the 'event2023' typo
    urls.append("https://mgmt.mait.ac.in/index.php/events-menu/event2023")

    unique_urls = list(set(urls))
    print(f"Generated {len(unique_urls)} total unique index pages to crawl.")
    return unique_urls

def get_clean_filename(url, content_type="text"):
    """
    Creates a robust, clean filename from a URL to avoid collisions.
    """
    try:
        # Use the path to create a more unique name
        parsed_url = urlparse(url)
        name = f"{parsed_url.netloc}{parsed_url.path}{parsed_url.query}".replace('http://', '').replace('https://', '').replace('/', '_').replace('?', '_').replace('=', '_')
        name = re.sub(r'[^a-zA-Z0-9_\-.]', '_', name) # Keep dots
        
        # Limit filename length
        name = name[:100] 
        
        if content_type == "pdf":
            return f"{name}.pdf" if not name.endswith('.pdf') else name
        elif content_type == "image":
            # Keep original extension if possible
            if any(url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                return name
            else:
                return f"{name}.jpg" # Default to jpg if unknown
        else: # text
            return f"{name}.txt"
    except Exception:
        return f"scraped_{hash(url)}.{content_type}"

def save_text_content(url, text):
    """Saves scraped text to a file in the data/text folder."""
    filename = get_clean_filename(url, content_type="text")
    filepath = os.path.join(TEXT_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"SOURCE_URL: {url}\n\n")
        f.write(text)
    print(f"  [+] Saved text to: {filepath}")

def download_pdf(url):
    """Downloads a PDF and saves it to the data/pdf folder."""
    filename = get_clean_filename(url, content_type="pdf")
    filepath = os.path.join(PDF_DIR, filename)
    if os.path.exists(filepath):
        print(f"  [=] PDF already exists: {filepath}")
        return

    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=10) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print(f"  [+] Saved PDF to: {filepath}")
    except Exception as e:
        print(f"  [!] FAILED to download PDF {url}: {e}")

def download_image(url):
    """Downloads an image and saves it to the data/images folder."""
    filename = get_clean_filename(url, content_type="image")
    filepath = os.path.join(IMG_DIR, filename)
    if os.path.exists(filepath): # Avoid re-downloading
        return

    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=10) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print(f"    [+] Saved Image to: {filepath}")
    except Exception as e:
        print(f"    [!] FAILED to download image {url}: {e}")

def scrape_article_page(url):
    """
    Visits a single article page and scrapes all text AND images.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        
        content_area = soup.find('div', class_='item-page')
        if not content_area:
            content_area = soup.find('article') 
        if not content_area:
            print(f"  [!] No 'item-page' or 'article' found for {url}. Grabbing all body text.")
            content_area = soup.body
            if not content_area:
                return None # Page is empty

        # 1. Scrape Text
        text = content_area.get_text(separator='\n', strip=True)
        
        # 2. Scrape Images (This is the new part)
        images = content_area.find_all('img')
        print(f"    Found {len(images)} images on page...")
        for img in images:
            if img.has_attr('src'):
                img_src = img['src']
                # Create the full, absolute URL for the image
                full_img_url = urljoin(url, img_src)
                
                # Avoid tiny spacer/icon images
                if 'width' in img.attrs and isinstance(img.attrs['width'], str) and img.attrs['width'].isdigit() and int(img.attrs['width']) < 50:
                    continue
                if 'height' in img.attrs and isinstance(img.attrs['height'], str) and img.attrs['height'].isdigit() and int(img.attrs['height']) < 50:
                    continue
                    
                download_image(full_img_url)
        
        return text
    
    except Exception as e:
        print(f"  [!] FAILED to scrape article {url}: {e}")
        return None

def find_links_on_index_page(index_url):
    """
    Visits an index page and finds all "sublinks" to PDFs and articles.
    """
    print(f"\n--- Crawling Index Page: {index_url} ---")
    pdf_links = set()
    article_links = set()
    
    try:
        # Be polite!
        time.sleep(0.1)
        
        response = requests.get(index_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')

        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href']
            
            # 1. Check for PDF links
            if href.endswith('.pdf'):
                full_pdf_url = urljoin(index_url, href)
                pdf_links.add(full_pdf_url)
                
            # 2. Check for Article links (a common Joomla pattern)
            if 'index.php/component/content/article' in href:
                full_article_url = urljoin(index_url, href)
                article_links.add(full_article_url)
        
        print(f"  Found {len(pdf_links)} PDF links and {len(article_links)} article links.")
        return pdf_links, article_links
        
    except Exception as e:
        print(f"  [!] FAILED to crawl index {index_url}: {e}")
        return set(), set()


# --- Main script execution ---
def main():
    print("Starting the MAIT Smart Multimodal Scraper...")
    
    # 1. Get all the main pages to crawl
    index_pages = get_target_urls()
    
    all_pdfs_found = set()
    all_articles_found = set()
    
    # 2. Find all the "sublinks" first
    for page_url in index_pages:
        pdfs, articles = find_links_on_index_page(page_url)
        all_pdfs_found.update(pdfs)
        all_articles_found.update(articles)
        
    print(f"\n\n--- Crawl Complete ---")
    print(f"Found {len(all_pdfs_found)} unique PDF links.")
    print(f"Found {len(all_articles_found)} unique article links.")
    print("---------------------------------")
    print("Starting content download and scrape...")

    # 3. Process all unique PDFs found
    print(f"\n--- Downloading {len(all_pdfs_found)} PDFs ---")
    for pdf_url in all_pdfs_found:
        download_pdf(pdf_url)

    # 4. Process all unique articles found
    print(f"\n--- Scraping {len(all_articles_found)} Articles (and their Images) ---")
    for article_url in all_articles_found:
        print(f"Scraping: {article_url}")
        text_content = scrape_article_page(article_url)
        if text_content:
            save_text_content(article_url, text_content)

    print("\n\n--- All Done! ---")
    print(f"Check your '{BASE_DATA_DIR}' folder.")
    print(f"You should have text files in '{TEXT_DIR}', PDFs in '{PDF_DIR}', and images in '{IMG_DIR}'.")

# This makes the script runnable from the command line
if __name__ == "__main__":
    main()