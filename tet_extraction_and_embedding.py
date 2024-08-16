from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import textwrap
from openai import OpenAI
import os

# Path to the PDF file
pdf_path = '/Users/sakshi_admin/Documents/Data Science Projects/diabetes-mellitus.pdf'

# Convert PDF pages to images
images = convert_from_path(pdf_path)

# Save images to files
for i, image in enumerate(images):
    image_path = f'page_{i + 1}.png'
    image.save(image_path, 'PNG')
    print(f'Saved {image_path}')
    
# Text Extraction

text = ""
# Path to the Tesseract executable 
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Extract text from images
for i in range(len(images)):
    image_path = f'page_{i + 1}.png'
    image = Image.open(image_path)
    text += pytesseract.image_to_string(image)

# Chunking

def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = textwrap.wrap(text, chunk_size)
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk = chunks[i-1][-overlap:] + chunk
        if i < len(chunks) - 1:
            chunk = chunk + chunks[i+1][:overlap]
        overlapped_chunks.append(chunk)
    return overlapped_chunks

# Create Embeddings

client = OpenAI()  

def create_embeddings(chunk_text):
    embeddings = []
    for chunk in chunk_text:
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"  
        )
        embedding = response.data[0].embedding
        embeddings.append({
            'text': chunk,
            'embedding': embedding
        })
    return embeddings

chunk = chunk_text(text)
embeddings = create_embeddings(chunk)

# Uploading to Pinecone

from pinecone import Pinecone

pc = Pinecone(api_key = os.getenv("pinecone_api_key"))
index = pc.Index("projecty")

vectors = []
for i, (chunk, embedding) in enumerate(zip(chunk_text, embeddings)):
    vector = {
        "id": f"vec{i + 1}",  # Generate unique ID for each vector
        "values": embedding,
        "metadata": {"text": chunk}  # Example metadata; you can customize this
    }
    vectors.append(vector)

index.upsert(
    vectors=vectors,
    namespace= "ns1"
)

    

