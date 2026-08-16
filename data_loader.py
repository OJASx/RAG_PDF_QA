import os
from google import genai
from google.genai import types  # Imported to handle dimension configuration
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

# Automatically grabs GEMINI_API_KEY from environment variables
client = genai.Client()

# Recommended options:
# 1. Use "gemini-embedding-2" (native 3072-dim)
# 2. Use "text-embedding-004" (768-dim by default, but can be scaled)
EMBED_MODEL = "gemini-embedding-2" 
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    # Call the correct method and parameter names
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts, # Changed from input=
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM) # Ensures 3072 dims
    )
    
    # Correctly parse Google's nested response structure
    return [item.values for item in response.embeddings]
