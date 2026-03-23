# scr/chunk_text.py
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_FOLDER = "chunks"

os.makedirs(CHUNK_FOLDER, exist_ok=True)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)


def chunk_text(text):
    """
    تقسيم نص واحد إلى chunks
    
    Args:
        text (str): النص المراد تقسيمه
    
    Returns:
        list: قائمة بالـ chunks
    """
    if not text or len(text.strip()) == 0:
        return []
    
    chunks = text_splitter.split_text(text)
    return chunks