# scr/process_pdf.py
"""
معالج PDF يستخدم الدوال الموجودة
"""

from dotenv import load_dotenv
import os
import pypdf
from pathlib import Path
import traceback
import re
import json
from groq import Groq

# Load environment
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ✅ Import الدوال الصحيحة بالـ parameters الصح
try:
    try:
        from scr.clean_text import clean_text
    except ImportError:
        from clean_text import clean_text
    print("✅ Imported clean_text")
except Exception as e:
    print(f"⚠️  Could not import clean_text: {e}")
    # Fallback implementation
    def clean_text(text):
        text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,?!\-–—/\n]+", "", text)
        text = re.sub(r"\n+", "\n", text)
        return text.strip()

try:
    try:
        from scr.chunk_text import chunk_text
    except ImportError:
        from chunk_text import chunk_text
    print("✅ Imported chunk_text")
except Exception as e:
    print(f"⚠️  Could not import chunk_text: {e}")
    # Fallback implementation
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    def chunk_text(text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        return splitter.split_text(text)

try:
    try:
        from scr.embedding import embed_single_file
    except ImportError:
        from embedding import embed_single_file
    print("✅ Imported embed_single_file")
except Exception as e:
    print(f"⚠️  Could not import embed_single_file: {e}")
    print(f"⚠️  Make sure embedding.py has the embed_single_file function!")
    raise Exception("embed_single_file function is required but not found")


# ======================================================
#  Extract text from PDF
# ======================================================
def extract_pdf_text(pdf_path):
    """استخراج النص من PDF"""
    try:
        text = ""
        with open(pdf_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            
            # Check if encrypted
            if reader.is_encrypted:
                try:
                    reader.decrypt('')
                except:
                    raise Exception("PDF is encrypted")
            
            # Extract from all pages
            total_pages = len(reader.pages)
            print(f"   📄 Total pages: {total_pages}")
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    print(f"   ⚠️  Error on page {page_num + 1}: {e}")
                    continue
        
        return text
    
    except Exception as e:
        print(f"❌ Error extracting PDF: {e}")
        raise


# ======================================================
#  Save chunks to file
# ======================================================
def save_chunks_to_file(chunks, pdf_filename, subject_name):
    """
    حفظ الـ chunks في ملف بنفس صيغة الملفات الموجودة
    """
    CHUNKS_FOLDER = "chunks"
    
    # Create folder if not exists
    os.makedirs(CHUNKS_FOLDER, exist_ok=True)
    
    # Create filename: SubjectName1.txt (same format as existing files)
    pdf_name = Path(pdf_filename).stem
    match = re.search(r"(\d+)", pdf_name)
    number = match.group(1) if match else "1"
    
    chunk_filename = f"{subject_name}{number}.txt"
    chunk_filepath = os.path.join(CHUNKS_FOLDER, chunk_filename)
    
    # Save chunks with separator ---CHUNK---
    with open(chunk_filepath, "w", encoding="utf-8") as f:
        f.write("---CHUNK---\n".join(chunks))
    
    print(f"   💾 Saved to: {chunk_filepath}")
    
    return chunk_filename  # نرجع اسم الملف فقط


# ======================================================
#  Generate Suggestions (Llama 3)
# ======================================================
def generate_key_questions(text):
    """Generate 5 key conceptual questions from the text using Groq"""
    if not groq_client:
        return []
    
    try:
        # Limit context to first 6000 chars to capture intro/overview
        context_preview = text[:6000]
        
        prompt = f"""
        Analyze the following lecture content and generate 5 short, important questions that a student should be able to answer after studying this.
        Focus on key concepts (e.g., definitions, comparisons, main ideas).
        
        Return the output strictly as a JSON object with a key "suggestions" containing a list of strings.
        Example: {{ "suggestions": ["What is X?", "Explain Y vs Z"] }}

        Content:
        {context_preview}
        """

        completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        response_content = completion.choices[0].message.content
        data = json.loads(response_content)
        return data.get("suggestions", [])
        
    except Exception as e:
        print(f"⚠️ Error generating suggestions: {e}")
        return []

# ======================================================
#  Main Process Function
# ======================================================
def process_new_pdf(pdf_path, subject_name):
    """
    معالجة PDF كامل باستخدام الدوال الموجودة
    
    Args:
        pdf_path: المسار الكامل للـ PDF
        subject_name: اسم المادة
    
    Returns:
        dict: {
            'success': bool,
            'total_chunks': int,
            'total_characters': int,
            'error': str (optional)
        }
    """
    
    try:
        filename = Path(pdf_path).name
        print(f"\n{'='*60}")
        print(f"🚀 Processing PDF")
        print(f"{'='*60}")
        print(f"📄 File: {filename}")
        print(f"📚 Subject: {subject_name}")
        print(f"📂 Path: {pdf_path}")
        print(f"{'='*60}\n")
        
        # Validate file
        if not os.path.exists(pdf_path):
            raise Exception(f"File not found: {pdf_path}")
        
        file_size = os.path.getsize(pdf_path)
        print(f"📦 File size: {file_size / 1024:.2f} KB")
        
        if file_size == 0:
            raise Exception("File is empty")
        
        # Step 1: Extract text from PDF
        print("📄 Extracting text from PDF...")
        raw_text = extract_pdf_text(pdf_path)
        
        if not raw_text or len(raw_text.strip()) < 50:
            raise Exception("No readable text found in PDF")
        
        print(f"   ✓ Extracted {len(raw_text)} characters")
        
        # Step 2: Clean text using clean_text(text)
        print("\n🧹 Cleaning text...")
        cleaned_text = clean_text(raw_text)  # ← بتاخد text parameter واحد بس
        print(f"   ✓ Cleaned: {len(cleaned_text)} characters")
        
        if len(cleaned_text) < 50:
            raise Exception("Cleaned text too short")
        
        # Step 3: Chunk text using chunk_text(text)
        print("\n✂️  Chunking text...")
        chunks = chunk_text(cleaned_text)  # ← بتاخد text parameter واحد بس
        print(f"   ✓ Created {len(chunks)} chunks")
        
        if not chunks or len(chunks) == 0:
            raise Exception("No chunks created")
        
        # Preview first chunk
        if chunks:
            preview = chunks[0][:100] + "..." if len(chunks[0]) > 100 else chunks[0]
            print(f"   📝 First chunk preview: {preview}")
        
        # Step 4: Save chunks to file
        print("\n💾 Saving chunks to file...")
        chunk_filename = save_chunks_to_file(chunks, filename, subject_name)
        
        # Step 5: Embed and upload using embed_single_file(chunk_filename)
        print("\n🔼 Creating embeddings and uploading to Qdrant...")
        result = embed_single_file(chunk_filename)  # ← بتاخد filename parameter واحد بس
        
        if not result or not result.get('success'):
            raise Exception(result.get('error', 'Upload failed'))
        
        # Step 6: Generate Suggestions
        print("\n💡 Generating key questions...")
        suggestions = generate_key_questions(cleaned_text)
        print(f"   ✓ Generated {len(suggestions)} suggestions")

        print(f"\n{'='*60}")
        print(f"✅ Successfully processed {filename}")
        print(f"{'='*60}")
        print(f"📊 Total chunks: {result['total_chunks']}")
        print(f"📏 Total characters: {len(cleaned_text)}")
        print(f"{'='*60}\n")
        
        return {
            'success': True,
            'total_chunks': result['total_chunks'],
            'total_characters': len(cleaned_text),
            'suggestions': suggestions
        }
    
    except Exception as e:
        error_msg = str(e)
        print(f"\n{'='*60}")
        print(f"❌ ERROR PROCESSING PDF")
        print(f"{'='*60}")
        print(f"Error: {error_msg}")
        print(f"{'='*60}\n")
        
        traceback.print_exc()
        
        return {
            'success': False,
            'error': error_msg,
            'total_chunks': 0,
            'total_characters': 0
        }


# ======================================================
#  Test
# ======================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_pdf = sys.argv[1]
        test_subject = sys.argv[2] if len(sys.argv) > 2 else "Test"
    else:
        test_pdf = r"C:\Users\DOWN TOWN  H\project\lectures\test.pdf"
        test_subject = "Mathematics"
    
    if os.path.exists(test_pdf):
        result = process_new_pdf(test_pdf, test_subject)
        print(f"\n📊 Final Result: {result}")
    else:
        print(f"❌ File not found: {test_pdf}")
        print(f"\nUsage: python scr/process_pdf.py <pdf_path> [subject]")