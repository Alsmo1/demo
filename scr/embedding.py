from dotenv import load_dotenv
import os
import re
import uuid
import time
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct

# === Load ENV ===
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# === Paths ===
CHUNKS_FOLDER = "chunks"
COLLECTION_NAME = "student_materials"

# === Load embedding model ===
print("Loading E5-Large model...")
model = SentenceTransformer("intfloat/e5-large")

# === Connect to Qdrant ===
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60  # مهم عشان يمنع فصل الاتصال
)

from qdrant_client.models import Distance

try:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE
            )
        )
except Exception as e:
    print(f"⚠️ Qdrant Connection Error during import: {e}")


# ======================================================
#  Extract metadata from filename
# ======================================================
def extract_metadata(filename):
    name = filename.replace(".txt", "")
    match = re.search(r"(\d+)", name)
    sheet_number = int(match.group(1)) if match else None
    course_name = name[:match.start()].strip() if match else name
    return course_name, sheet_number

# ======================================================
#  Read chunks
# ======================================================
def read_chunks_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_chunks = content.split("---CHUNK---")
    cleaned_chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 20]
    return cleaned_chunks

# ======================================================
#  Process single file (NEW - للملفات الجديدة)
# ======================================================
def embed_single_file(chunk_filename, batch_size=10, retry_times=5):
    """
    معالجة ملف واحد محدد بدلاً من كل الملفات
    
    Args:
        chunk_filename: اسم الملف فقط (مثل: Mathematics1.txt)
        batch_size: حجم الـ batch
        retry_times: عدد المحاولات
    
    Returns:
        dict: {'success': bool, 'total_chunks': int}
    """
    filepath = os.path.join(CHUNKS_FOLDER, chunk_filename)
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return {'success': False, 'total_chunks': 0}
    
    chunks = read_chunks_from_file(filepath)
    course_name, sheet_number = extract_metadata(chunk_filename)
    
    print(f"\n📌 File: {chunk_filename} | Chunks: {len(chunks)}")
    print(f"    Course: {course_name} | Sheet: {sheet_number}")
    
    uploaded_count = 0
    
    # تقسيم إلى batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        # Embed
        vectors = model.encode(batch).tolist()
        
        # Prepare points
        points = []
        for vec, chunk in zip(vectors, batch):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "text": chunk,
                        "filename": chunk_filename,
                        "course": course_name,
                        "sheet_number": sheet_number
                    }
                )
            )
        
        # Upsert with retry
        for attempt in range(retry_times):
            try:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                uploaded_count += len(batch)
                print(f"   → Uploaded batch {i//batch_size + 1}")
                break
            
            except Exception as e:
                print(f"⚠ خطأ في الاتصال! محاولة {attempt+1}/{retry_times}")
                print(e)
                time.sleep(3)
                
                if attempt == retry_times - 1:
                    print("❌ فشل نهائي في رفع هذا batch")
                    return {'success': False, 'total_chunks': uploaded_count}
        
        time.sleep(0.5)
    
    print(f"\n🔥 Uploaded {uploaded_count} chunks successfully!")
    
    return {'success': True, 'total_chunks': uploaded_count}


# ======================================================
#  Batched embedding + retries (الدالة الأصلية لكل الملفات)
# ======================================================
def embed_chunks_and_upload(batch_size=10, retry_times=5):
    files = [f for f in os.listdir(CHUNKS_FOLDER) if f.endswith(".txt")]
    print(f"Found {len(files)} chunk files.\n")

    for filename in files:

        filepath = os.path.join(CHUNKS_FOLDER, filename)
        chunks = read_chunks_from_file(filepath)
        course_name, sheet_number = extract_metadata(filename)

        print(f"\n📌 File: {filename} | Chunks: {len(chunks)}")
        print(f"    Course: {course_name} | Sheet: {sheet_number}")

        # تقسيم الـ chunks إلى batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]

            # Embed batch
            vectors = model.encode(batch).tolist()

            # Prepare points
            points = []
            for vec, chunk in zip(vectors, batch):
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec,
                        payload={
                            "text": chunk,
                            "filename": filename,
                            "course": course_name,
                            "sheet_number": sheet_number
                        }
                    )
                )

            # Upsert with retry handling
            for attempt in range(retry_times):
                try:
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points
                    )
                    print(f"   → Uploaded batch {i//batch_size + 1}")
                    break

                except Exception as e:
                    print(f"⚠ خطأ في الاتصال! محاولة {attempt+1}/{retry_times}")
                    print(e)

                    time.sleep(3)

                    if attempt == retry_times - 1:
                        print("❌ فشل نهائي في رفع هذا batch، بنتخطّاه...")
            
            time.sleep(0.5)  # منع الضغط على السيرفر

    print("\n🔥 All chunks uploaded successfully with batching + retry!")

# ======================================================
#  Run
# ======================================================
if __name__ == "__main__":
    embed_chunks_and_upload()
    print("\n🎉 Done!")