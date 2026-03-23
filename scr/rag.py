"""
Simplified Production-Ready RAG System
Compatible with existing project structure
"""

from dotenv import load_dotenv
import os
from typing import List, Dict, Optional, Generator
import logging

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import AsyncGroq
import json
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

missing = []
if not QDRANT_URL:      missing.append("QDRANT_URL")
if not QDRANT_API_KEY:  missing.append("QDRANT_API_KEY")
if not GROQ_API_KEY:    missing.append("GROQ_API_KEY")

if missing:
    raise RuntimeError(
        f"RAG service cannot start. Missing env vars: {missing}\n"
        f"Add them to your .env file."
    )

# ============================================
# CONFIGURATION
# ============================================
COLLECTION_NAME = "student_materials"
RAG_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.80"))

# ============================================
# INITIALIZE CLIENTS
# ============================================

logger.info("🔌 Initializing Qdrant Client...")
try:
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=10
    )
    # Verify connection is actually alive
    client.get_collections()
    logger.info("✅ Qdrant connected successfully")
except Exception as e:
    logger.error(f"❌ Qdrant connection failed: {e}")
    logger.error("RAG search will be unavailable until Qdrant is reachable")
    client = None

logger.info("🤖 Loading Embedding Model (e5-large)...")
try:
    embedder = SentenceTransformer("intfloat/e5-large")
    logger.info("✅ Embedding model loaded")
except Exception as e:
    logger.error(f"❌ Embedding model failed: {e}")
    embedder = None

logger.info("⚡ Initializing Groq Client...")
try:
    groq_client = AsyncGroq(api_key=GROQ_API_KEY)
    logger.info("✅ Groq client initialized")
except Exception as e:
    logger.error(f"❌ Groq initialization failed: {e}")
    groq_client = None

# ============================================
# TRANSLATION LAYER
# ============================================

async def translate_with_groq(text: str, target_lang: str = "en") -> str:
    """Translate using Groq LLM (Primary method)"""
    try:
        detected_lang = detect(text)
        if detected_lang == target_lang:
            return text
        
        lang_names = {'ar': 'Arabic', 'en': 'English'}
        source_name = lang_names.get(detected_lang, detected_lang)
        target_name = lang_names.get(target_lang, target_lang)
        
        prompt = f"""Translate this {source_name} text to {target_name}.
Return ONLY the translation, no explanations.

Text: {text}

Translation:"""
        
        completion = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=300
        )
        
        translated = completion.choices[0].message.content.strip()
        logger.info(f"🌍 Groq Translation: '{text[:50]}...' -> '{translated[:50]}...'")
        return translated
        
    except Exception as e:
        logger.warning(f"⚠️ Groq translation failed: {e}")
        return text


def translate_with_google(text: str, target_lang: str = "en") -> str:
    """Translate using Google Translate (Fallback)"""
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated = translator.translate(text)
        logger.info(f"🌍 Google Translation: '{text[:50]}...' -> '{translated[:50]}...'")
        return translated
    except Exception as e:
        logger.warning(f"⚠️ Google translation failed: {e}")
        return text


async def smart_translate(text: str, target_lang: str = "en") -> str:
    """Smart translation with fallback chain"""
    if not groq_client:
        return text  # return original without modification

    try:
        detected_lang = detect(text)
        if detected_lang == target_lang or detected_lang == "unknown":
            logger.info(f"ℹ️ No translation needed (lang: {detected_lang})")
            return text
        
        logger.info(f"🌍 Translation needed: {detected_lang} -> {target_lang}")
        
        # Try Groq first (faster + more reliable)
        translated = await translate_with_groq(text, target_lang)
        if translated != text:
            return translated
        
        # Fallback to Google
        logger.info("🔄 Falling back to Google Translate...")
        return translate_with_google(text, target_lang)
        
    except LangDetectException:
        logger.warning("⚠️ Language detection failed, using original text")
        return text


# ============================================
# VECTOR SEARCH
# ============================================

def format_chunk(point) -> str:
    """Format search result into readable context"""
    text = point.payload.get("text", "")
    course = point.payload.get("course", "Unknown")
    sheet = point.payload.get("sheet_number", "Unknown")
    return f"[COURSE: {course} | SHEET: {sheet}]\n{text}"


def search_qdrant(query: str) -> tuple:
    """
    Search Qdrant for relevant chunks
    
    Returns:
        (context_string, max_score)
    """
    if not client or not embedder:
        logger.error("❌ Qdrant or Embedder not initialized")
        return ("", 0.0)
    
    try:
        # Enhanced query for better embedding
        enhanced_query = f"query: {query}"
        vec = embedder.encode(enhanced_query).tolist()
        
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            limit=5
        )
        
        if not results.points:
            logger.warning("⚠️ No results found")
            return ("", 0.0)
        
        logger.info(f"📊 Found {len(results.points)} chunks:")
        for i, p in enumerate(results.points, 1):
            logger.info(
                f"  {i}. Score: {p.score:.4f} | "
                f"Course: {p.payload.get('course', 'N/A')}"
            )
        
        chunks = [format_chunk(p) for p in results.points]
        context = "\n\n---\n\n".join(chunks)
        max_score = results.points[0].score
        
        return (context, max_score)
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return ("", 0.0)


# ============================================
# PROMPT BUILDING
# ============================================

def build_prompt(question: str, context: str, history: List[Dict] = None) -> str:
    """Build system prompt for LLM"""
    history_text = ""
    if history:
        limited = history[-6:]  # Last 3 exchanges
        history_text = "CONVERSATION HISTORY:\n" + "\n".join([
            f"- {m['role'].upper()}: {m['content']}"
            for m in limited
        ]) + "\n\n"
    
    return f"""You are a helpful AI teacher for university students.

Rules:
1. Answer ONLY from the context provided below
2. Use simple language with examples
3. Answer in the SAME language as the question
4. If not in context, say: "I couldn't find that in the lecture materials"
5. Bold key terms: **term**
6. Use LaTeX for math: $E=mc^2$
7. Add code examples if relevant

{history_text}Context:
{context}

Question:
{question}

Answer:"""


# ============================================
# CONVERSATION CONTEXT
# ============================================

async def contextualize_query(question: str, history: List[Dict]) -> str:
    """Rephrase question based on conversation history"""
    if not history:
        return question
    if not groq_client:
        return question  # return original without modification
    relevant = history[-6:]
    history_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in relevant
    ])
    
    prompt = f"""Rephrase the follow-up question to be standalone.
Keep the SAME language as the input.
Do NOT answer it.

Chat History:
{history_text}

Follow Up: {question}

Standalone Question:"""
    
    try:
        completion = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=200
        )
        
        new_q = completion.choices[0].message.content.strip()
        new_q = new_q.replace("Standalone Question:", "").strip('"').strip("'")
        
        logger.info(f"🔄 Contextualized: '{question}' -> '{new_q}'")
        return new_q
        
    except Exception as e:
        logger.warning(f"⚠️ Contextualization failed: {e}")
        return question


# ============================================
# MAIN RAG FUNCTIONS (EXPORTED)
# ============================================

async def rag_answer(question: str, history: List[Dict] = None) -> str:
    """
    Generate answer using RAG (Non-streaming)
    
    Args:
        question: User's question
        history: Conversation history (default: [])
    
    Returns:
        Generated answer string
    """
    if history is None:
        history = []
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"💬 Question: {question}")
        logger.info(f"{'='*60}")
        
        # Step 1: Contextualize if there's history
        standalone_q = await contextualize_query(question, history)
        
        # Step 2: Translate for search (if needed)
        search_query = await smart_translate(standalone_q, target_lang='en')
        
        # Step 3: Search Qdrant
        logger.info("🔍 Searching Qdrant...")
        context, score = search_qdrant(search_query)
        
        # Safety checks
        if not groq_client:
            return "⚠️ AI service is currently unavailable. Please try again later."
        
        if not context or score < RAG_THRESHOLD:
            return "I couldn't find that information in the lecture materials. Please try rephrasing your question."
        
        # Step 4: Generate answer (in original language)
        logger.info("🤖 Generating answer...")
        prompt = build_prompt(question, context, history)
        
        completion = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1024
        )
        
        answer = completion.choices[0].message.content
        
        logger.info(f"✅ Answer generated (score: {score:.4f})")
        return answer
        
    except Exception as e:
        logger.error(f"❌ RAG Error: {e}", exc_info=True)
        return "I encountered an error while processing your question. Please try again."


async def rag_answer_stream(question: str, history: List[Dict] = None):
    """
    Generate answer using RAG (Streaming)
    
    Args:
        question: User's question
        history: Conversation history (default: [])
    
    Yields:
        Chunks of the generated answer
        Status messages are prefixed with [STATUS] marker
        Content messages are prefixed with [CONTENT] marker
    """
    if history is None:
        history = []
    
    if not groq_client:
        yield json.dumps({"error": "AI service is currently unavailable. Please try again later."})
        return

    try:
        # Status updates (with special marker for frontend to detect)
        yield "[STATUS]🔍 Analyzing context..."
        
        # Step 1: Contextualize
        standalone_q = await contextualize_query(question, history)
        
        # Step 2: Translate
        search_query = await smart_translate(standalone_q, target_lang='en')
        
        # Indicate translation
        try:
            if detect(question) == 'ar':
                yield "[STATUS]🌍 Translating query..."
        except:
            pass
        
        # Step 3: Search
        yield "[STATUS]📚 Searching lecture materials..."
        context, score = search_qdrant(search_query)
        
        # Safety checks
        if not groq_client:
            yield "[CONTENT]⚠️ AI service unavailable"
            return
        
        if not context or score < RAG_THRESHOLD:
            yield "[CONTENT]I couldn't find that in the lecture materials"
            return
        
        if score < RAG_THRESHOLD + 0.1: # Give a warning if the score is close to the threshold
            yield "[STATUS]⚠️ Low confidence match..."
        
        # Step 4: Stream answer
        yield "[STATUS]💡 Generating answer..."
        
        prompt = build_prompt(question, context, history)
        
        stream = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1024,
            stream=True
        )
        
        # Mark the start of actual content
        yield "[CONTENT]"
        
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
                
    except Exception as e:
        logger.error(f"❌ Stream Error: {e}", exc_info=True)
        yield "[CONTENT]\n\n⚠️ An error occurred. Please try again."


# ============================================
# TEST RUNNER
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("🧪 Testing RAG System")
    print("="*60)
    import asyncio
    
    test_q = input("\n💬 Enter test question: ").strip() or "What is machine learning?"
    
    print(f"\n🔍 Processing: {test_q}\n")
    answer = asyncio.run(rag_answer(test_q))
    
    print("\n" + "="*60)
    print("✅ ANSWER:")
    print("="*60)
    print(answer)
    print("="*60)