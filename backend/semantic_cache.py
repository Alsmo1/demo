# backend/semantic_cache.py - Advanced Version
import os
import time
import uuid
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import sqlite3

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
CACHE_COLLECTION = "semantic_cache"
ANALYTICS_DB = "cache_analytics.db"

class CacheAnalytics:
    """Track cache performance metrics"""
    
    def __init__(self):
        self.db_path = ANALYTICS_DB
        self._init_db()
    
    def _init_db(self):
        """Initialize analytics database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            CREATE TABLE IF NOT EXISTS cache_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                query TEXT,
                score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        c.execute("""
            CREATE TABLE IF NOT EXISTS cache_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                total_requests INTEGER DEFAULT 0,
                cache_hits INTEGER DEFAULT 0,
                cache_misses INTEGER DEFAULT 0,
                avg_score REAL DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        print("✅ Cache analytics DB initialized")
    
    def log_event(self, event_type: str, query: str = None, score: float = None):
        """Log a cache event"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute(
                "INSERT INTO cache_events (event_type, query, score) VALUES (?, ?, ?)",
                (event_type, query, score)
            )
            
            # Update daily stats
            today = datetime.now().strftime("%Y-%m-%d")
            
            if event_type == "hit":
                c.execute("""
                    INSERT INTO cache_stats (date, total_requests, cache_hits)
                    VALUES (?, 1, 1)
                    ON CONFLICT(date) DO UPDATE SET
                        total_requests = total_requests + 1,
                        cache_hits = cache_hits + 1,
                        updated_at = CURRENT_TIMESTAMP
                """, (today,))
            elif event_type == "miss":
                c.execute("""
                    INSERT INTO cache_stats (date, total_requests, cache_misses)
                    VALUES (?, 1, 1)
                    ON CONFLICT(date) DO UPDATE SET
                        total_requests = total_requests + 1,
                        cache_misses = cache_misses + 1,
                        updated_at = CURRENT_TIMESTAMP
                """, (today,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Error logging cache event: {e}")
    
    def get_stats(self, days: int = 7) -> Dict:
        """Get cache statistics for the last N days"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Overall stats
            c.execute("""
                SELECT 
                    SUM(total_requests) as total_requests,
                    SUM(cache_hits) as total_hits,
                    SUM(cache_misses) as total_misses
                FROM cache_stats
                WHERE date >= date('now', '-' || ? || ' days')
            """, (days,))
            
            overall = c.fetchone()
            
            total_requests = overall['total_requests'] or 0
            total_hits = overall['total_hits'] or 0
            total_misses = overall['total_misses'] or 0
            
            hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
            
            # Daily breakdown
            c.execute("""
                SELECT 
                    date,
                    total_requests,
                    cache_hits,
                    cache_misses,
                    CASE 
                        WHEN total_requests > 0 
                        THEN ROUND(cache_hits * 100.0 / total_requests, 2)
                        ELSE 0 
                    END as hit_rate
                FROM cache_stats
                WHERE date >= date('now', '-' || ? || ' days')
                ORDER BY date DESC
            """, (days,))
            
            daily_stats = [dict(row) for row in c.fetchall()]
            
            # Average scores
            c.execute("""
                SELECT AVG(score) as avg_score
                FROM cache_events
                WHERE event_type = 'hit' 
                AND timestamp >= datetime('now', '-' || ? || ' days')
                AND score IS NOT NULL
            """, (days,))
            
            avg_score_row = c.fetchone()
            avg_score = avg_score_row['avg_score'] or 0.0
            
            conn.close()
            
            return {
                "period_days": days,
                "total_requests": total_requests,
                "cache_hits": total_hits,
                "cache_misses": total_misses,
                "hit_rate": round(hit_rate, 2),
                "avg_similarity_score": round(avg_score, 4),
                "daily_breakdown": daily_stats
            }
            
        except Exception as e:
            print(f"⚠️ Error getting cache stats: {e}")
            return {
                "error": str(e),
                "total_requests": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "hit_rate": 0
            }
    
    def get_top_queries(self, limit: int = 10) -> List[Dict]:
        """Get most frequently cached queries"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            c.execute("""
                SELECT 
                    query,
                    COUNT(*) as hit_count,
                    AVG(score) as avg_score,
                    MAX(timestamp) as last_hit
                FROM cache_events
                WHERE event_type = 'hit' AND query IS NOT NULL
                GROUP BY query
                ORDER BY hit_count DESC
                LIMIT ?
            """, (limit,))
            
            results = [dict(row) for row in c.fetchall()]
            conn.close()
            
            return results
            
        except Exception as e:
            print(f"⚠️ Error getting top queries: {e}")
            return []


class SemanticCache:
    def __init__(self, threshold=0.90, ttl_hours=24, enable_analytics=True):
        """
        Initialize Semantic Cache with advanced features
        
        Args:
            threshold: Similarity threshold (0.90 = 90%)
            ttl_hours: Time to live in hours
            enable_analytics: Enable performance tracking
        """
        self.threshold = threshold
        self.ttl_seconds = ttl_hours * 3600
        self.enable_analytics = enable_analytics
        
        try:
            self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            self.client.get_collections()
            print("✅ SemanticCache Qdrant connected")
        except Exception as e:
            print(f"❌ SemanticCache Qdrant failed: {e}")
            self.client = None
        
        # Initialize Embedding Model
        print("⚡ Loading Semantic Cache Model (e5-large)...")
        self.model = SentenceTransformer('intfloat/e5-large')
        
        # Initialize Analytics
        if self.enable_analytics:
            self.analytics = CacheAnalytics()
        
        # Ensure collection exists
        self._init_collection()
        
        print(f"✅ Semantic Cache initialized (threshold={threshold}, ttl={ttl_hours}h)")

    def _init_collection(self):
        """Initialize Qdrant collection"""
        try:
            if not self.client.collection_exists(CACHE_COLLECTION):
                self.client.create_collection(
                    collection_name=CACHE_COLLECTION,
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
                )
                print(f"✅ Created cache collection: {CACHE_COLLECTION}")
        except Exception as e:
            print(f"⚠️ Cache collection init error: {e}")

    def get_cached_response(self, query_text: str) -> Optional[str]:
        """Search for semantically similar question in cache"""
        if not self.client:
            return None  # cache miss — proceed to RAG normally

        try:
            vector = self.model.encode(query_text).tolist()
            
            results = self.client.query_points(
                collection_name=CACHE_COLLECTION,
                query=vector,
                limit=1,
                score_threshold=self.threshold
            ).points
            
            if results:
                result = results[0]
                timestamp = result.payload.get("timestamp", 0)
                
                # Check TTL
                age = time.time() - timestamp
                if age < self.ttl_seconds:
                    response = result.payload.get("response")
                    score = result.score
                    
                    # Log analytics
                    if self.enable_analytics:
                        self.analytics.log_event("hit", query_text, score)
                    
                    print(f"⚡ CACHE HIT (Score: {score:.4f}, Age: {age/3600:.1f}h)")
                    return response
                else:
                    # Expired - delete it
                    print(f"⏰ Cache expired (Age: {age/3600:.1f}h), deleting...")
                    self._delete_point(result.id)
            
            # Log miss
            if self.enable_analytics:
                self.analytics.log_event("miss", query_text)
            
            print("❌ CACHE MISS")
            return None
            
        except Exception as e:
            print(f"⚠️ Cache lookup failed: {e}")
            if self.enable_analytics:
                self.analytics.log_event("error", query_text)
            return None

    def set_cached_response(self, query_text: str, response_text: str, metadata: Dict = None):
        """Store question and answer in cache"""
        if not self.client:
            return None  # cache miss — proceed to RAG normally

        try:
            vector = self.model.encode(query_text).tolist()
            
            payload = {
                "query": query_text,
                "response": response_text,
                "timestamp": time.time(),
                "created_at": datetime.now().isoformat()
            }
            
            # Add optional metadata
            if metadata:
                payload.update(metadata)
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
            
            self.client.upsert(
                collection_name=CACHE_COLLECTION,
                points=[point]
            )
            
            # Log save event
            if self.enable_analytics:
                self.analytics.log_event("save", query_text)
            
            print("💾 Saved to semantic cache")
            
        except Exception as e:
            print(f"⚠️ Cache write failed: {e}")

    def _delete_point(self, point_id: str):
        """Delete a single point from cache"""
        try:
            self.client.delete(
                collection_name=CACHE_COLLECTION,
                points_selector=[point_id]
            )
        except Exception as e:
            print(f"⚠️ Failed to delete cache point: {e}")

    def clear_cache(self):
        """Clear entire cache"""
        try:
            self.client.delete_collection(CACHE_COLLECTION)
            self._init_collection()
            
            if self.enable_analytics:
                self.analytics.log_event("clear_all")
            
            print("🗑️ Cache cleared")
            
        except Exception as e:
            print(f"⚠️ Failed to clear cache: {e}")

    def invalidate_old_entries(self, max_age_hours: int = None):
        """Smart cache invalidation - remove expired entries"""
        if max_age_hours is None:
            max_age_hours = self.ttl_seconds / 3600
        
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            
            # Scroll through all points
            offset = None
            deleted_count = 0
            
            while True:
                points, offset = self.client.scroll(
                    collection_name=CACHE_COLLECTION,
                    limit=100,
                    offset=offset
                )
                
                if not points:
                    break
                
                # Find expired points
                expired_ids = [
                    p.id for p in points 
                    if p.payload.get("timestamp", 0) < cutoff_time
                ]
                
                if expired_ids:
                    self.client.delete(
                        collection_name=CACHE_COLLECTION,
                        points_selector=expired_ids
                    )
                    deleted_count += len(expired_ids)
                
                if offset is None:
                    break
            
            if self.enable_analytics:
                self.analytics.log_event("invalidate", f"Removed {deleted_count} entries")
            
            print(f"🧹 Invalidated {deleted_count} expired cache entries")
            return deleted_count
            
        except Exception as e:
            print(f"⚠️ Cache invalidation failed: {e}")
            return 0

    def warm_cache(self, qa_pairs: List[Dict[str, str]]):
        """
        Warm cache with common questions
        
        Args:
            qa_pairs: List of {"question": "...", "answer": "..."}
        """
        try:
            print(f"\n🔥 Warming cache with {len(qa_pairs)} Q&A pairs...")
            
            points = []
            for qa in qa_pairs:
                question = qa.get("question", "")
                answer = qa.get("answer", "")
                
                if not question or not answer:
                    continue
                
                vector = self.model.encode(question).tolist()
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "query": question,
                        "response": answer,
                        "timestamp": time.time(),
                        "created_at": datetime.now().isoformat(),
                        "warmed": True  # Flag to identify pre-warmed entries
                    }
                )
                points.append(point)
            
            # Batch upsert
            self.client.upsert(
                collection_name=CACHE_COLLECTION,
                points=points
            )
            
            if self.enable_analytics:
                self.analytics.log_event("warm", f"Added {len(points)} entries")
            
            print(f"✅ Cache warmed with {len(points)} entries")
            return len(points)
            
        except Exception as e:
            print(f"⚠️ Cache warming failed: {e}")
            return 0

    def get_cache_info(self) -> Dict:
        """Get current cache information"""
        try:
            collection_info = self.client.get_collection(CACHE_COLLECTION)
            
            return {
                "collection_name": CACHE_COLLECTION,
                "total_entries": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "threshold": self.threshold,
                "ttl_hours": self.ttl_seconds / 3600,
                "model": "intfloat/e5-large",
                "analytics_enabled": self.enable_analytics
            }
        except Exception as e:
            print(f"⚠️ Error getting cache info: {e}")
            return {"error": str(e)}

    def search_cache(self, query: str, limit: int = 5) -> List[Dict]:
        """Search cache and return top N similar entries"""
        try:
            vector = self.model.encode(query).tolist()
            
            results = self.client.query_points(
                collection_name=CACHE_COLLECTION,
                query=vector,
                limit=limit
            ).points
            
            return [
                {
                    "query": r.payload.get("query"),
                    "response": r.payload.get("response", "")[:200] + "...",  # Preview
                    "score": r.score,
                    "age_hours": (time.time() - r.payload.get("timestamp", 0)) / 3600,
                    "created_at": r.payload.get("created_at")
                }
                for r in results
            ]
            
        except Exception as e:
            print(f"⚠️ Cache search failed: {e}")
            return []


# Initialize Singleton
semantic_cache = None
if not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY"):
    print("⚠️ SemanticCache: QDRANT_URL or QDRANT_API_KEY not set. Cache disabled.")
    semantic_cache = None
else:
    try:
        semantic_cache = SemanticCache(
            threshold=0.90,
            ttl_hours=24,
            enable_analytics=True
        )
    except Exception as e:
        print(f"⚠️ SemanticCache initialization failed: {e}")
        semantic_cache = None