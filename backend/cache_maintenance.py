# backend/cache_maintenance.py
"""
Automated cache maintenance script
Run this periodically (e.g., via cron job)
"""
import asyncio
from semantic_cache import semantic_cache

async def run_maintenance():
    """Run cache maintenance tasks"""
    
    if not semantic_cache:
        print("⚠️ Semantic cache not available")
        return
    
    print("\n" + "="*60)
    print("🔧 Running Cache Maintenance")
    print("="*60 + "\n")
    
    # 1. Invalidate old entries
    print("1️⃣ Invalidating expired entries...")
    deleted = semantic_cache.invalidate_old_entries(max_age_hours=24)
    print(f"   ✅ Removed {deleted} expired entries\n")
    
    # 2. Get current stats
    print("2️⃣ Current cache statistics:")
    if semantic_cache.enable_analytics:
        stats = semantic_cache.analytics.get_stats(days=7)
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   Cache Hits: {stats['cache_hits']}")
        print(f"   Hit Rate: {stats['hit_rate']}%")
        print(f"   Avg Score: {stats['avg_similarity_score']}\n")
    
    # 3. Cache info
    print("3️⃣ Cache collection info:")
    info = semantic_cache.get_cache_info()
    print(f"   Total Entries: {info['total_entries']}")
    print(f"   Threshold: {info['threshold']}")
    print(f"   TTL: {info['ttl_hours']} hours\n")
    
    print("="*60)
    print("✅ Maintenance completed")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(run_maintenance())