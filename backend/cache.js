/* 
  Edge Caching Strategy (IndexedDB)
  Dependencies: Dexie.js (https://unpkg.com/dexie/dist/dexie.js)
*/

// Initialize Dexie Database
const db = new Dexie('UniversityAICache');

// Define Schema
db.version(1).stores({
  queries: '++id, hash_key, question, answer, timestamp'
});

const EdgeCache = {
    /**
     * Generates a SHA-256 hash for the question to use as a key.
     */
    async generateHash(text) {
        const msgBuffer = new TextEncoder().encode(text.trim().toLowerCase());
        const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
        return Array.from(new Uint8Array(hashBuffer))
            .map(b => b.toString(16).padStart(2, '0')).join('');
    },

    /**
     * Checks if the question exists in the cache.
     * Returns the cached object or null.
     */
    async get(question) {
        try {
            const hash = await this.generateHash(question);
            const cached = await db.queries.where('hash_key').equals(hash).first();
            
            if (cached) {
                console.log(`⚡ Edge Cache Hit: "${question.substring(0, 20)}..."`);
                return cached;
            }
        } catch (e) {
            console.error("Cache Read Error:", e);
        }
        return null;
    },

    /**
     * Retrieves all cached items for offline search.
     */
    async getAll() {
        try {
            return await db.queries.toArray();
        } catch (e) {
            console.error("Cache Read Error:", e);
            return [];
        }
    },

    /**
     * Saves the answer to the cache.
     */
    async set(question, answer) {
        try {
            if (!answer || answer.length < 5) return; // Don't cache empty/short answers
            
            const hash = await this.generateHash(question);
            await db.queries.put({
                hash_key: hash,
                question: question.trim(),
                answer: answer,
                timestamp: Date.now()
            });
            console.log("💾 Saved to Edge Cache");
        } catch (e) {
            console.error("Cache Write Error:", e);
        }
    }
};

// Expose to window for use in chat.html
window.EdgeCache = EdgeCache;