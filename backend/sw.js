// backend/sw.js

const CACHE_NAME = 'university-ai-cache-v3'; // Increment version on change
const urlsToCache = [
    // App Shell
    '/chat.html',
    '/login.html',
    '/index.html',

    // Static Assets
    '/static/js/cache.js',
    '/static/ch.png',

    // Key CDN resources (optional, might fail if CORS is restrictive)
    'https://cdn.jsdelivr.net/npm/marked/marked.min.js',
    'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js',
    'https://unpkg.com/dexie/dist/dexie.js'
];

// Install the service worker and cache the app shell
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                console.log('SW: Caching app shell');
                // Use addAll for atomic operation. Use individual add with catch for non-critical assets.
                const cachePromises = urlsToCache.map(url => {
                    return cache.add(url).catch(err => {
                        console.warn(`SW: Failed to cache ${url}`, err);
                    });
                });
                return Promise.all(cachePromises);
            })
    );
});

// Activate event to clean up old caches
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    if (cacheName !== CACHE_NAME) {
                        console.log('SW: Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});


// Intercept fetch requests
self.addEventListener('fetch', event => {
    // For API calls, always go to the network.
    if (event.request.url.includes('/student/') || event.request.url.includes('/auth/')) {
        event.respondWith(fetch(event.request));
        return;
    }

    // For other requests (app shell, assets), use a Cache-First strategy.
    event.respondWith(
        caches.match(event.request)
            .then(response => response || fetch(event.request))
    );
});