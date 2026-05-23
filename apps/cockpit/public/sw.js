// Minimal service worker so the cockpit installs as a PWA.
// Cache strategy will be tuned in Phase 4.5; today we just register.
self.addEventListener("install", (e) => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", () => {});
