// API base URL. Overridden per-environment:
//  - local dev:   window.BC_API_BASE set below to localhost
//  - production:  set to https://api.braincomputer.in at deploy (or via a build step)
// A <meta name="bc-api-base"> tag, if present, takes precedence.
(function () {
  const meta = document.querySelector('meta[name="bc-api-base"]');
  const fromMeta = meta && meta.content;
  const isLocal = ["localhost", "127.0.0.1"].includes(location.hostname);
  window.BC_API_BASE =
    fromMeta || (isLocal ? "http://localhost:8000" : "https://api.braincomputer.in");
})();
