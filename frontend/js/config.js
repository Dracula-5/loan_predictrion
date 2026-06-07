/**
 * Update API_URL to your Render deployment URL before deploying to Netlify.
 * Leave as-is for local development (Flask on port 5000).
 */
const CONFIG = {
  API_URL: (function () {
    const h = window.location.hostname;
    if (h === "localhost" || h === "127.0.0.1") {
      return "http://localhost:5000";
    }
    return "https://YOUR-APP.onrender.com"; // <-- replace after deploying to Render
  })(),
};
