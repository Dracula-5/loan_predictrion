"use strict";

const API = {
  async _fetch(endpoint, options = {}) {
    const response = await fetch(CONFIG.API_URL + endpoint, options);
    if (!response.ok) {
      const err = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
      throw new Error(err.error || err.message || `HTTP ${response.status}`);
    }
    return response;
  },

  async json(endpoint, options = {}) {
    const opts = {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    };
    const r = await this._fetch(endpoint, opts);
    return r.json();
  },

  async predict(payload) {
    return this.json("/api/predict", { method: "POST", body: JSON.stringify(payload) });
  },

  async getResult(pid) {
    return this.json(`/api/result/${pid}`);
  },

  async getDashboard() {
    return this.json("/api/dashboard");
  },

  async getMetrics() {
    return this.json("/api/metrics");
  },

  async getHistory(page = 1, perPage = 25) {
    return this.json(`/api/history?page=${page}&per_page=${perPage}`);
  },

  async deleteRecord(pid) {
    return this.json(`/api/history/${pid}`, { method: "DELETE" });
  },

  async clearHistory() {
    return this.json("/api/history/clear", { method: "POST" });
  },

  async batchPredict(formData) {
    const r = await this._fetch("/api/batch", { method: "POST", body: formData });
    return r.json();
  },

  downloadReport(pid) {
    window.open(CONFIG.API_URL + `/api/download/${pid}`, "_blank");
  },

  downloadBatch(filename) {
    window.open(CONFIG.API_URL + `/api/download_batch/${filename}`, "_blank");
  },

  reportUrl(name) {
    return `${CONFIG.API_URL}/api/reports/${name}.png`;
  },
};
