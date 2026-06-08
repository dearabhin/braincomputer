// Thin client for the BrainComputer API. No framework — plain fetch.

const BC = {
  async analyze(file) {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${window.BC_API_BASE}/api/analyze`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) {
      const msg = await res.json().catch(() => ({}));
      throw new Error(msg.error || `upload failed (${res.status})`);
    }
    return res.json(); // { job_id, status, modality }
  },

  async getJob(jobId) {
    const res = await fetch(`${window.BC_API_BASE}/api/jobs/${jobId}`);
    if (!res.ok) throw new Error(`job fetch failed (${res.status})`);
    return res.json();
  },

  // Poll until the job finishes (status done/error) or times out.
  async pollJob(jobId, { intervalMs = 2500, timeoutMs = 240000, onTick } = {}) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const data = await this.getJob(jobId);
      if (onTick) onTick(data);
      if (data.status === "done" || data.status === "error") return data;
      await new Promise((r) => setTimeout(r, intervalMs));
    }
    throw new Error("analysis timed out");
  },
};

window.BC = BC;
